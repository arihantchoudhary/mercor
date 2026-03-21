#!/usr/bin/env python3
"""
Quick MMLU evaluation script.

Downloads a subset of MMLU from the HuggingFace datasets-server API,
evaluates using forced single-token extraction (same approach as improve/infer.py),
and saves results to eval_runner/results/mmlu_results.json.

Strategy for data download:
  1. Try cais/mmlu with per-subject configs (config=abstract_algebra, etc.)
  2. Fallback: try cais/mmlu config=all and filter by 'subject' field
  3. Fallback: try tasksource/mmlu, hails/mmlu_no_train with per-subject configs

Usage:
    python eval_runner/run_mmlu_quick.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required.  pip install requests")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
REQUEST_TIMEOUT = 300

HF_API = "https://datasets-server.huggingface.co/rows"

# MMLU subjects to evaluate
TARGET_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_geography",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "human_aging",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# We want at least 3 subjects, 50 each; aim for more if time allows
MAX_SUBJECTS = 5
EXAMPLES_PER_SUBJECT = 50
HF_DOWNLOAD_TIMEOUT = 30  # shorter timeout to avoid hanging on rate limits

LABELS = ["A", "B", "C", "D"]

# Persistent session
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=10
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Row parsing helper
# ---------------------------------------------------------------------------

def _parse_row(row: dict, subject: str) -> dict | None:
    """Parse a single HuggingFace row into our standard format. Returns None on failure."""
    question = row.get("question", "")
    if not question:
        return None

    # Get choices -- try several field names
    choices = None
    for key in ["choices", "options"]:
        if key in row and row[key]:
            choices = row[key]
            if isinstance(choices, str):
                choices = json.loads(choices)
            break

    if not choices:
        # Try A/B/C/D fields
        choices = []
        for letter in LABELS:
            val = row.get(letter, row.get(letter.lower(), ""))
            if val:
                choices.append(val)

    if not choices or len(choices) != 4:
        return None

    # Get answer -- try several field names
    answer = row.get("answer", row.get("target", ""))
    if isinstance(answer, int):
        gold_idx = answer
    elif isinstance(answer, str):
        a = answer.strip().upper()
        if a in LABELS:
            gold_idx = LABELS.index(a)
        elif a.isdigit():
            gold_idx = int(a)
        else:
            return None
    else:
        return None

    if not (0 <= gold_idx <= 3):
        return None

    return {
        "question": question,
        "choices": choices,
        "gold_idx": gold_idx,
        "subject": subject,
    }


# ---------------------------------------------------------------------------
# Download MMLU data from HuggingFace
# ---------------------------------------------------------------------------

def _hf_fetch(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict] | None:
    """Raw HF datasets-server fetch. Returns list of row dicts, or None."""
    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": min(length, 100),
    }
    try:
        resp = _get_session().get(HF_API, params=params, timeout=HF_DOWNLOAD_TIMEOUT)
        if resp.status_code != 200:
            return None
        data = resp.json()
        rows = data.get("rows", [])
        if not rows:
            return None
        return [r["row"] for r in rows]
    except Exception:
        return None


def fetch_mmlu_subject(subject: str, n: int = 50) -> list[dict] | None:
    """Try multiple dataset sources to fetch MMLU data for a specific subject."""

    # Strategy 1: cais/mmlu with per-subject config
    for split in ["test", "validation"]:
        rows = _hf_fetch("cais/mmlu", subject, split, 0, n)
        if rows:
            parsed = [_parse_row(r, subject) for r in rows]
            parsed = [p for p in parsed if p is not None]
            if parsed:
                return parsed[:n]

    # Strategy 2: tasksource/mmlu with per-subject config
    for split in ["test", "validation"]:
        rows = _hf_fetch("tasksource/mmlu", subject, split, 0, n)
        if rows:
            parsed = [_parse_row(r, subject) for r in rows]
            parsed = [p for p in parsed if p is not None]
            if parsed:
                return parsed[:n]

    # Strategy 3: hails/mmlu_no_train with per-subject config
    for split in ["test", "validation"]:
        rows = _hf_fetch("hails/mmlu_no_train", subject, split, 0, n)
        if rows:
            parsed = [_parse_row(r, subject) for r in rows]
            parsed = [p for p in parsed if p is not None]
            if parsed:
                return parsed[:n]

    return None


def fetch_mmlu_from_all_config(subjects: list[str], n_per_subject: int = 50) -> dict[str, list[dict]]:
    """Fallback: fetch from cais/mmlu config=all and filter by subject field.

    The 'all' config has ~14K test rows. We fetch in batches and group by subject.
    """
    print("  [fallback] Fetching from cais/mmlu config=all and grouping by subject...")
    wanted = set(subjects)
    result: dict[str, list[dict]] = {}
    offset = 0
    batch_size = 100
    max_fetches = 50  # up to 5000 rows

    for _ in range(max_fetches):
        if all(len(result.get(s, [])) >= n_per_subject for s in wanted if s in result) and len(result) >= len(wanted):
            break

        rows = _hf_fetch("cais/mmlu", "all", "test", offset, batch_size)
        if not rows:
            break

        for row in rows:
            subj = row.get("subject", "")
            if subj not in wanted:
                continue
            if len(result.get(subj, [])) >= n_per_subject:
                continue
            parsed = _parse_row(row, subj)
            if parsed:
                result.setdefault(subj, []).append(parsed)

        offset += batch_size
        time.sleep(0.2)

        # Check if we have enough
        filled = sum(1 for s in wanted if len(result.get(s, [])) >= n_per_subject)
        if filled >= MAX_SUBJECTS:
            break

    return result


# ---------------------------------------------------------------------------
# Ollama generation
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str) -> str:
    """Send a forced single-token generation request to Ollama."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 1,
        },
    }
    try:
        session = _get_session()
        resp = session.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.RequestException as exc:
        print(f"  [WARN] Ollama request failed: {exc}")
        return ""


def extract_answer(raw: str) -> str:
    """Extract answer letter from model response."""
    text = raw.strip().upper()
    for ch in text:
        if ch in "ABCD":
            return ch
    return "A"  # fallback


# ---------------------------------------------------------------------------
# Build MMLU prompt (same forced single-token approach as infer.py)
# ---------------------------------------------------------------------------

def build_mmlu_prompt(example: dict) -> str:
    """Format an MMLU question with forced single-token answer."""
    q = example["question"]
    choices = example["choices"]
    return (
        f"Question: {q}\n"
        f"A) {choices[0]}\n"
        f"B) {choices[1]}\n"
        f"C) {choices[2]}\n"
        f"D) {choices[3]}\n\n"
        f"The answer is"
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_subject(examples: list[dict], subject: str) -> list[dict]:
    """Evaluate all examples for a single MMLU subject."""
    results = []
    for i, ex in enumerate(examples):
        prompt = build_mmlu_prompt(ex)

        t0 = time.time()
        raw_response = ollama_generate(prompt)
        latency = time.time() - t0

        predicted = extract_answer(raw_response)
        gold = LABELS[ex["gold_idx"]]
        correct = predicted == gold

        results.append({
            "idx": i,
            "subject": subject,
            "question_preview": ex["question"][:80],
            "gold": gold,
            "gold_idx": ex["gold_idx"],
            "predicted": predicted,
            "correct": correct,
            "raw_response": raw_response.strip(),
            "latency_s": round(latency, 3),
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


def compute_accuracy(results: list[dict]) -> dict:
    """Compute accuracy with 95% Wilson CI."""
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    correct = sum(1 for r in results if r["correct"])
    acc = correct / n

    z = 1.96
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    spread = z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n) / denom
    ci_lower = max(0.0, centre - spread)
    ci_upper = min(1.0, centre + spread)

    return {
        "accuracy": round(acc, 5),
        "correct": correct,
        "total": n,
        "ci_95_wilson": [round(ci_lower, 5), round(ci_upper, 5)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Health check
    try:
        session = _get_session()
        resp = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(f"ERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
                 f"Start the server first: python serve/serve.py (or make serve)")
    except requests.RequestException as exc:
        sys.exit(f"ERROR: Ollama health check failed: {exc}")

    print("=" * 60)
    print("  MMLU Quick Evaluation")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Target: {MAX_SUBJECTS} subjects, {EXAMPLES_PER_SUBJECT} examples each")
    print("=" * 60)

    # Step 1: Download MMLU data
    print("\n[1/3] Downloading MMLU data from HuggingFace...")

    all_examples: dict[str, list[dict]] = {}

    # Try per-subject configs first (most reliable for subject isolation)
    for subject in TARGET_SUBJECTS:
        if len(all_examples) >= MAX_SUBJECTS:
            break
        print(f"  Trying {subject}...", end=" ", flush=True)
        examples = fetch_mmlu_subject(subject, n=EXAMPLES_PER_SUBJECT)
        if examples:
            # Sanity check: verify questions actually match the subject
            # (detect if API is ignoring config and returning same data)
            if all_examples:
                first_existing = list(all_examples.values())[0]
                if (len(examples) > 0 and len(first_existing) > 0
                        and examples[0]["question"] == first_existing[0]["question"]):
                    print(f"SKIP (duplicate of first subject -- API ignoring config)")
                    continue
            all_examples[subject] = examples
            print(f"OK ({len(examples)} examples)")
        else:
            print("SKIP (not available)")
        time.sleep(1.0)  # polite rate limit for HF API

    # Fallback: if per-subject configs didn't work, use 'all' config with subject filtering
    if len(all_examples) < 3:
        remaining_subjects = [s for s in TARGET_SUBJECTS if s not in all_examples]
        needed = MAX_SUBJECTS - len(all_examples)
        batch_result = fetch_mmlu_from_all_config(
            remaining_subjects[:needed * 3],  # try more than needed
            n_per_subject=EXAMPLES_PER_SUBJECT,
        )
        for subj, exs in batch_result.items():
            if len(all_examples) >= MAX_SUBJECTS:
                break
            if subj not in all_examples and len(exs) >= 10:
                all_examples[subj] = exs[:EXAMPLES_PER_SUBJECT]
                print(f"  Got {len(all_examples[subj])} from '{subj}' (all config)")

    total_examples = sum(len(v) for v in all_examples.values())
    print(f"\n  Total: {len(all_examples)} subjects, {total_examples} examples")

    if total_examples == 0:
        sys.exit("ERROR: Could not download any MMLU data. Check network.")

    # Print a quick data sanity check
    print("\n  Data sanity check (first question per subject):")
    for subj, exs in all_examples.items():
        print(f"    {subj}: \"{exs[0]['question'][:70]}...\"")

    # Step 2: Run evaluation
    print(f"\n[2/3] Running evaluation with forced single-token decoding...")
    print(f"       (temperature=0, seed=42, num_predict=1)")

    all_results: list[dict] = []
    subject_metrics: dict[str, dict] = {}

    for subject, examples in all_examples.items():
        print(f"\n  --- {subject} ({len(examples)} examples) ---")
        results = evaluate_subject(examples, subject)
        all_results.extend(results)
        metrics = compute_accuracy(results)
        subject_metrics[subject] = metrics
        print(f"  => Accuracy: {metrics['accuracy']:.4f} "
              f"({metrics['correct']}/{metrics['total']}) "
              f"95% CI: [{metrics['ci_95_wilson'][0]:.4f}, {metrics['ci_95_wilson'][1]:.4f}]")

    # Step 3: Compute overall metrics
    overall_metrics = compute_accuracy(all_results)
    avg_latency = sum(r["latency_s"] for r in all_results) / len(all_results)
    total_time = sum(r["latency_s"] for r in all_results)

    # Print results table
    print("\n" + "=" * 70)
    print("  MMLU RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Subject':<40} {'Acc':>8} {'N':>5} {'CI 95%':>20}")
    print("  " + "-" * 66)
    for subject, metrics in sorted(subject_metrics.items()):
        ci = metrics["ci_95_wilson"]
        print(f"  {subject:<40} {metrics['accuracy']:>8.4f} {metrics['total']:>5} "
              f"  [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("  " + "-" * 66)
    ci = overall_metrics["ci_95_wilson"]
    print(f"  {'OVERALL':<40} {overall_metrics['accuracy']:>8.4f} "
          f"{overall_metrics['total']:>5} "
          f"  [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("=" * 70)
    print(f"  Avg latency: {avg_latency:.3f}s  |  Total time: {total_time:.1f}s")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 70)

    # Step 4: Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results_payload = {
        "benchmark": "mmlu",
        "model": MODEL_NAME,
        "timestamp": ts,
        "overall_metrics": overall_metrics,
        "overall_metrics_extra": {
            "avg_latency_s": round(avg_latency, 3),
            "total_time_s": round(total_time, 1),
        },
        "subject_metrics": subject_metrics,
        "num_subjects": len(subject_metrics),
        "total_examples": overall_metrics["total"],
        "config": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 1,
            "examples_per_subject": EXAMPLES_PER_SUBJECT,
            "prompt_format": "Question: {q}\\nA) ...\\nB) ...\\nC) ...\\nD) ...\\n\\nThe answer is",
        },
        "per_example": all_results,
    }

    out_path = RESULTS_DIR / "mmlu_results.json"
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Also save timestamped copy
    ts_path = RESULTS_DIR / f"mmlu_{ts}.json"
    with open(ts_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"  Timestamped copy: {ts_path}")

    return overall_metrics


if __name__ == "__main__":
    main()
