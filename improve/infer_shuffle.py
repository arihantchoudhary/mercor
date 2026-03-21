#!/usr/bin/env python3
"""
Choice-shuffling inference for HellaSwag evaluation.

For each example, runs K=5 shuffled orderings of the answer choices,
then majority-votes across them. This counters position bias (tendency
to always pick "A").

Computes accuracy with Wilson CI, and runs McNemar's test against the
baseline to check statistical significance.

Usage:
    python improve/infer_shuffle.py --n-examples 200 --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required.  pip install requests")

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
REQUEST_TIMEOUT = 300
LABELS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = (
    "Context: {ctx}\n\n"
    "Which continuation is most natural?\n"
    "A) {end_a}\n"
    "B) {end_b}\n"
    "C) {end_c}\n"
    "D) {end_d}\n\n"
    "The answer is"
)

# Persistent session for connection pooling
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
# Ollama helper
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 1,
) -> str:
    """Send a generation request to Ollama and return the response text."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
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


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    """Extract answer letter from a single-token model response."""
    text = raw.strip().upper()
    for ch in text:
        if ch in "ABCD":
            return ch
    return "A"  # fallback


# ---------------------------------------------------------------------------
# Shuffle inference
# ---------------------------------------------------------------------------

def generate_permutations(k: int, rng: random.Random) -> list[list[int]]:
    """Generate k random permutations of [0, 1, 2, 3]."""
    perms = []
    for _ in range(k):
        p = [0, 1, 2, 3]
        rng.shuffle(p)
        perms.append(p)
    return perms


def run_shuffle(
    examples: list[dict],
    model: str = MODEL_NAME,
    seed: int = 42,
    k: int = 5,
) -> list[dict]:
    """Run K shuffled orderings per example and majority-vote."""
    rng = random.Random(seed)
    results = []

    for i, ex in enumerate(examples):
        t0 = time.time()

        ctx = ex["ctx"]
        endings = ex["endings"]
        gold_idx = ex["label"]
        gold_label = LABELS[gold_idx] if 0 <= gold_idx <= 3 else "?"

        # Generate K permutations for this example
        perms = generate_permutations(k, rng)

        votes: list[int] = []  # votes as original indices
        raw_responses: list[str] = []
        perm_details: list[dict] = []

        for perm_idx, perm in enumerate(perms):
            # Build prompt with shuffled endings
            shuffled_endings = [endings[j] for j in perm]
            prompt = PROMPT_TEMPLATE.format(
                ctx=ctx,
                end_a=shuffled_endings[0].strip(),
                end_b=shuffled_endings[1].strip(),
                end_c=shuffled_endings[2].strip(),
                end_d=shuffled_endings[3].strip(),
            )

            raw = ollama_generate(
                prompt,
                model=model,
                temperature=0,
                seed=42 + perm_idx,
                max_tokens=1,
            )
            raw_responses.append(raw.strip())

            # Extract predicted letter (in shuffled space)
            pred_letter = extract_answer(raw)
            pred_shuffled_pos = LABELS.index(pred_letter)  # 0-3 in shuffled space

            # Map back to original index
            original_idx = perm[pred_shuffled_pos]
            votes.append(original_idx)

            perm_details.append({
                "perm": perm,
                "pred_letter": pred_letter,
                "original_idx": original_idx,
                "raw": raw.strip(),
            })

        # Majority vote across K runs (original indices)
        counter = Counter(votes)
        best_count = counter.most_common(1)[0][1]
        # Tie-break: first vote that has the max count
        final_idx = next(v for v in votes if counter[v] == best_count)
        predicted = LABELS[final_idx]

        latency = time.time() - t0
        correct = predicted == gold_label

        results.append({
            "idx": i,
            "ind": ex.get("ind"),
            "ctx_preview": ctx[:80],
            "gold": gold_label,
            "gold_idx": gold_idx,
            "predicted": predicted,
            "predicted_idx": final_idx,
            "correct": correct,
            "raw_response": " | ".join(raw_responses),
            "votes": votes,
            "perm_details": perm_details,
            "latency_s": round(latency, 3),
            "strategy": "shuffle",
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


# ---------------------------------------------------------------------------
# Accuracy + Confidence Intervals (Wilson score)
# ---------------------------------------------------------------------------

def compute_accuracy(results: list[dict]) -> dict:
    """Compute accuracy with 95% CI (Wilson score interval + bootstrap)."""
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "n": 0, "ci_lower": 0.0, "ci_upper": 0.0}

    correct = sum(1 for r in results if r["correct"])
    acc = correct / n

    # Wilson score interval for 95% CI
    z = 1.96
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    spread = z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n) / denom
    ci_lower_wilson = max(0.0, centre - spread)
    ci_upper_wilson = min(1.0, centre + spread)

    # Bootstrap CI (if numpy available)
    ci_lower_boot = ci_lower_wilson
    ci_upper_boot = ci_upper_wilson
    if np is not None:
        rng_np = np.random.RandomState(42)
        correctness = np.array([1 if r["correct"] else 0 for r in results])
        boot_accs = []
        for _ in range(2000):
            sample = rng_np.choice(correctness, size=n, replace=True)
            boot_accs.append(sample.mean())
        boot_accs_arr = np.array(boot_accs)
        ci_lower_boot = float(np.percentile(boot_accs_arr, 2.5))
        ci_upper_boot = float(np.percentile(boot_accs_arr, 97.5))

    return {
        "accuracy": round(acc, 5),
        "correct": correct,
        "total": n,
        "ci_95_wilson": [round(ci_lower_wilson, 5), round(ci_upper_wilson, 5)],
        "ci_95_bootstrap": [round(ci_lower_boot, 5), round(ci_upper_boot, 5)],
    }


# ---------------------------------------------------------------------------
# McNemar's test vs baseline
# ---------------------------------------------------------------------------

def mcnemar_test(shuffle_results: list[dict], baseline_path: Path) -> dict | None:
    """Run McNemar's test comparing shuffle results to baseline.

    Counts discordant pairs and uses exact binomial test.
    Returns dict with test statistics, or None if baseline not found.
    """
    if not baseline_path.exists():
        print(f"  [WARN] Baseline file not found at {baseline_path}, skipping McNemar test.")
        return None

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    baseline_per_example = baseline_data.get("per_example", [])

    # Build lookup: idx -> correct (bool)
    baseline_correct = {}
    for r in baseline_per_example:
        baseline_correct[r["idx"]] = r["correct"]

    # Count discordant pairs
    # b = baseline wrong, shuffle right (improvement)
    # c = baseline right, shuffle wrong (regression)
    b = 0  # shuffle correct, baseline wrong
    c = 0  # shuffle wrong, baseline correct
    n_both_correct = 0
    n_both_wrong = 0
    n_matched = 0

    for r in shuffle_results:
        idx = r["idx"]
        if idx not in baseline_correct:
            continue
        n_matched += 1
        s_correct = r["correct"]
        b_correct = baseline_correct[idx]

        if s_correct and b_correct:
            n_both_correct += 1
        elif s_correct and not b_correct:
            b += 1
        elif not s_correct and b_correct:
            c += 1
        else:
            n_both_wrong += 1

    if n_matched == 0:
        print("  [WARN] No matched examples for McNemar test.")
        return None

    # Exact binomial test (two-sided)
    # Under H0, b / (b + c) ~ Binomial(b + c, 0.5)
    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    elif scipy_stats is not None:
        # Use scipy for exact binomial test
        result = scipy_stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue
    else:
        # Fallback: manual exact binomial test using math.comb
        # P = 2 * sum_{k=max(b,c)}^{n} C(n,k) * 0.5^n
        k_obs = max(b, c)
        p_tail = 0.0
        for k in range(k_obs, n_discordant + 1):
            p_tail += math.comb(n_discordant, k) * (0.5 ** n_discordant)
        p_value = min(1.0, 2.0 * p_tail)

    mcnemar_result = {
        "n_matched": n_matched,
        "both_correct": n_both_correct,
        "both_wrong": n_both_wrong,
        "shuffle_better": b,
        "baseline_better": c,
        "n_discordant": n_discordant,
        "p_value": round(p_value, 6),
        "significant_at_005": p_value < 0.05,
    }

    return mcnemar_result


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict],
    metrics: dict,
    tag: str,
    config: dict,
    mcnemar: dict | None = None,
) -> Path:
    """Save detailed results and metrics to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{tag}_{ts}.json"
    path = RESULTS_DIR / filename

    payload = {
        "tag": tag,
        "timestamp": ts,
        "config": config,
        "metrics": metrics,
        "per_example": results,
    }
    if mcnemar is not None:
        payload["mcnemar_vs_baseline"] = mcnemar

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Also save a "latest" copy for convenience
    latest_path = RESULTS_DIR / f"{tag}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OLLAMA_BASE_URL, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="HellaSwag choice-shuffling inference with majority voting."
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of shuffled permutations per example")
    parser.add_argument("--baseline-path", type=str, default=None,
                        help="Path to baseline results for McNemar test")

    args = parser.parse_args()

    OLLAMA_BASE_URL = args.base_url
    MODEL_NAME = args.model

    # ---- Health check ----
    try:
        session = _get_session()
        resp = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(f"ERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
                 f"Start the server first.")
    except requests.RequestException as exc:
        sys.exit(f"ERROR: Ollama health check failed: {exc}")

    # ---- Load data ----
    val_path = DATA_DIR / "hellaswag_val.json"
    if not val_path.exists():
        sys.exit(f"ERROR: Validation data not found at {val_path}.\n"
                 f"Run:  python improve/prepare_data.py")

    with open(val_path) as f:
        val_examples = json.load(f)
    print(f"[shuffle] Loaded {len(val_examples)} validation examples.")

    if args.n_examples > 0:
        val_examples = val_examples[: args.n_examples]
        print(f"[shuffle] Limited to {len(val_examples)} examples.")

    # ---- Build config ----
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "n_examples": len(val_examples),
        "k_shuffles": args.k,
        "strategy": "shuffle",
        "temperature": 0,
        "num_predict": 1,
    }

    # ---- Run inference ----
    print(f"\n[shuffle] === CHOICE SHUFFLING (K={args.k} permutations, majority vote) ===")
    t_start = time.time()
    results = run_shuffle(
        val_examples,
        model=args.model,
        seed=args.seed,
        k=args.k,
    )
    total_wall = time.time() - t_start

    # ---- Compute metrics ----
    metrics = compute_accuracy(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    metrics["avg_latency_s"] = round(avg_latency, 3)
    metrics["total_time_s"] = round(total_wall, 1)

    print(f"\n{'='*60}")
    print(f"Strategy:           shuffle (K={args.k})")
    print(f"Accuracy:           {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
    print(f"95% CI (Wilson):    [{metrics['ci_95_wilson'][0]:.4f}, {metrics['ci_95_wilson'][1]:.4f}]")
    print(f"95% CI (Bootstrap): [{metrics['ci_95_bootstrap'][0]:.4f}, {metrics['ci_95_bootstrap'][1]:.4f}]")
    print(f"Avg latency:        {metrics['avg_latency_s']:.3f}s")
    print(f"Total time:         {metrics['total_time_s']:.1f}s")
    print(f"{'='*60}")

    # ---- McNemar test vs baseline ----
    baseline_path = Path(args.baseline_path) if args.baseline_path else RESULTS_DIR / "baseline_latest.json"
    mcnemar = mcnemar_test(results, baseline_path)

    if mcnemar is not None:
        print(f"\n--- McNemar's Test vs Baseline ---")
        print(f"  Matched examples:    {mcnemar['n_matched']}")
        print(f"  Both correct:        {mcnemar['both_correct']}")
        print(f"  Both wrong:          {mcnemar['both_wrong']}")
        print(f"  Shuffle better:      {mcnemar['shuffle_better']}")
        print(f"  Baseline better:     {mcnemar['baseline_better']}")
        print(f"  Discordant pairs:    {mcnemar['n_discordant']}")
        print(f"  p-value:             {mcnemar['p_value']:.6f}")
        print(f"  Significant (p<0.05): {mcnemar['significant_at_005']}")
        print(f"{'='*60}")

    # ---- Save ----
    tag = "shuffle"
    path = save_results(results, metrics, tag, config, mcnemar)
    print(f"\n[shuffle] Results saved to {path}")

    # ---- Sample incorrect predictions ----
    wrong = [r for r in results if not r["correct"]]
    right = [r for r in results if r["correct"]]
    print(f"\n[shuffle] Correct: {len(right)}, Wrong: {len(wrong)}")
    if wrong:
        print("\n  Sample incorrect predictions:")
        for r in wrong[:5]:
            print(f"    #{r['idx']}: gold={r['gold']} pred={r['predicted']}  "
                  f"votes={r['votes']}  "
                  f"ctx=\"{r['ctx_preview'][:50]}...\"")


if __name__ == "__main__":
    main()
