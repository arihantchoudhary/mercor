#!/usr/bin/env python3
"""
Part E -- Step 3: Main inference script for HellaSwag evaluation.

Loads HellaSwag data, applies inference-time optimizations, queries the
Ollama endpoint, computes accuracy with 95% confidence intervals, and
saves detailed results.

KEY APPROACH: HellaSwag is a multiple-choice task. Instead of generating
free text and trying to extract A/B/C/D (which fails badly), we force the
model to output a single token (num_predict=1) after a prompt that ends
with "The answer is". This makes answer extraction trivial.

For the optimized mode, we add:
  - Few-shot examples (TF-IDF selected from training data)
  - Self-consistency decoding (k=3, temperature=0.3)

Usage:
    # Baseline (zero-shot, forced single-token)
    python improve/infer.py --baseline --n-examples 200 --model qwen2.5:7b

    # Optimized (few-shot + self-consistency)
    python improve/infer.py --optimized --n-examples 200 --model qwen2.5:7b \
        --template-name direct --n-shot 3 --sc-k 3 --sc-temp 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    np = None  # fallback to pure-python bootstrap

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from improve.optimize_prompt import FewShotSelector

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
REQUEST_TIMEOUT = 300
LABELS = ["A", "B", "C", "D"]

# Persistent session for connection pooling (reuses TCP connections)
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
# Ollama generation helper
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 40,
    seed: int = 42,
    max_tokens: int = 1,
    stop: list[str] | None = None,
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
            "top_p": top_p,
            "top_k": top_k,
        },
    }
    if stop:
        payload["options"]["stop"] = stop

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
# Answer extraction -- trivial with forced single-token output
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    """Extract answer letter from a (typically single-token) model response.

    With num_predict=1, the response is usually just 'A', 'B', 'C', or 'D'.
    We handle minor variations (whitespace, lowercase, punctuation).
    """
    text = raw.strip().upper()
    for ch in text:
        if ch in "ABCD":
            return ch
    return "A"  # fallback


# ---------------------------------------------------------------------------
# Format choices helper
# ---------------------------------------------------------------------------

def format_choices(endings: list[str]) -> str:
    """Format the four endings as A/B/C/D options."""
    labels = ["A", "B", "C", "D"]
    lines = []
    for label, ending in zip(labels, endings):
        lines.append(f"{label}) {ending.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline inference (zero-shot, forced single-token)
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = (
    "Context: {ctx}\n\n"
    "Which continuation is most natural?\n"
    "{choices}\n\n"
    "The answer is"
)


def run_baseline(
    examples: list[dict],
    model: str = MODEL_NAME,
    seed: int = 42,
) -> list[dict]:
    """Zero-shot baseline: single-token forced choice, greedy decoding."""
    results = []

    for i, ex in enumerate(examples):
        choices_str = format_choices(ex["endings"])
        prompt = BASELINE_TEMPLATE.format(ctx=ex["ctx"], choices=choices_str)

        t0 = time.time()
        raw_response = ollama_generate(
            prompt,
            model=model,
            temperature=0.0,
            seed=seed,
            max_tokens=1,
        )
        latency = time.time() - t0

        predicted = extract_answer(raw_response)
        gold_label = LABELS[ex["label"]] if 0 <= ex["label"] <= 3 else "?"
        correct = predicted == gold_label

        results.append({
            "idx": i,
            "ind": ex.get("ind"),
            "ctx_preview": ex["ctx"][:80],
            "gold": gold_label,
            "gold_idx": ex["label"],
            "predicted": predicted,
            "correct": correct,
            "raw_response": raw_response.strip(),
            "latency_s": round(latency, 3),
            "strategy": "baseline",
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


# ---------------------------------------------------------------------------
# Optimized inference (few-shot + self-consistency)
# ---------------------------------------------------------------------------

OPTIMIZED_TEMPLATE = (
    "{few_shot_block}"
    "Context: {ctx}\n\n"
    "Which continuation is most natural?\n"
    "{choices}\n\n"
    "The answer is"
)


def build_few_shot_block_for_mcq(
    selector: FewShotSelector,
    query_ctx: str,
    k: int = 3,
) -> str:
    """Build a few-shot prefix with the same forced-choice format."""
    examples = selector.select(query_ctx, k=k)
    if not examples:
        return ""

    blocks: list[str] = []
    for ex in examples:
        choices_str = format_choices(ex["endings"])
        gold = LABELS[ex["label"]] if 0 <= ex["label"] <= 3 else "A"
        blocks.append(
            f"Context: {ex['ctx']}\n\n"
            f"Which continuation is most natural?\n"
            f"{choices_str}\n\n"
            f"The answer is {gold}\n"
        )
    return "\n".join(blocks) + "\n"


def run_optimized(
    examples: list[dict],
    train_examples: list[dict] | None = None,
    model: str = MODEL_NAME,
    template_name: str = "direct",
    n_shot: int = 3,
    sc_k: int = 3,
    sc_temp: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """Optimized: few-shot examples + self-consistency majority voting.

    Each sample uses num_predict=1 for forced single-token output.
    Self-consistency generates k samples at low temperature and majority-votes.
    """
    # Build few-shot selector if training data available
    few_shot_selector = None
    if train_examples and n_shot > 0:
        print(f"  Building TF-IDF index over {len(train_examples)} training examples...")
        few_shot_selector = FewShotSelector(train_examples)

    results = []

    for i, ex in enumerate(examples):
        t0 = time.time()

        # Build few-shot block
        few_shot_block = ""
        if few_shot_selector:
            few_shot_block = build_few_shot_block_for_mcq(
                few_shot_selector, ex["ctx"], k=n_shot
            )

        choices_str = format_choices(ex["endings"])
        prompt = OPTIMIZED_TEMPLATE.format(
            few_shot_block=few_shot_block,
            ctx=ex["ctx"],
            choices=choices_str,
        )

        # Self-consistency: k samples with majority voting
        votes: list[str] = []
        raw_responses: list[str] = []

        for j in range(sc_k):
            raw = ollama_generate(
                prompt,
                model=model,
                temperature=sc_temp,
                seed=seed + j * 7,
                top_p=0.95,
                top_k=50,
                max_tokens=1,
            )
            raw_responses.append(raw.strip())
            votes.append(extract_answer(raw))

        # Majority vote (tie-break: first vote that has max count)
        counter = Counter(votes)
        best_count = counter.most_common(1)[0][1]
        predicted = next(v for v in votes if counter[v] == best_count)

        latency = time.time() - t0

        gold_label = LABELS[ex["label"]] if 0 <= ex["label"] <= 3 else "?"
        correct = predicted == gold_label

        results.append({
            "idx": i,
            "ind": ex.get("ind"),
            "ctx_preview": ex["ctx"][:80],
            "gold": gold_label,
            "gold_idx": ex["label"],
            "predicted": predicted,
            "correct": correct,
            "raw_response": " | ".join(raw_responses),
            "votes": votes,
            "latency_s": round(latency, 3),
            "strategy": "optimized",
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


# ---------------------------------------------------------------------------
# Accuracy + Confidence Intervals
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
        rng = np.random.RandomState(42)
        correctness = np.array([1 if r["correct"] else 0 for r in results])
        boot_accs = []
        for _ in range(2000):
            sample = rng.choice(correctness, size=n, replace=True)
            boot_accs.append(sample.mean())
        boot_accs = np.array(boot_accs)
        ci_lower_boot = float(np.percentile(boot_accs, 2.5))
        ci_upper_boot = float(np.percentile(boot_accs, 97.5))

    return {
        "accuracy": round(acc, 5),
        "correct": correct,
        "total": n,
        "ci_95_wilson": [round(ci_lower_wilson, 5), round(ci_upper_wilson, 5)],
        "ci_95_bootstrap": [round(ci_lower_boot, 5), round(ci_upper_boot, 5)],
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict],
    metrics: dict,
    tag: str,
    config: dict,
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

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    # Also save a "latest" symlink / copy for convenience
    latest_path = RESULTS_DIR / f"{tag}_latest.json"
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OLLAMA_BASE_URL, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="HellaSwag inference with forced single-token MCQ approach."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline", action="store_true",
        help="Run zero-shot baseline (forced single-token, greedy)",
    )
    mode.add_argument(
        "--optimized", action="store_true",
        help="Run with few-shot + self-consistency optimizations",
    )

    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")

    # Strategy-specific args
    parser.add_argument("--template-name", default="direct",
                        help="Prompt template name")
    parser.add_argument("--n-shot", type=int, default=3,
                        help="Number of few-shot examples")
    parser.add_argument("--sc-k", type=int, default=3,
                        help="Self-consistency: number of samples")
    parser.add_argument("--sc-temp", type=float, default=0.3,
                        help="Self-consistency: sampling temperature")

    args = parser.parse_args()

    # Update globals from args
    OLLAMA_BASE_URL = args.base_url
    MODEL_NAME = args.model

    # ---- Health check ----
    try:
        session = _get_session()
        resp = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(f"ERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
                 f"Start the server first: python serve/serve.py (or make serve)")
    except requests.RequestException as exc:
        sys.exit(f"ERROR: Ollama health check failed: {exc}")

    # ---- Load data ----
    val_path = DATA_DIR / "hellaswag_val.json"
    if not val_path.exists():
        sys.exit(
            f"ERROR: Validation data not found at {val_path}.\n"
            f"Run:  python improve/prepare_data.py"
        )

    with open(val_path) as f:
        val_examples = json.load(f)
    print(f"[infer] Loaded {len(val_examples)} validation examples.")

    if args.n_examples > 0:
        val_examples = val_examples[: args.n_examples]
        print(f"[infer] Limited to {len(val_examples)} examples.")

    # ---- Load training data (for few-shot) ----
    train_examples = None
    train_path = DATA_DIR / "hellaswag_train.json"
    if train_path.exists():
        with open(train_path) as f:
            train_examples = json.load(f)
        print(f"[infer] Loaded {len(train_examples)} training examples for few-shot.")

    # ---- Build config ----
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "n_examples": len(val_examples),
        "template_name": args.template_name,
        "n_shot": args.n_shot,
        "sc_k": args.sc_k,
        "sc_temp": args.sc_temp,
    }

    # ---- Run inference ----
    if args.baseline:
        print("\n[infer] === BASELINE (zero-shot, forced single-token, greedy) ===")
        config["strategy"] = "baseline"
        results = run_baseline(val_examples, model=args.model, seed=args.seed)
        tag = "baseline"

    elif args.optimized:
        print("\n[infer] === OPTIMIZED (few-shot + self-consistency) ===")
        config["strategy"] = "optimized"
        results = run_optimized(
            val_examples,
            train_examples=train_examples,
            model=args.model,
            template_name=args.template_name,
            n_shot=args.n_shot,
            sc_k=args.sc_k,
            sc_temp=args.sc_temp,
            seed=args.seed,
        )
        tag = "optimized"

    # ---- Compute metrics ----
    metrics = compute_accuracy(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results) if results else 0
    metrics["avg_latency_s"] = round(avg_latency, 3)
    metrics["total_time_s"] = round(sum(r["latency_s"] for r in results), 1)

    print(f"\n{'='*50}")
    print(f"Strategy:  {config['strategy']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total']})")
    print(f"95% CI (Wilson):    [{metrics['ci_95_wilson'][0]:.4f}, {metrics['ci_95_wilson'][1]:.4f}]")
    print(f"95% CI (Bootstrap): [{metrics['ci_95_bootstrap'][0]:.4f}, {metrics['ci_95_bootstrap'][1]:.4f}]")
    print(f"Avg latency:        {metrics['avg_latency_s']:.3f}s")
    print(f"Total time:         {metrics['total_time_s']:.1f}s")
    print(f"{'='*50}")

    # ---- Save ----
    path = save_results(results, metrics, tag, config)
    print(f"\n[infer] Results saved to {path}")

    # ---- Print sample correct/incorrect ----
    wrong = [r for r in results if not r["correct"]]
    right = [r for r in results if r["correct"]]
    print(f"\n[infer] Correct: {len(right)}, Wrong: {len(wrong)}")
    if wrong:
        print("\n  Sample incorrect predictions:")
        for r in wrong[:5]:
            print(f"    #{r['idx']}: gold={r['gold']} pred={r['predicted']}  "
                  f"raw=\"{r['raw_response'][:30]}\"  "
                  f"ctx=\"{r['ctx_preview'][:50]}...\"")


if __name__ == "__main__":
    main()
