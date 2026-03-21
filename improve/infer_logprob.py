#!/usr/bin/env python3
"""
HellaSwag inference using logprob-based scoring via Ollama's
OpenAI-compatible /v1/chat/completions endpoint.

Instead of relying on the raw generated token, this script extracts the
model's full probability distribution over the answer tokens A/B/C/D from
the top_logprobs array and picks the one with the highest logprob (least
negative). This is MUCH more robust than single-token generation.

Baseline mode:  zero-shot, logprob scoring
Optimized mode: few-shot examples + logprob scoring (the logprob approach
                itself IS the optimisation -- it uses the model's full
                probability distribution rather than a single greedy sample)

Usage:
    python improve/infer_logprob.py --baseline --n-examples 200 --model qwen2.5:7b
    python improve/infer_logprob.py --optimized --n-examples 200 --model qwen2.5:7b --n-shot 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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
# Logprob-based query
# ---------------------------------------------------------------------------

def query_logprobs(
    prompt: str,
    model: str = MODEL_NAME,
    top_logprobs: int = 10,
) -> dict[str, float]:
    """Send a chat completion request and return logprobs for A/B/C/D.

    Returns a dict like {"A": -0.5, "B": -2.1, "C": -3.4, "D": -5.0}.
    Missing labels get a large negative penalty (-100).
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "temperature": 0,
    }

    session = _get_session()
    try:
        resp = session.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        print(f"  [WARN] Logprob request failed: {exc}")
        return {label: -100.0 for label in LABELS}

    # Navigate: choices[0].logprobs.content[0].top_logprobs
    try:
        top_lps = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        print("  [WARN] Could not parse logprobs from response")
        return {label: -100.0 for label in LABELS}

    # Build a lookup from the top_logprobs entries
    token_logprobs: dict[str, float] = {}
    for entry in top_lps:
        tok = entry.get("token", "").strip()
        lp = entry.get("logprob", -100.0)
        token_logprobs[tok] = lp

    # Extract logprobs for each label
    result: dict[str, float] = {}
    for label in LABELS:
        # Try exact match first, then case variations
        if label in token_logprobs:
            result[label] = token_logprobs[label]
        elif label.lower() in token_logprobs:
            result[label] = token_logprobs[label.lower()]
        else:
            result[label] = -100.0

    return result


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def format_choices(endings: list[str]) -> str:
    lines = []
    for label, ending in zip(LABELS, endings):
        lines.append(f"{label}) {ending.strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = (
    "Context: {ctx}\n\n"
    "Which continuation is most natural?\n"
    "{choices}\n\n"
    "The answer is"
)

OPTIMIZED_TEMPLATE = (
    "{few_shot_block}"
    "Context: {ctx}\n\n"
    "Which continuation is most natural?\n"
    "{choices}\n\n"
    "The answer is"
)


# ---------------------------------------------------------------------------
# Few-shot block builder
# ---------------------------------------------------------------------------

def build_few_shot_block(
    selector: FewShotSelector,
    query_ctx: str,
    k: int = 3,
) -> str:
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


# ---------------------------------------------------------------------------
# Baseline inference (zero-shot, logprob scoring)
# ---------------------------------------------------------------------------

def run_baseline(
    examples: list[dict],
    model: str = MODEL_NAME,
) -> list[dict]:
    results = []

    for i, ex in enumerate(examples):
        choices_str = format_choices(ex["endings"])
        prompt = BASELINE_TEMPLATE.format(ctx=ex["ctx"], choices=choices_str)

        t0 = time.time()
        logprobs = query_logprobs(prompt, model=model)
        latency = time.time() - t0

        # Pick the label with the highest logprob
        predicted = max(logprobs, key=logprobs.get)
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
            "logprobs": {k: round(v, 4) for k, v in logprobs.items()},
            "latency_s": round(latency, 3),
            "strategy": "logprob_baseline",
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


# ---------------------------------------------------------------------------
# Optimized inference (few-shot + logprob scoring)
# ---------------------------------------------------------------------------

def run_optimized(
    examples: list[dict],
    train_examples: list[dict] | None = None,
    model: str = MODEL_NAME,
    n_shot: int = 3,
) -> list[dict]:
    # Build few-shot selector
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
            few_shot_block = build_few_shot_block(
                few_shot_selector, ex["ctx"], k=n_shot
            )

        choices_str = format_choices(ex["endings"])
        prompt = OPTIMIZED_TEMPLATE.format(
            few_shot_block=few_shot_block,
            ctx=ex["ctx"],
            choices=choices_str,
        )

        logprobs = query_logprobs(prompt, model=model)
        latency = time.time() - t0

        predicted = max(logprobs, key=logprobs.get)
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
            "logprobs": {k: round(v, 4) for k, v in logprobs.items()},
            "latency_s": round(latency, 3),
            "strategy": "logprob_optimized",
        })

        if (i + 1) % 20 == 0 or i == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i+1}/{len(examples)}] Running acc: {acc_so_far:.3f}")

    return results


# ---------------------------------------------------------------------------
# Accuracy + Confidence Intervals
# ---------------------------------------------------------------------------

def compute_accuracy(results: list[dict]) -> dict:
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

    # Also save a "latest" copy for convenience
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
        description="HellaSwag inference with logprob-based scoring."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline", action="store_true",
        help="Run zero-shot baseline with logprob scoring",
    )
    mode.add_argument(
        "--optimized", action="store_true",
        help="Run few-shot + logprob scoring",
    )

    parser.add_argument("--model", default=MODEL_NAME, help="Ollama model name")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama URL")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    parser.add_argument("--n-shot", type=int, default=3,
                        help="Number of few-shot examples (optimized mode)")

    args = parser.parse_args()

    OLLAMA_BASE_URL = args.base_url
    MODEL_NAME = args.model

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

    # Load data
    val_path = DATA_DIR / "hellaswag_val.json"
    if not val_path.exists():
        sys.exit(
            f"ERROR: Validation data not found at {val_path}.\n"
            f"Run:  python improve/prepare_data.py"
        )

    with open(val_path) as f:
        val_examples = json.load(f)
    print(f"[infer_logprob] Loaded {len(val_examples)} validation examples.")

    if args.n_examples > 0:
        val_examples = val_examples[: args.n_examples]
        print(f"[infer_logprob] Limited to {len(val_examples)} examples.")

    # Load training data (for few-shot)
    train_examples = None
    train_path = DATA_DIR / "hellaswag_train.json"
    if train_path.exists():
        with open(train_path) as f:
            train_examples = json.load(f)
        print(f"[infer_logprob] Loaded {len(train_examples)} training examples for few-shot.")

    # Build config
    config = {
        "model": args.model,
        "base_url": args.base_url,
        "n_examples": len(val_examples),
        "n_shot": args.n_shot,
        "scoring": "logprob",
        "top_logprobs": 10,
        "temperature": 0,
    }

    # Run inference
    if args.baseline:
        print("\n[infer_logprob] === BASELINE (zero-shot, logprob scoring) ===")
        config["strategy"] = "logprob_baseline"
        results = run_baseline(val_examples, model=args.model)
        tag = "logprob_baseline"

    elif args.optimized:
        print(f"\n[infer_logprob] === OPTIMIZED ({args.n_shot}-shot, logprob scoring) ===")
        config["strategy"] = "logprob_optimized"
        results = run_optimized(
            val_examples,
            train_examples=train_examples,
            model=args.model,
            n_shot=args.n_shot,
        )
        tag = "logprob_optimized"

    # Compute metrics
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

    # Save
    path = save_results(results, metrics, tag, config)
    print(f"\n[infer_logprob] Results saved to {path}")

    # Print sample correct/incorrect
    wrong = [r for r in results if not r["correct"]]
    right = [r for r in results if r["correct"]]
    print(f"\n[infer_logprob] Correct: {len(right)}, Wrong: {len(wrong)}")
    if wrong:
        print("\n  Sample incorrect predictions:")
        for r in wrong[:5]:
            print(f"    #{r['idx']}: gold={r['gold']} pred={r['predicted']}  "
                  f"logprobs={r['logprobs']}  "
                  f"ctx=\"{r['ctx_preview'][:50]}...\"")


if __name__ == "__main__":
    main()
