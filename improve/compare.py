#!/usr/bin/env python3
"""
Part E -- Compare baseline vs improved evaluation results.

Loads results/baseline.json and results/improved.json, computes per-example
deltas, and prints a formatted comparison table with summary statistics.
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline.json")
IMPROVED_PATH = os.path.join(RESULTS_DIR, "improved.json")


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_header(title: str, width: int = 78):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def run():
    if not os.path.exists(BASELINE_PATH):
        print(f"ERROR: {BASELINE_PATH} not found. Run baseline.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(IMPROVED_PATH):
        print(f"ERROR: {IMPROVED_PATH} not found. Run improved.py first.", file=sys.stderr)
        sys.exit(1)

    baseline = load(BASELINE_PATH)
    improved = load(IMPROVED_PATH)

    b_results = baseline["results"]
    i_results = improved["results"]

    # ── Per-example comparison table ─────────────────────────────────────
    print_header("Part E: Baseline vs Improved -- Per-Example Comparison")

    col = f"{'#':>3s}  {'Expected':<14s}  {'Baseline':<14s}  {'Improved':<14s}  {'B':>3s}  {'I':>3s}  Delta"
    print(f"  {col}")
    print(f"  {'-' * len(col)}")

    flipped_to_correct = 0
    flipped_to_wrong = 0
    both_correct = 0
    both_wrong = 0

    for b, imp in zip(b_results, i_results):
        idx = b["index"] + 1
        expected = b["expected"]
        b_pred = b["predicted"]
        i_pred = imp["predicted"]
        b_ok = b["correct"]
        i_ok = imp["correct"]

        b_mark = "Y" if b_ok else "N"
        i_mark = "Y" if i_ok else "N"

        if not b_ok and i_ok:
            delta = " +  (fixed)"
            flipped_to_correct += 1
        elif b_ok and not i_ok:
            delta = " -  (regressed)"
            flipped_to_wrong += 1
        elif b_ok and i_ok:
            delta = "    (both OK)"
            both_correct += 1
        else:
            delta = "    (both wrong)"
            both_wrong += 1

        print(f"  {idx:3d}  {expected:<14s}  {b_pred:<14s}  {i_pred:<14s}  {b_mark:>3s}  {i_mark:>3s} {delta}")

    # ── Summary table ────────────────────────────────────────────────────
    print_header("Summary")

    b_acc = baseline["accuracy"]
    i_acc = improved["accuracy"]
    delta_acc = i_acc - b_acc
    b_correct = baseline["correct"]
    i_correct = improved["correct"]
    total = baseline["total"]

    sign = "+" if delta_acc >= 0 else ""

    rows = [
        ("Method",      "Baseline",               "Improved"),
        ("Techniques",  "temp=0, bare prompt",     ", ".join(improved.get("techniques", []))),
        ("Samples",     "1",                       str(improved.get("num_samples", "?"))),
        ("Correct",     f"{b_correct}/{total}",    f"{i_correct}/{total}"),
        ("Accuracy",    f"{b_acc:.1%}",            f"{i_acc:.1%}"),
    ]

    w1, w2, w3 = 12, 24, 44
    print(f"  {'Metric':<{w1}s}  {'Baseline':<{w2}s}  {'Improved':<{w3}s}")
    print(f"  {'-'*w1}  {'-'*w2}  {'-'*w3}")
    for label, bv, iv in rows:
        print(f"  {label:<{w1}s}  {bv:<{w2}s}  {iv:<{w3}s}")

    # ── Delta highlight ──────────────────────────────────────────────────
    print()
    print(f"  Accuracy delta:  {sign}{delta_acc:.1%}  ({sign}{i_correct - b_correct} examples)")
    print(f"  Fixed (N->Y):    {flipped_to_correct}")
    print(f"  Regressed (Y->N): {flipped_to_wrong}")
    print(f"  Both correct:    {both_correct}")
    print(f"  Both wrong:      {both_wrong}")

    if delta_acc > 0:
        print(f"\n  >>> Improvement confirmed: {sign}{delta_acc:.1%} accuracy gain <<<")
    elif delta_acc == 0:
        print(f"\n  --- No change in accuracy ---")
    else:
        print(f"\n  !!! Regression: accuracy dropped by {abs(delta_acc):.1%} !!!")

    print()


if __name__ == "__main__":
    run()
