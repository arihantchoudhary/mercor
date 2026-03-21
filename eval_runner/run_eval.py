#!/usr/bin/env python3
"""
Runner script for Part B of the LLM evaluation pipeline.

Supports three task categories:
  - mmlu      : Massive Multitask Language Understanding (via lm-eval)
  - hellaswag : HellaSwag commonsense reasoning (via lm-eval)
  - custom    : Programming Language Identification (local benchmark)

Results are written as JSON to eval_runner/results/ and a summary table
is printed to stdout.

Usage
-----
    python eval_runner/run_eval.py --model qwen2.5:7b --tasks mmlu hellaswag
    python eval_runner/run_eval.py --model qwen2.5:7b --tasks custom
    python eval_runner/run_eval.py --model qwen2.5:7b --tasks mmlu hellaswag custom
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so `from eval_runner.model` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
CUSTOM_BENCHMARK_PATH = SCRIPT_DIR / "custom_benchmark.json"
CUSTOM_TASK_YAML_DIR = str(SCRIPT_DIR)  # directory containing custom_task.yaml

# Official lm-eval task names
OFFICIAL_TASKS = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_results_dir():
    """Create results directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timestamp_slug() -> str:
    """Short timestamp string for file naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_results(data: dict, tag: str) -> Path:
    """Persist *data* as a timestamped JSON file and return the path."""
    ensure_results_dir()
    filename = f"{tag}_{timestamp_slug()}.json"
    path = RESULTS_DIR / filename
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


def print_summary_table(rows: list[dict]):
    """Print a nicely formatted summary table to stdout.

    Each *row* is a dict with keys: task, metric, value, num_samples.
    """
    if not rows:
        print("No results to display.")
        return

    # Column widths
    col_task = max(len("Task"), max(len(r["task"]) for r in rows))
    col_metric = max(len("Metric"), max(len(r["metric"]) for r in rows))
    col_value = max(len("Value"), 10)
    col_n = max(len("N"), 6)

    header = (
        f"  {'Task':<{col_task}}  "
        f"{'Metric':<{col_metric}}  "
        f"{'Value':>{col_value}}  "
        f"{'N':>{col_n}}"
    )
    sep = "  " + "-" * (col_task + col_metric + col_value + col_n + 8)

    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        val = r["value"]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(
            f"  {r['task']:<{col_task}}  "
            f"{r['metric']:<{col_metric}}  "
            f"{val_str:>{col_value}}  "
            f"{str(r['num_samples']):>{col_n}}"
        )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Run official lm-eval benchmarks (mmlu, hellaswag)
# ---------------------------------------------------------------------------

def run_official_tasks(
    model_name: str,
    tasks: list[str],
    base_url: str,
) -> dict:
    """Run one or more official lm-eval tasks using our Ollama wrapper.

    Returns the raw results dict produced by lm_eval.simple_evaluate().
    """
    # Lazy imports so the script can still --help without lm-eval installed
    import lm_eval

    from eval_runner.model import OllamaLM

    lm = OllamaLM(model_name=model_name, base_url=base_url)

    task_names = [OFFICIAL_TASKS[t] for t in tasks]
    print(f"[lm-eval] Running tasks: {task_names} with model '{model_name}'")
    print(f"[lm-eval] This may take a while...\n")

    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_names,
        num_fewshot=0,
        batch_size=1,
        log_samples=False,
    )
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    results["model_name"] = model_name

    return results


# ---------------------------------------------------------------------------
# Run custom benchmark
# ---------------------------------------------------------------------------

def run_custom_benchmark(model_name: str, base_url: str) -> dict:
    """Run the local programming-language-identification benchmark.

    This uses lm_eval.simple_evaluate with our custom task YAML that
    points to custom_benchmark.json.  If lm-eval fails to load the task
    (version incompatibility, missing dataset deps, etc.), we fall back
    to a simple manual evaluation loop.
    """
    try:
        return _run_custom_via_harness(model_name, base_url)
    except Exception as exc:
        print(
            f"[custom] lm-eval harness approach failed ({exc}); "
            f"falling back to manual evaluation.\n"
        )
        return _run_custom_manual(model_name, base_url)


def _run_custom_via_harness(model_name: str, base_url: str) -> dict:
    """Attempt to run the custom task through lm-eval's task system."""
    import lm_eval

    from eval_runner.model import OllamaLM

    lm = OllamaLM(model_name=model_name, base_url=base_url)

    print(f"[custom] Running prog_lang_id via lm-eval harness...")
    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["prog_lang_id"],
        task_manager=lm_eval.tasks.TaskManager(
            include_path=CUSTOM_TASK_YAML_DIR
        ),
        num_fewshot=0,
        batch_size=1,
        log_samples=True,
    )
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    results["model_name"] = model_name
    return results


def _run_custom_manual(model_name: str, base_url: str) -> dict:
    """Fallback: manually iterate over the benchmark JSON and evaluate
    exact-match accuracy using the Ollama generate API directly.
    """
    import requests as req_lib
    from requests.adapters import HTTPAdapter
    from tqdm import tqdm

    with open(CUSTOM_BENCHMARK_PATH) as fh:
        benchmark = json.load(fh)

    examples = benchmark["examples"]
    print(f"[custom] Running manual evaluation on {len(examples)} examples...")

    # Use a session with connection pooling
    session = req_lib.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    correct = 0
    details: list[dict] = []
    t0 = time.time()

    for ex in tqdm(examples, desc="custom eval"):
        prompt = (
            "Identify the programming language of the following code snippet. "
            "Reply with only the language name, nothing else.\n\n"
            f"Code:\n```\n{ex['snippet']}\n```\n\n"
            "Language:"
        )

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {
                "temperature": 0.0,
                "seed": 42,
                "num_predict": 20,
                "stop": ["\n"],
            },
        }

        try:
            resp = session.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
        except req_lib.RequestException as e:
            answer = f"[error: {e}]"

        expected = ex["language"]
        match = answer.lower().strip() == expected.lower().strip()
        if match:
            correct += 1

        details.append({
            "expected": expected,
            "predicted": answer,
            "correct": match,
        })

    elapsed = time.time() - t0
    accuracy = correct / len(examples) if examples else 0.0

    return {
        "task": "prog_lang_id",
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "elapsed_seconds": elapsed,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Result extraction helpers
# ---------------------------------------------------------------------------

def extract_summary_rows(results: dict, task_label: str) -> list[dict]:
    """Pull metric rows from an lm-eval results dict or a manual-eval dict.

    Returns a list of row dicts suitable for print_summary_table().
    """
    rows: list[dict] = []

    # Manual custom eval format
    if "accuracy" in results and "details" in results:
        rows.append({
            "task": task_label,
            "metric": "exact_match",
            "value": results["accuracy"],
            "num_samples": results["total"],
        })
        return rows

    # lm-eval simple_evaluate format: results["results"][task_name][metric]
    lm_results = results.get("results", {})
    for task_name, metrics in lm_results.items():
        for metric_key, value in metrics.items():
            # Skip internal / alias keys
            if metric_key.startswith("alias"):
                continue
            # lm-eval stores metrics like "acc,none" or "acc_norm,none"
            if isinstance(value, (int, float)):
                # Determine sample count from n-samples if available
                n_samples = results.get("n-samples", {}).get(task_name, "?")
                rows.append({
                    "task": task_name,
                    "metric": metric_key,
                    "value": value,
                    "num_samples": n_samples,
                })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM evaluation benchmarks via Ollama."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["mmlu", "hellaswag", "custom"],
        required=True,
        help="Tasks to evaluate (mmlu, hellaswag, custom)",
    )
    parser.add_argument(
        "--base-url",
        default=OLLAMA_BASE_URL,
        help=f"Ollama base URL (default: {OLLAMA_BASE_URL})",
    )
    args = parser.parse_args()

    all_summary_rows: list[dict] = []
    all_results: dict = {}

    # ---- Official lm-eval tasks ----
    official = [t for t in args.tasks if t in OFFICIAL_TASKS]
    if official:
        try:
            results = run_official_tasks(
                model_name=args.model,
                tasks=official,
                base_url=args.base_url,
            )
            all_results["official"] = results
            for t in official:
                all_summary_rows.extend(
                    extract_summary_rows(results, t)
                )
            path = save_results(results, "official")
            print(f"[lm-eval] Results saved to {path}")
        except Exception as exc:
            print(f"ERROR running official tasks: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # ---- Custom benchmark ----
    if "custom" in args.tasks:
        try:
            results = run_custom_benchmark(
                model_name=args.model,
                base_url=args.base_url,
            )
            all_results["custom"] = results
            all_summary_rows.extend(
                extract_summary_rows(results, "prog_lang_id")
            )
            path = save_results(results, "custom")
            print(f"[custom] Results saved to {path}")
        except Exception as exc:
            print(f"ERROR running custom benchmark: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # ---- Summary ----
    print_summary_table(all_summary_rows)

    # Also save a combined summary
    if all_results:
        summary = {
            "model": args.model,
            "tasks": args.tasks,
            "summary": all_summary_rows,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = save_results(summary, "summary")
        print(f"Combined summary saved to {path}")


if __name__ == "__main__":
    main()
