#!/usr/bin/env python3
"""
Part D -- Determinism and output validation for the LLM evaluation pipeline.

Tests:
  1. Deterministic mode: send identical prompts with temperature=0, top_p=1,
     seed=42 across multiple trials and verify outputs are byte-identical.
  2. Output validation: for a programming-language-identification benchmark,
     check that the model's responses pass regex and schema validation.

Usage:
    python guardrails/validate.py
    python guardrails/validate.py --model qwen2.5:7b --trials 5
    python guardrails/validate.py --base-url http://localhost:11434
"""

import argparse
import json
import re
import sys
import time

import requests
from requests.adapters import HTTPAdapter

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TRIALS = 5

# Persistent session with connection pooling
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Determinism prompts -- five short, diverse prompts whose answers should be
# fully reproducible when sampled with temperature=0 and a fixed seed.
# ---------------------------------------------------------------------------
DETERMINISM_PROMPTS = [
    "What is the capital of Japan? Answer with just the city name.",
    "What is 7 * 8? Answer with just the number.",
    "Name one primary color. Answer with a single word.",
    "What programming language is Flask written in? Answer with one word.",
    "Is the earth round or flat? Answer with one word.",
]

# ---------------------------------------------------------------------------
# Programming-language-identification benchmark -- custom evaluation task.
# Each item is (code_snippet, expected_language).
# ---------------------------------------------------------------------------
LANG_ID_BENCHMARK = [
    {
        "code": 'fmt.Println("Hello, World!")',
        "expected": "Go",
    },
    {
        "code": "console.log('Hello, World!');",
        "expected": "JavaScript",
    },
    {
        "code": 'print("Hello, World!")',
        "expected": "Python",
    },
    {
        "code": 'System.out.println("Hello, World!");',
        "expected": "Java",
    },
    {
        "code": 'puts "Hello, World!"',
        "expected": "Ruby",
    },
]

# Allowed language names for schema validation (case-insensitive matching).
ALLOWED_LANGUAGES = {
    "assembly", "bash", "c", "c#", "c++", "clojure", "cobol", "css",
    "dart", "elixir", "erlang", "fortran", "go", "golang", "groovy",
    "haskell", "html", "java", "javascript", "js", "json", "julia",
    "kotlin", "lisp", "lua", "matlab", "objective-c", "ocaml", "pascal",
    "perl", "php", "powershell", "python", "r", "ruby", "rust", "scala",
    "shell", "sql", "swift", "typescript", "vb.net", "zig",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate(base_url: str, model: str, prompt: str, deterministic: bool = False) -> str:
    """Send a generate request and return the response text."""
    options = {"num_predict": 32}
    if deterministic:
        options.update({"temperature": 0, "top_p": 1, "seed": 42})

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    session = _get_session()
    resp = session.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def health_check(base_url: str) -> bool:
    """Return True if Ollama is reachable."""
    try:
        session = _get_session()
        resp = session.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Test 1 -- Determinism
# ---------------------------------------------------------------------------

def run_determinism_test(base_url: str, model: str, trials: int) -> list[dict]:
    """Run each determinism prompt `trials` times and report results."""
    results = []
    for idx, prompt in enumerate(DETERMINISM_PROMPTS, 1):
        outputs = []
        for t in range(trials):
            text = generate(base_url, model, prompt, deterministic=True)
            outputs.append(text)
        unique = set(outputs)
        passed = len(unique) == 1
        results.append({
            "prompt_idx": idx,
            "prompt": prompt,
            "passed": passed,
            "unique_outputs": len(unique),
            "outputs": outputs,
        })
    return results


# ---------------------------------------------------------------------------
# Test 2 -- Output validation (programming language identification)
# ---------------------------------------------------------------------------

# Regex: the model should respond with one or two words (the language name),
# optionally followed by a period or punctuation.  We strip whitespace first.
LANG_REGEX = re.compile(
    r"^[A-Za-z][A-Za-z+#.\-]*(?:\s[A-Za-z+#.\-]+)?\.?$"
)


def validate_output(raw: str) -> dict:
    """Run regex and schema checks on a raw model output.

    Returns dict with keys: cleaned, regex_pass, schema_pass.
    """
    # Strip markdown backticks, quotes, trailing punctuation noise
    cleaned = raw.strip().strip("`'\"").strip()
    # Take only the first line if multi-line
    cleaned = cleaned.split("\n")[0].strip().rstrip(".")

    regex_pass = bool(LANG_REGEX.match(cleaned))
    schema_pass = cleaned.lower() in ALLOWED_LANGUAGES

    return {
        "cleaned": cleaned,
        "regex_pass": regex_pass,
        "schema_pass": schema_pass,
    }


def run_validation_test(base_url: str, model: str) -> list[dict]:
    """Run the language-identification benchmark with output validation."""
    results = []
    for item in LANG_ID_BENCHMARK:
        prompt = (
            f"What programming language is the following code written in? "
            f"Reply with ONLY the language name, nothing else.\n\n"
            f"```\n{item['code']}\n```"
        )
        raw = generate(base_url, model, prompt, deterministic=True)
        validation = validate_output(raw)
        correct = validation["cleaned"].lower() == item["expected"].lower()
        results.append({
            "code_snippet": item["code"],
            "expected": item["expected"],
            "raw_output": raw,
            "cleaned": validation["cleaned"],
            "regex_pass": validation["regex_pass"],
            "schema_pass": validation["schema_pass"],
            "correct": correct,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_determinism_report(results: list[dict], trials: int):
    """Pretty-print determinism test results."""
    print("=" * 64)
    print(f"  DETERMINISM TEST  ({trials} trials per prompt)")
    print("=" * 64)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n  [{status}] Prompt {r['prompt_idx']}: {r['prompt']}")
        if r["passed"]:
            print(f"         Output: {r['outputs'][0]!r}")
        else:
            print(f"         Unique outputs ({r['unique_outputs']}):")
            for i, o in enumerate(r["outputs"]):
                print(f"           Trial {i+1}: {o!r}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n  Summary: {passed}/{total} prompts deterministic\n")


def print_validation_report(results: list[dict]):
    """Pretty-print output-validation test results."""
    print("=" * 64)
    print("  OUTPUT VALIDATION TEST  (programming language identification)")
    print("=" * 64)
    for r in results:
        print(f"\n  Code: {r['code_snippet']}")
        print(f"    Expected:    {r['expected']}")
        print(f"    Raw output:  {r['raw_output']!r}")
        print(f"    Cleaned:     {r['cleaned']!r}")
        regex_s = "PASS" if r["regex_pass"] else "FAIL"
        schema_s = "PASS" if r["schema_pass"] else "FAIL"
        correct_s = "PASS" if r["correct"] else "FAIL"
        print(f"    Regex:       [{regex_s}]  (single language-name format)")
        print(f"    Schema:      [{schema_s}]  (in allowed languages list)")
        print(f"    Correct:     [{correct_s}]  (matches expected language)")

    regex_passed = sum(1 for r in results if r["regex_pass"])
    schema_passed = sum(1 for r in results if r["schema_pass"])
    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\n  Summary:")
    print(f"    Regex validation:  {regex_passed}/{total} passed")
    print(f"    Schema validation: {schema_passed}/{total} passed")
    print(f"    Correct answer:    {correct_count}/{total} passed\n")


def save_results(det_results: list[dict], val_results: list[dict],
                 model: str, trials: int):
    """Save results to guardrails/results/validation.json."""
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validation.json")

    report = {
        "model": model,
        "trials": trials,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "determinism": {
            "total_prompts": len(det_results),
            "deterministic_count": sum(1 for r in det_results if r["passed"]),
            "details": det_results,
        },
        "validation": {
            "total_items": len(val_results),
            "regex_passed": sum(1 for r in val_results if r["regex_pass"]),
            "schema_passed": sum(1 for r in val_results if r["schema_pass"]),
            "correct": sum(1 for r in val_results if r["correct"]),
            "details": val_results,
        },
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Results saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Part D -- Determinism and output validation checks"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url", default=OLLAMA_BASE,
        help=f"Ollama base URL (default: {OLLAMA_BASE})",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Number of trials for determinism test (default: {DEFAULT_TRIALS})",
    )
    args = parser.parse_args()

    # Health check
    if not health_check(args.base_url):
        print(f"ERROR: Cannot reach Ollama at {args.base_url}", file=sys.stderr)
        print("Start the server first: python serve/serve.py (or make serve)",
              file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Endpoint: {args.base_url}")
    print(f"Trials: {args.trials}\n")

    # Test 1 -- Determinism
    det_results = run_determinism_test(args.base_url, args.model, args.trials)
    print_determinism_report(det_results, args.trials)

    # Test 2 -- Output validation
    val_results = run_validation_test(args.base_url, args.model)
    print_validation_report(val_results)

    # Persist
    save_results(det_results, val_results, args.model, args.trials)

    # Exit code: non-zero if any check failed
    all_det = all(r["passed"] for r in det_results)
    all_val = all(r["regex_pass"] and r["schema_pass"] for r in val_results)
    sys.exit(0 if (all_det and all_val) else 1)


if __name__ == "__main__":
    main()
