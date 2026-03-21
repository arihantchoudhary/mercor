#!/usr/bin/env python3
"""
Part E -- Baseline evaluation for prog_lang_id benchmark.

Runs each snippet through qwen2.5:7b with minimal prompting (temperature=0,
no system prompt, no chain-of-thought).  Saves results to results/baseline.json.
"""

import json
import os
import sys
import time

import requests

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "qwen2.5:7b")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_runner", "custom_benchmark.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "baseline.json")
TIMEOUT = 300

# Persistent session for connection pooling
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)["examples"]


def query_model(prompt: str) -> str:
    """Send a bare prompt to Ollama /api/chat and return the response text."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 32,
        },
    }
    resp = _session.post(
        f"{OLLAMA_BASE}/api/chat", json=payload, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_language(raw_response: str) -> str:
    """Pull the language name from the model's response.

    The model is asked to reply with only the language name, but it may
    add punctuation, articles, or extra words.  We take the first
    recognisable line.
    """
    # Take first non-empty line, strip markdown/punctuation
    for line in raw_response.splitlines():
        line = line.strip().strip("*`#.-)")
        if line:
            return line
    return raw_response.strip()


def run() -> dict:
    examples = load_benchmark()
    results = []
    correct = 0

    print(f"Baseline eval  |  model={MODEL}  |  {len(examples)} examples")
    print("-" * 60)

    for i, ex in enumerate(examples):
        prompt = (
            f"What programming language is this code written in?\n"
            f"Reply with only the language name, nothing else.\n\n"
            f"```\n{ex['snippet']}\n```"
        )

        t0 = time.time()
        raw = query_model(prompt)
        elapsed = time.time() - t0
        predicted = extract_language(raw)

        match = predicted.lower() == ex["language"].lower()
        if match:
            correct += 1

        status = "OK" if match else "MISS"
        print(f"  [{i+1:2d}/{len(examples)}] {status}  expected={ex['language']:<12s}  got={predicted:<16s}  ({elapsed:.1f}s)")

        results.append({
            "index": i,
            "expected": ex["language"],
            "predicted": predicted,
            "raw_response": raw,
            "correct": match,
            "elapsed_s": round(elapsed, 2),
        })

    accuracy = correct / len(examples)
    print("-" * 60)
    print(f"Baseline accuracy: {correct}/{len(examples)} = {accuracy:.1%}")

    output = {
        "method": "baseline",
        "model": MODEL,
        "total": len(examples),
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    try:
        _session.get(f"{OLLAMA_BASE}/api/tags", timeout=10).raise_for_status()
    except requests.RequestException:
        print(f"ERROR: Ollama not reachable at {OLLAMA_BASE}", file=sys.stderr)
        sys.exit(1)
    run()
