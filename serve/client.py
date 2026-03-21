#!/usr/bin/env python3
"""
Sample client that runs several prompt generations against the Ollama endpoint.

Usage:
    python client.py
    python client.py --model llama3.2:3b
    python client.py --base-url http://localhost:11434
"""

import argparse
import sys
import time

import requests
from requests.adapters import HTTPAdapter


OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

SAMPLE_PROMPTS = [
    # Short factual
    "What is the capital of France? Answer in one sentence.",
    # Reasoning
    "If a train travels at 60 mph for 2.5 hours, how far does it go? Show your work.",
    # Creative
    "Write a haiku about machine learning.",
    # Code generation
    "Write a Python function that checks if a string is a palindrome.",
    # Multi-turn style instruction
    "Explain the difference between TCP and UDP in exactly three bullet points.",
]

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


def generate(base_url: str, model: str, prompt: str, stream: bool = False) -> dict:
    """Send a generation request to the Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }
    session = _get_session()
    resp = session.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def chat(base_url: str, model: str, messages: list[dict]) -> dict:
    """Send a chat completion request (OpenAI-compatible endpoint)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }
    session = _get_session()
    resp = session.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def run_demo(base_url: str, model: str):
    """Run sample prompts and display results."""
    print(f"Model: {model}")
    print(f"Endpoint: {base_url}")
    print("=" * 60)

    session = _get_session()

    # --- Test 1: Basic generations ---
    print("\n[1/3] Running basic prompt generations\n")
    for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
        print(f"--- Prompt {i}/{len(SAMPLE_PROMPTS)} ---")
        print(f"Q: {prompt}")
        t0 = time.time()
        result = generate(base_url, model, prompt)
        elapsed = time.time() - t0
        response_text = result.get("response", "")
        total_tokens = result.get("eval_count", 0)
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(f"A: {response_text.strip()}")
        print(f"   [{elapsed:.2f}s | {total_tokens} tokens | {tps:.1f} tok/s]\n")

    # --- Test 2: Chat-style multi-turn ---
    print("[2/3] Running chat-style multi-turn conversation\n")
    messages = [
        {"role": "user", "content": "What are the three laws of thermodynamics?"},
    ]
    t0 = time.time()
    result = chat(base_url, model, messages)
    elapsed = time.time() - t0
    assistant_msg = result.get("message", {}).get("content", "")
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {assistant_msg.strip()}")
    print(f"   [{elapsed:.2f}s]\n")

    # Follow-up
    messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": "Summarize that in one sentence."})
    t0 = time.time()
    result = chat(base_url, model, messages)
    elapsed = time.time() - t0
    assistant_msg = result.get("message", {}).get("content", "")
    print(f"User: Summarize that in one sentence.")
    print(f"Assistant: {assistant_msg.strip()}")
    print(f"   [{elapsed:.2f}s]\n")

    # --- Test 3: Deterministic mode ---
    print("[3/3] Testing deterministic output (temperature=0, same seed)\n")
    det_prompt = "What is 2 + 2? Answer with just the number."
    results = []
    for trial in range(3):
        payload = {
            "model": model,
            "prompt": det_prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 16,
            },
        }
        resp = session.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "").strip()
        results.append(text)
        print(f"   Trial {trial+1}: {text!r}")

    if len(set(results)) == 1:
        print("   => All outputs identical (deterministic)")
    else:
        print("   => WARNING: outputs differ (nondeterministic)")

    print("\n" + "=" * 60)
    print("Demo complete.")


def main():
    parser = argparse.ArgumentParser(description="Sample Ollama client")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=OLLAMA_BASE, help=f"Ollama base URL (default: {OLLAMA_BASE})")
    args = parser.parse_args()

    # Quick health check
    session = _get_session()
    try:
        resp = session.get(f"{args.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach Ollama at {args.base_url}", file=sys.stderr)
        print("Start the server first: python serve.py (or make serve)", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"ERROR: Ollama health check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    run_demo(args.base_url, args.model)


if __name__ == "__main__":
    main()
