#!/usr/bin/env python3
"""
Serve an LLM locally via Ollama.

Usage:
    python serve.py                  # uses default model (qwen2.5:7b)
    python serve.py --model llama3.2:3b
    python serve.py --pull           # pull the model first, then serve

Ollama exposes an OpenAI-compatible API at http://localhost:11434
"""

import argparse
import subprocess
import sys
import time
import urllib.request
import json


OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"


def is_ollama_running() -> bool:
    """Check if the ollama server is already responding."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start_ollama_server():
    """Start the ollama serve process in the background."""
    print("Starting ollama server...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for the server to be ready
    for _ in range(30):
        if is_ollama_running():
            print(f"Ollama server running at {OLLAMA_BASE}")
            return proc
        time.sleep(1)
    print("ERROR: ollama server did not start within 30 seconds", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def pull_model(model: str):
    """Pull/download a model if not already available."""
    print(f"Pulling model '{model}' (this may take a while on first run)...")
    result = subprocess.run(["ollama", "pull", model], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: failed to pull model '{model}'", file=sys.stderr)
        sys.exit(1)
    print(f"Model '{model}' ready.")


def model_available(model: str) -> bool:
    """Check if the model is already downloaded."""
    try:
        data = urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5).read()
        tags = json.loads(data)
        return any(m["name"] == model for m in tags.get("models", []))
    except Exception:
        return False


def warm_up(model: str):
    """Send a short request to load the model into memory."""
    print(f"Warming up model '{model}'...")
    payload = json.dumps({
        "model": model,
        "prompt": "Hello",
        "stream": False,
        "options": {"num_predict": 1},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=120)
        print("Model loaded and ready.")
    except Exception as e:
        print(f"Warning: warm-up request failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Serve an LLM via Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to serve (default: {DEFAULT_MODEL})")
    parser.add_argument("--pull", action="store_true", help="Pull model before serving")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warm-up request")
    args = parser.parse_args()

    # Start server if not already running
    proc = None
    if not is_ollama_running():
        proc = start_ollama_server()
    else:
        print(f"Ollama server already running at {OLLAMA_BASE}")

    # Pull model if requested or not available
    if args.pull or not model_available(args.model):
        pull_model(args.model)

    # Warm up
    if not args.no_warmup:
        warm_up(args.model)

    print(f"\n{'='*50}")
    print(f"Model:    {args.model}")
    print(f"API:      {OLLAMA_BASE}/api/generate")
    print(f"Chat API: {OLLAMA_BASE}/api/chat")
    print(f"OpenAI-compatible: {OLLAMA_BASE}/v1/chat/completions")
    print(f"{'='*50}")
    print("Server is ready. Press Ctrl+C to stop.\n")

    try:
        if proc:
            proc.wait()
        else:
            # Keep the script alive so user can Ctrl+C
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
