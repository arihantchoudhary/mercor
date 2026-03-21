#!/usr/bin/env python3
"""Quick test to understand Ollama logprobs behavior."""
import requests
import json

BASE = "http://localhost:11434"

def test_chat_logprobs():
    """Test /v1/chat/completions with logprobs."""
    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "Complete the following text. Output only the continuation."},
            {"role": "user", "content": "The capital of France is"},
        ],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    content = data["choices"][0]["logprobs"]["content"][0]
    print("=== Chat completions test ===")
    print(f"Generated token: {repr(content['token'])}")
    print("Top logprobs:")
    for entry in content["top_logprobs"][:10]:
        print(f"  {repr(entry['token']):20s} -> {entry['logprob']:.4f}")
    print()


def test_token_by_token():
    """Test token-by-token scoring approach.

    For context='The capital of France is', continuation=' Paris'
    We send the context and check if 'Paris' or ' Paris' appears in top logprobs.
    """
    context = "The capital of France is"
    continuation = " Paris"

    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": context}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    content = data["choices"][0]["logprobs"]["content"][0]

    print("=== Token-by-token test ===")
    print(f"Context: {repr(context)}")
    print(f"Continuation: {repr(continuation)}")
    print(f"Generated token: {repr(content['token'])}")
    print("Top logprobs:")
    for entry in content["top_logprobs"][:20]:
        print(f"  {repr(entry['token']):20s} -> {entry['logprob']:.4f}")

    # Check if continuation's first chars match any token
    cont_lower = continuation.strip().lower()
    for entry in content["top_logprobs"]:
        tok = entry["token"].strip().lower()
        if cont_lower.startswith(tok) or tok.startswith(cont_lower):
            print(f"\n  MATCH: {repr(entry['token'])} -> {entry['logprob']:.4f}")
    print()


def test_hellaswag_like():
    """Test with a HellaSwag-like example to see if first-token approach works."""
    # HellaSwag presents context + multiple continuations
    context = "A person is seen sitting on a bench. They"
    continuations = [
        " stand up and walk away from the bench.",
        " begin to fly through the air like a bird.",
        " transform into a giant purple elephant.",
        " eat the bench using only their teeth.",
    ]

    print("=== HellaSwag-like test ===")
    print(f"Context: {repr(context)}")

    for i, cont in enumerate(continuations):
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": context}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        }, timeout=30)
        data = resp.json()
        top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

        # Find matching token for this continuation's first token
        cont_stripped = cont.strip()
        first_word = cont_stripped.split()[0].lower()

        best_logprob = -100.0
        best_token = None
        for entry in top_logprobs:
            tok = entry["token"].strip().lower()
            if first_word.startswith(tok) or tok.startswith(first_word):
                if entry["logprob"] > best_logprob:
                    best_logprob = entry["logprob"]
                    best_token = entry["token"]

        print(f"  Continuation {i}: {repr(cont[:50])}...")
        print(f"    First word: {repr(first_word)}, Best match: {repr(best_token)}, logprob: {best_logprob:.4f}")
    print()


if __name__ == "__main__":
    test_chat_logprobs()
    test_token_by_token()
    test_hellaswag_like()
