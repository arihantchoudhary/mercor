#!/usr/bin/env python3
"""Test completions endpoint with logprobs as integer."""
import requests
import json

BASE = "http://localhost:11434"


def test_completions_int_logprobs():
    """logprobs should be an integer (number of top logprobs to return)."""
    resp = requests.post(f"{BASE}/v1/completions", json={
        "model": "qwen2.5:7b",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    print("=== /v1/completions with logprobs=20 ===")
    print(json.dumps(data, indent=2))
    print()


def test_completions_with_echo():
    """Test completions with echo to get prompt tokens back."""
    resp = requests.post(f"{BASE}/v1/completions", json={
        "model": "qwen2.5:7b",
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "logprobs": 5,
        "echo": True,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    print("=== /v1/completions with echo=True, logprobs=5 ===")
    print(json.dumps(data, indent=2))
    print()


def test_chat_completions_correct_approach():
    """
    The correct approach for HellaSwag:
    - Use /api/generate with raw=True to get raw text completion
    - The prompt is the context, the model generates tokens
    - We need logprobs from the generation

    Since /api/generate doesn't give logprobs, let's try a workaround:
    Use /v1/chat/completions but structure the messages so the 'assistant'
    has already started producing the context text.
    """
    context = "A person is seen sitting on a bench. They"
    continuations = [
        " stand up and walk away from the bench.",
        " begin to fly through the air like a bird.",
        " transform into a giant purple elephant.",
        " eat the bench using only their teeth.",
    ]

    print("=== Using assistant prefix trick ===")
    # Try: put context as the start of assistant's message
    for i, cont in enumerate(continuations):
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "assistant", "content": context},
            ],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        }, timeout=30)
        data = resp.json()
        if "error" in data:
            print(f"  Error: {data['error']}")
            break
        top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

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

        print(f"  Cont {i}: first_word={repr(first_word)}, match={repr(best_token)}, logprob={best_logprob:.4f}")
        print(f"    Top 5: {[(e['token'], round(e['logprob'], 3)) for e in top_logprobs[:5]]}")
    print()


def test_raw_api_generate_with_multiple_tokens():
    """Use /api/generate raw mode to see the text completion behavior."""
    context = "A person is seen sitting on a bench. They"
    resp = requests.post(f"{BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": context,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 20,
        },
    }, timeout=30)
    data = resp.json()
    print("=== Raw /api/generate continuation ===")
    print(f"Context: {repr(context)}")
    print(f"Generated: {repr(data.get('response', ''))}")
    print()


if __name__ == "__main__":
    test_completions_int_logprobs()
    test_completions_with_echo()
    test_chat_completions_correct_approach()
    test_raw_api_generate_with_multiple_tokens()
