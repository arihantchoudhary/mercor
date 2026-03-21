#!/usr/bin/env python3
"""Test raw generation with logprobs."""
import requests
import json

BASE = "http://localhost:11434"


def test_raw_generate():
    """Test /api/generate with raw=true to see if we get logprobs."""
    resp = requests.post(f"{BASE}/api/generate", json={
        "model": "qwen2.5:7b",
        "prompt": "The capital of France is",
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 3,
        },
    }, timeout=30)
    data = resp.json()
    print("=== /api/generate raw response ===")
    print(f"Response text: {repr(data.get('response', ''))}")
    # Check if there are any logprob-related fields
    for key in data:
        if key not in ('response', 'model', 'created_at', 'done', 'done_reason',
                       'context', 'total_duration', 'load_duration',
                       'prompt_eval_count', 'prompt_eval_duration',
                       'eval_count', 'eval_duration'):
            print(f"  Extra key: {key} = {data[key]}")
    print()


def test_completions_with_logprobs():
    """Test /v1/completions endpoint (non-chat) with logprobs."""
    resp = requests.post(f"{BASE}/v1/completions", json={
        "model": "qwen2.5:7b",
        "prompt": "The capital of France is",
        "max_tokens": 3,
        "logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    print("=== /v1/completions response ===")
    print(json.dumps(data, indent=2))
    print()


def test_chat_with_raw_text():
    """Use the chat endpoint but put the raw text as assistant prefix.

    The idea: use system message to set up context, then put the text
    we want to continue as assistant's partial message.
    Actually Ollama may not support that. Let's try another approach:
    put the full context in a system message.
    """
    context = "A person is seen sitting on a bench. They"
    continuations = [
        " stand up and walk away from the bench.",
        " begin to fly through the air like a bird.",
        " transform into a giant purple elephant.",
        " eat the bench using only their teeth.",
    ]

    print("=== Raw text as system prompt ===")
    for i, cont in enumerate(continuations):
        # Put context as system message, ask model to continue
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "system", "content": f"Continue this text with the next few words. Just output the continuation, nothing else: {context}"},
            ],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        }, timeout=30)
        data = resp.json()
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


def test_completions_logprobs_field():
    """Test completions endpoint - maybe logprobs is in a different format."""
    resp = requests.post(f"{BASE}/v1/completions", json={
        "model": "qwen2.5:7b",
        "prompt": "The capital of France is",
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()
    print("=== /v1/completions with logprobs=True ===")
    print(json.dumps(data, indent=2))
    print()


if __name__ == "__main__":
    test_raw_generate()
    test_completions_with_logprobs()
    test_completions_logprobs_field()
    test_chat_with_raw_text()
