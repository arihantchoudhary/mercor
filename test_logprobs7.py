#!/usr/bin/env python3
"""
Debug: What tokens does the model predict when continuing from the context
via assistant prefix?
"""
import requests
import json

BASE = "http://localhost:11434"


def show_predictions(context, n_tokens=5):
    """Show what the model predicts for each token position."""
    print(f"Context: {repr(context)}")
    current = context

    for step in range(n_tokens):
        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "assistant", "content": current},
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

        token_info = data["choices"][0]["logprobs"]["content"][0]
        gen_tok = token_info["token"]
        gen_lp = token_info["logprob"]

        print(f"\n  Step {step}: generated {repr(gen_tok)} (logprob={gen_lp:.4f})")
        print(f"  Top 10:")
        for rank, entry in enumerate(token_info["top_logprobs"][:10]):
            print(f"    [{rank}] {repr(entry['token']):20s} logprob={entry['logprob']:.4f}")

        current += gen_tok

    print(f"\nFull generation: {repr(current[len(context):])}")
    print()


def show_predictions_multi(context, n_tokens=10):
    """Show what the model generates in one shot (multi-token)."""
    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "assistant", "content": context},
        ],
        "max_tokens": n_tokens,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()

    print(f"Context: {repr(context)}")
    content = data["choices"][0]["logprobs"]["content"]
    gen_text = ""
    for i, ti in enumerate(content):
        gen_text += ti["token"]
        print(f"  Token {i}: {repr(ti['token']):15s} logprob={ti['logprob']:.4f}")
    print(f"  Full: {repr(gen_text)}")
    print()


if __name__ == "__main__":
    ctx1 = "Roof shingle removal: A man is sitting on a roof. He"
    print("=== Token-by-token predictions ===")
    show_predictions(ctx1, n_tokens=5)

    print("=== Multi-token predictions ===")
    show_predictions_multi(ctx1, n_tokens=10)

    ctx2 = "Clean and jerk: A lady walks to a barbell. She bends down and grabs the pole. The lady"
    print("=== Token-by-token predictions (ctx2) ===")
    show_predictions(ctx2, n_tokens=5)

    print("=== Multi-token predictions (ctx2) ===")
    show_predictions_multi(ctx2, n_tokens=10)
