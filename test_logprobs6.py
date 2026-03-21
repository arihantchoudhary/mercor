#!/usr/bin/env python3
"""
Test approach: Use /v1/chat/completions endpoint with a carefully crafted prompt
to get logprobs that serve as a ranking signal for continuations.

Key insight: For ranking MCQ continuations, we don't need exact logprobs.
We need a RELATIVE score that ranks the correct continuation higher.

Approach: "Completion likelihood estimation"
- Send: "Complete this text: [context]. The next part is: [continuation]"
- Ask model to rate it (but that's slow and unreliable)

Better approach: Direct generation matching
- For each continuation, send context and generate tokens
- Score = how well the generated tokens match the continuation
- Use logprobs to weight the matching

Actually, let me try the SIMPLEST approach that might work:
- For each (context, continuation), compute the generation probability
  by generating from context with logprobs and matching against continuation
- The matching should be character-level, not word-level
"""
import requests
import json

BASE = "http://localhost:11434"


def score_continuation_v2(context, continuation, max_gen_tokens=15):
    """
    Score a continuation by generating from context and matching tokens.

    Uses the assistant prefix trick but with better matching:
    - Generate multiple tokens from the context
    - For each generated token, check if it (or something close to it)
      appears in the continuation
    - Track position in the continuation and score based on logprobs
    """
    # Use assistant message to condition on context
    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "assistant", "content": context},
        ],
        "max_tokens": max_gen_tokens,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=60)
    data = resp.json()

    if "error" in data:
        return -100.0, False

    content_logprobs = data["choices"][0]["logprobs"]["content"]

    total_logprob = 0.0
    is_greedy = True
    cont_pos = 0  # Position in continuation string

    for token_info in content_logprobs:
        if cont_pos >= len(continuation):
            break

        remaining = continuation[cont_pos:]
        top_logprobs = token_info["top_logprobs"]

        # Find best matching token that matches the start of remaining continuation
        best_match = None
        best_logprob = None
        best_rank = None

        for rank, entry in enumerate(top_logprobs):
            tok = entry["token"]
            if remaining.startswith(tok) and len(tok) > 0:
                if best_match is None or len(tok) > len(best_match):
                    best_match = tok
                    best_logprob = entry["logprob"]
                    best_rank = rank

        if best_match is not None:
            total_logprob += best_logprob
            if best_rank != 0:
                is_greedy = False
            cont_pos += len(best_match)
        else:
            # No match found - heavy penalty
            total_logprob += -20.0
            is_greedy = False
            # Advance by generated token length to keep going
            gen_tok = token_info["token"]
            cont_pos += max(1, len(gen_tok))

    # Penalize unmatched remainder
    if cont_pos < len(continuation):
        unmatched_chars = len(continuation) - cont_pos
        total_logprob += -0.5 * unmatched_chars

    return total_logprob, is_greedy


def score_continuation_v3(context, continuation, n_tokens=5):
    """
    Token-by-token scoring with assistant prefix.

    For each step:
    1. Send context (growing) as assistant message
    2. Get top 20 logprobs for next token
    3. Find continuation token in top_logprobs
    4. If found, use its logprob; if not, use -20 penalty
    5. Advance context by the matched/expected token
    """
    total_logprob = 0.0
    is_greedy = True
    current_context = context
    remaining_cont = continuation
    tokens_scored = 0

    for step in range(n_tokens):
        if not remaining_cont:
            break

        resp = requests.post(f"{BASE}/v1/chat/completions", json={
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "assistant", "content": current_context},
            ],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        }, timeout=30)
        data = resp.json()

        if "error" in data:
            total_logprob += -20.0
            break

        top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

        # Find best matching token
        best_match = None
        best_logprob = None
        best_rank = None

        for rank, entry in enumerate(top_logprobs):
            tok = entry["token"]
            if remaining_cont.startswith(tok) and len(tok) > 0:
                if best_match is None or len(tok) > len(best_match):
                    best_match = tok
                    best_logprob = entry["logprob"]
                    best_rank = rank

        if best_match is not None:
            total_logprob += best_logprob
            if best_rank != 0:
                is_greedy = False
            current_context += best_match
            remaining_cont = remaining_cont[len(best_match):]
            tokens_scored += 1
        else:
            # Not found - use the generated token to advance
            gen_tok = data["choices"][0]["logprobs"]["content"][0]["token"]
            total_logprob += -20.0
            is_greedy = False
            advance = max(1, len(gen_tok))
            current_context += remaining_cont[:advance]
            remaining_cont = remaining_cont[advance:]
            tokens_scored += 1

    return total_logprob, is_greedy


def test_all():
    examples = [
        {
            "context": "Roof shingle removal: A man is sitting on a roof. He",
            "continuations": [
                " is using wrap to wrap a pair of skis.",
                " is ripping level tiles off.",
                " is holding a rubik's cube.",
                " starts pulling up roofing on a roof.",
            ],
            "correct": 3,
        },
        {
            "context": "Clean and jerk: A lady walks to a barbell. She bends down and grabs the pole. The lady",
            "continuations": [
                " swings and lands in her arms.",
                " pulls the barbell forward.",
                " pulls a rope attached to the barbell.",
                " stands and lifts the weight over her head.",
            ],
            "correct": 3,
        },
    ]

    for ex_idx, ex in enumerate(examples):
        context = ex["context"]
        continuations = ex["continuations"]
        correct = ex["correct"]

        print(f"\n=== Example {ex_idx + 1} (correct={correct}) ===")
        print(f"Context: {repr(context[:80])}")

        print("\n--- v2: Multi-token generation matching (15 tokens) ---")
        scores = []
        for i, cont in enumerate(continuations):
            score, greedy = score_continuation_v2(context, cont, max_gen_tokens=15)
            scores.append(score)
            marker = " <-- CORRECT" if i == correct else ""
            print(f"  [{i}] score={score:8.4f} | {repr(cont[:50])}{marker}")
        predicted = max(range(len(scores)), key=lambda x: scores[x])
        print(f"  Predicted: {predicted} {'OK' if predicted == correct else 'WRONG'}")

        print("\n--- v3: Token-by-token (5 tokens) ---")
        scores = []
        for i, cont in enumerate(continuations):
            score, greedy = score_continuation_v3(context, cont, n_tokens=5)
            scores.append(score)
            marker = " <-- CORRECT" if i == correct else ""
            print(f"  [{i}] score={score:8.4f} | {repr(cont[:50])}{marker}")
        predicted = max(range(len(scores)), key=lambda x: scores[x])
        print(f"  Predicted: {predicted} {'OK' if predicted == correct else 'WRONG'}")


if __name__ == "__main__":
    test_all()
