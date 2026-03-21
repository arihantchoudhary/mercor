#!/usr/bin/env python3
"""
Test the correct approach for loglikelihood scoring via Ollama.

Strategy: Token-by-token scoring using /v1/chat/completions.
For each token in the continuation, we:
1. Set up messages so that context is what the model conditions on
2. Generate 1 token with logprobs
3. Look up the actual continuation token in top_logprobs
4. Advance by adding that token to the context

To avoid the chat template issue, we can try putting the context
as an assistant turn (partially generated), forcing the model to
continue from there.

Alternative: Use the prefix approach where we set up:
- system: "You are a text completion engine. You will be given text and must continue it exactly."
- user: [context text]
- Then look for continuation tokens in the generated logprobs
"""
import requests
import json

BASE = "http://localhost:11434"


def score_continuation_chat_prefix(context, continuation, n_score_tokens=3):
    """
    Score a continuation using the chat completions endpoint.

    For HellaSwag, we use this approach:
    1. Send context as assistant message (prefix/partial generation)
    2. Generate n tokens with logprobs
    3. For each token position, find the continuation's corresponding token
       in top_logprobs and sum up

    Since we don't know the exact tokenization, we match by checking if
    the continuation text starts with any of the top logprob tokens.
    """
    total_logprob = 0.0
    is_greedy = True
    remaining_cont = continuation

    for step in range(n_score_tokens):
        if not remaining_cont.strip():
            break

        # Build messages - use assistant prefix approach
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
            return -100.0, False

        top_logprobs = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        generated_token = data["choices"][0]["logprobs"]["content"][0]["token"]

        # Find the best matching token from top_logprobs that matches
        # the start of remaining_cont
        best_match = None
        best_logprob = None
        best_rank = None

        for rank, entry in enumerate(top_logprobs):
            tok = entry["token"]
            # Check if remaining continuation starts with this token
            if remaining_cont.startswith(tok):
                if best_match is None or len(tok) > len(best_match):
                    best_match = tok
                    best_logprob = entry["logprob"]
                    best_rank = rank

        if best_match is not None:
            total_logprob += best_logprob
            if best_rank != 0:
                is_greedy = False
            # Advance context and remaining continuation
            context = context + best_match
            remaining_cont = remaining_cont[len(best_match):]
        else:
            # Token not found in top-20, assign penalty
            total_logprob += -15.0
            is_greedy = False
            # Try to advance using the first character
            # Use the generated token's length as reference
            advance = max(1, len(generated_token))
            context = context + remaining_cont[:advance]
            remaining_cont = remaining_cont[advance:]

    return total_logprob, is_greedy


def score_continuation_user_msg(context, continuation, n_score_tokens=3):
    """
    Score using user message approach.
    Send context as user message, look for continuation tokens in response.
    """
    total_logprob = 0.0
    is_greedy = True
    remaining_cont = continuation

    # For user-message approach, we can't easily advance token by token
    # because the model wraps context in a template.
    # Instead, generate multiple tokens and match against continuation.
    resp = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "system", "content": "Continue the following text naturally. Just output the continuation, nothing else."},
            {"role": "user", "content": context},
        ],
        "max_tokens": min(n_score_tokens, 10),
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }, timeout=30)
    data = resp.json()

    if "error" in data:
        return -100.0, False

    content_logprobs = data["choices"][0]["logprobs"]["content"]

    for token_info in content_logprobs:
        if not remaining_cont.strip():
            break

        top_logprobs = token_info["top_logprobs"]

        best_match = None
        best_logprob = None
        best_rank = None

        for rank, entry in enumerate(top_logprobs):
            tok = entry["token"]
            if remaining_cont.startswith(tok):
                if best_match is None or len(tok) > len(best_match):
                    best_match = tok
                    best_logprob = entry["logprob"]
                    best_rank = rank

        if best_match is not None:
            total_logprob += best_logprob
            if best_rank != 0:
                is_greedy = False
            remaining_cont = remaining_cont[len(best_match):]
        else:
            total_logprob += -15.0
            is_greedy = False
            # advance by generated token length
            gen_tok = token_info["token"]
            remaining_cont = remaining_cont[max(1, len(gen_tok)):]

    return total_logprob, is_greedy


def test_hellaswag_example():
    """Test with actual HellaSwag data."""
    context = "Roof shingle removal: A man is sitting on a roof. He"
    continuations = [
        " is using wrap to wrap a pair of skis.",
        " is ripping level tiles off.",
        " is holding a rubik's cube.",
        " starts pulling up roofing on a roof.",
    ]
    correct = 3  # "starts pulling up roofing on a roof"

    print("=== HellaSwag Example 1 ===")
    print(f"Context: {repr(context)}")
    print(f"Correct answer: {correct}")
    print()

    print("--- Assistant prefix approach (3 tokens) ---")
    scores_prefix = []
    for i, cont in enumerate(continuations):
        score, greedy = score_continuation_chat_prefix(context, cont, n_score_tokens=3)
        scores_prefix.append(score)
        print(f"  [{i}] score={score:.4f} greedy={greedy} | {repr(cont[:60])}")
    predicted = max(range(len(scores_prefix)), key=lambda x: scores_prefix[x])
    print(f"  Predicted: {predicted}, Correct: {correct}, {'OK' if predicted == correct else 'WRONG'}")
    print()

    print("--- User message approach (5 tokens) ---")
    scores_user = []
    for i, cont in enumerate(continuations):
        score, greedy = score_continuation_user_msg(context, cont, n_score_tokens=5)
        scores_user.append(score)
        print(f"  [{i}] score={score:.4f} greedy={greedy} | {repr(cont[:60])}")
    predicted = max(range(len(scores_user)), key=lambda x: scores_user[x])
    print(f"  Predicted: {predicted}, Correct: {correct}, {'OK' if predicted == correct else 'WRONG'}")
    print()

    # Example 2
    context2 = "Clean and jerk: A lady walks to a barbell. She bends down and grabs the pole. The lady"
    continuations2 = [
        " swings and lands in her arms.",
        " pulls the barbell forward.",
        " pulls a rope attached to the barbell.",
        " stands and lifts the weight over her head.",
    ]
    correct2 = 3  # "stands and lifts the weight over her head"

    print("=== HellaSwag Example 2 ===")
    print(f"Context: {repr(context2)}")
    print(f"Correct answer: {correct2}")
    print()

    print("--- Assistant prefix approach (3 tokens) ---")
    scores = []
    for i, cont in enumerate(continuations2):
        score, greedy = score_continuation_chat_prefix(context2, cont, n_score_tokens=3)
        scores.append(score)
        print(f"  [{i}] score={score:.4f} greedy={greedy} | {repr(cont[:60])}")
    predicted = max(range(len(scores)), key=lambda x: scores[x])
    print(f"  Predicted: {predicted}, Correct: {correct2}, {'OK' if predicted == correct2 else 'WRONG'}")
    print()


if __name__ == "__main__":
    test_hellaswag_example()
