#!/usr/bin/env python3
"""
Part E -- Improved evaluation for prog_lang_id benchmark.

Applies inference-time techniques to boost accuracy over the baseline:
  1. Expert system prompt (persona priming)
  2. Few-shot examples in the prompt
  3. Chain-of-thought reasoning (think step-by-step, then answer)
  4. Self-consistency via majority vote over multiple samples

No fine-tuning is performed -- all improvements are purely at inference time.
Saves results to results/improved.json.
"""

import json
import os
import re
import sys
import time
from collections import Counter

import requests

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "qwen2.5:7b")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_runner", "custom_benchmark.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "improved.json")
TIMEOUT = 300

# Number of samples for self-consistency majority vote
NUM_SAMPLES = 5

# Persistent session for connection pooling
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# ── System prompt: expert persona ────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert software engineer and programming language specialist "
    "with 20 years of experience across dozens of languages. You can identify "
    "any programming language from a short code snippet by examining syntax, "
    "keywords, standard library usage, and idioms."
)

# ── Few-shot exemplars (deliberately chosen to NOT overlap with benchmark) ──
FEW_SHOT_EXAMPLES = [
    {
        "snippet": 'val greeting = "Hello"\nprintln(greeting.reversed())',
        "reasoning": (
            "I see `val` for immutable variable declaration and `println` as a "
            "top-level function. The `.reversed()` extension function and the "
            "overall syntax are characteristic of Kotlin."
        ),
        "answer": "Kotlin",
    },
    {
        "snippet": "puts [expr {2 ** 10}]",
        "reasoning": (
            "The `puts` command for output and `[expr {...}]` for expression "
            "evaluation are hallmarks of Tcl (Tool Command Language)."
        ),
        "answer": "Tcl",
    },
    {
        "snippet": '#include <stdio.h>\nint main() { printf("hi"); return 0; }',
        "reasoning": (
            "The `#include <stdio.h>` preprocessor directive, `int main()` "
            "entry point, and `printf` function are classic C. There are no "
            "C++ features like `iostream`, classes, or namespaces."
        ),
        "answer": "C",
    },
    {
        "snippet": 'using System;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        Console.WriteLine("Hello");\n    }\n}',
        "reasoning": (
            "The `using System` and `using System.Linq` directives, `Console.WriteLine`, "
            "and `class Program` with `static void Main()` are hallmarks of C#. "
            "Although it uses `class` like Java, the `using` keyword (not `import`), "
            "`Console.WriteLine` (not `System.out.println`), and LINQ namespace "
            "are unique to C#."
        ),
        "answer": "C#",
    },
]

# ── Canonical language names for fuzzy matching ──────────────────────────
CANONICAL = {
    "c++": "C++", "cpp": "C++",
    "c#": "C#", "csharp": "C#", "c sharp": "C#",
    "javascript": "JavaScript", "js": "JavaScript",
    "typescript": "TypeScript", "ts": "TypeScript",
    "python": "Python", "py": "Python",
    "rust": "Rust",
    "go": "Go", "golang": "Go",
    "java": "Java",
    "ruby": "Ruby", "rb": "Ruby",
    "swift": "Swift",
    "kotlin": "Kotlin", "kt": "Kotlin",
    "php": "PHP",
    "sql": "SQL",
    "elixir": "Elixir", "ex": "Elixir",
    "scala": "Scala",
    "haskell": "Haskell", "hs": "Haskell",
    "bash": "Bash", "shell": "Bash", "sh": "Bash",
    "c": "C",
}


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH) as f:
        return json.load(f)["examples"]


def build_few_shot_block() -> str:
    """Format the few-shot examples into a prompt section."""
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(
            f"Example {i}:\n"
            f"```\n{ex['snippet']}\n```\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Answer: {ex['answer']}"
        )
    return "\n\n".join(parts)


def build_user_prompt(snippet: str) -> str:
    """Construct the full user message with few-shot + CoT instruction."""
    few_shot = build_few_shot_block()
    return (
        f"Identify the programming language of the code snippet below.\n\n"
        f"Here are some worked examples:\n\n"
        f"{few_shot}\n\n"
        f"---\n\n"
        f"Now identify this snippet. Think step by step: examine the syntax, "
        f"keywords, and idioms, then state your reasoning. On the very last "
        f"line, write ONLY the language name after \"Answer: \".\n\n"
        f"```\n{snippet}\n```"
    )


def query_model(user_prompt: str, temperature: float = 0.0) -> str:
    """Send a chat request with the expert system prompt."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": 42,
            "num_predict": 256,
        },
    }
    resp = _session.post(
        f"{OLLAMA_BASE}/api/chat", json=payload, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_language(raw_response: str) -> str:
    """Extract the language name from a CoT response.

    Looks for the last line matching 'Answer: <language>' first,
    then falls back to the last non-empty line.
    """
    # Try explicit "Answer:" pattern (case-insensitive)
    matches = re.findall(r"(?i)^answer:\s*(.+)$", raw_response, re.MULTILINE)
    if matches:
        candidate = matches[-1].strip().strip("*`#.-)")
        # Strip parenthetical qualifiers like "JavaScript (using Node.js)"
        # Also handles unclosed parens: "JavaScript (using Node.js"
        candidate = re.sub(r"\s*\(.*$", "", candidate).strip()
        # Strip slash-prefixed alternatives: "Node.js/JavaScript" -> take last part
        if "/" in candidate:
            candidate = candidate.split("/")[-1].strip()
        return canonicalize(candidate)

    # Fallback: last non-empty line
    for line in reversed(raw_response.splitlines()):
        line = line.strip().strip("*`#.-)")
        if line:
            line = re.sub(r"\s*\(.*$", "", line).strip()
            if "/" in line:
                line = line.split("/")[-1].strip()
            return canonicalize(line)

    return raw_response.strip()


def canonicalize(name: str) -> str:
    """Normalize a language name to its canonical form."""
    key = name.lower().strip()
    if key in CANONICAL:
        return CANONICAL[key]
    # Title-case as last resort
    return name.strip()


def majority_vote(predictions: list[str]) -> str:
    """Return the most common prediction (majority vote)."""
    counts = Counter(p.lower() for p in predictions)
    winner_lower = counts.most_common(1)[0][0]
    # Return the original-cased version
    for p in predictions:
        if p.lower() == winner_lower:
            return p
    return predictions[0]


def run() -> dict:
    examples = load_benchmark()
    results = []
    correct = 0

    print(f"Improved eval  |  model={MODEL}  |  {len(examples)} examples")
    print(f"Techniques: expert-persona + few-shot + CoT + majority-vote (k={NUM_SAMPLES})")
    print("-" * 70)

    for i, ex in enumerate(examples):
        user_prompt = build_user_prompt(ex["snippet"])
        t0 = time.time()

        # Self-consistency: generate multiple samples sequentially
        # (Ollama serialises inference, so parallel requests just queue up)
        sample_preds = []

        # Deterministic sample
        raw_det = query_model(user_prompt, temperature=0.0)
        sample_preds.append(extract_language(raw_det))

        # Stochastic samples with temperature for diversity
        for k in range(NUM_SAMPLES - 1):
            raw_k = query_model(user_prompt, temperature=0.6)
            sample_preds.append(extract_language(raw_k))

        predicted = majority_vote(sample_preds)
        elapsed = time.time() - t0

        match = predicted.lower() == ex["language"].lower()
        if match:
            correct += 1

        status = "OK" if match else "MISS"
        vote_str = ", ".join(sample_preds)
        print(f"  [{i+1:2d}/{len(examples)}] {status}  expected={ex['language']:<12s}  "
              f"voted={predicted:<12s}  samples=[{vote_str}]  ({elapsed:.1f}s)")

        results.append({
            "index": i,
            "expected": ex["language"],
            "predicted": predicted,
            "samples": sample_preds,
            "correct": match,
            "elapsed_s": round(elapsed, 2),
        })

    accuracy = correct / len(examples)
    print("-" * 70)
    print(f"Improved accuracy: {correct}/{len(examples)} = {accuracy:.1%}")

    output = {
        "method": "improved",
        "model": MODEL,
        "techniques": [
            "expert_system_prompt",
            "few_shot_examples",
            "chain_of_thought",
            "self_consistency_majority_vote",
        ],
        "num_samples": NUM_SAMPLES,
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
