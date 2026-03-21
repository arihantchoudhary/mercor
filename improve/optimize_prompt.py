#!/usr/bin/env python3
"""
Part E -- Step 2: Inference-time optimization strategies for HellaSwag.

This module defines five independent optimization strategies that can be
composed or tested in isolation:

    1. PromptTemplateOptimizer  -- test alternative prompt framings
    2. FewShotSelector          -- TF-IDF / BM25-style few-shot selection
    3. ChainOfThoughtPrompter   -- add step-by-step reasoning instructions
    4. SelfConsistencyDecoder   -- k-sample majority voting
    5. PromptEnsembler          -- aggregate answers across prompt variants

Each strategy exposes a uniform interface:
    - build_prompt(example, ...) -> str | list[str]
    - aggregate(responses)       -> str  (where applicable)

Usage (standalone testing):
    python improve/optimize_prompt.py --strategy all --sample 5
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Prompt Template Optimization
# ---------------------------------------------------------------------------

# A library of prompt templates for HellaSwag.  Each template receives
# {ctx} (the context), {choices} (formatted A/B/C/D list), and optionally
# {few_shot_block} (few-shot examples).

PROMPT_TEMPLATES: dict[str, str] = {
    "direct": (
        "{few_shot_block}"
        "Context: {ctx}\n\n"
        "Which ending correctly completes the text above?\n"
        "{choices}\n\n"
        "Answer (A, B, C, or D):"
    ),
    "instruction": (
        "{few_shot_block}"
        "Below is a passage followed by four possible continuations. "
        "Select the most plausible continuation.\n\n"
        "Passage: {ctx}\n\n"
        "{choices}\n\n"
        "The correct answer is:"
    ),
    "what_next": (
        "{few_shot_block}"
        "Read the following scenario and decide what happens next.\n\n"
        "{ctx}\n\n"
        "Options:\n{choices}\n\n"
        "The most likely next event is option:"
    ),
    "sense": (
        "{few_shot_block}"
        "{ctx}\n\n"
        "Which of the following endings makes the most sense?\n"
        "{choices}\n\n"
        "Answer:"
    ),
    "complete": (
        "{few_shot_block}"
        "Complete the following passage by choosing the best continuation.\n\n"
        "{ctx} ...\n\n"
        "{choices}\n\n"
        "Best continuation:"
    ),
}

DEFAULT_TEMPLATE = "direct"


class PromptTemplateOptimizer:
    """Test and select the best prompt template for HellaSwag."""

    def __init__(self, template_name: str = DEFAULT_TEMPLATE):
        if template_name not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(PROMPT_TEMPLATES.keys())}"
            )
        self.template_name = template_name
        self.template = PROMPT_TEMPLATES[template_name]

    @staticmethod
    def format_choices(endings: list[str]) -> str:
        """Format the four endings as A/B/C/D options."""
        labels = ["A", "B", "C", "D"]
        lines = []
        for label, ending in zip(labels, endings):
            lines.append(f"  {label}. {ending.strip()}")
        return "\n".join(lines)

    def build_prompt(
        self,
        example: dict,
        few_shot_block: str = "",
    ) -> str:
        """Build a full prompt string for a single HellaSwag example."""
        choices_str = self.format_choices(example["endings"])
        return self.template.format(
            ctx=example["ctx"],
            choices=choices_str,
            few_shot_block=few_shot_block,
        )

    @staticmethod
    def available_templates() -> list[str]:
        return list(PROMPT_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# 2. Few-Shot Selection (TF-IDF similarity)
# ---------------------------------------------------------------------------

class FewShotSelector:
    """Select the best few-shot examples from a training pool using
    TF-IDF cosine similarity with the query context.

    Uses only stdlib + basic math (no sklearn/scipy).
    """

    def __init__(self, train_examples: list[dict]):
        self.train = train_examples
        # Build vocabulary and TF-IDF vectors for all training contexts
        self._docs = [ex["ctx"].lower() for ex in self.train]
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._tfidf_vecs: list[dict[str, float]] = []
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokeniser."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_index(self):
        """Compute IDF scores and per-document TF-IDF vectors."""
        n = len(self._docs)
        doc_freq: Counter = Counter()
        tokenized_docs: list[list[str]] = []

        for doc in self._docs:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            unique = set(tokens)
            for tok in unique:
                doc_freq[tok] += 1

        # IDF = log(N / df)
        for tok, df in doc_freq.items():
            self._idf[tok] = math.log((n + 1) / (df + 1)) + 1.0

        # TF-IDF per document (sparse dict)
        for tokens in tokenized_docs:
            tf: Counter = Counter(tokens)
            total = len(tokens) if tokens else 1
            vec: dict[str, float] = {}
            for tok, count in tf.items():
                vec[tok] = (count / total) * self._idf.get(tok, 1.0)
            self._tfidf_vecs.append(vec)

    def _query_vec(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        tf: Counter = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec: dict[str, float] = {}
        for tok, count in tf.items():
            vec[tok] = (count / total) * self._idf.get(tok, 1.0)
        return vec

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in a)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-12
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-12
        return dot / (norm_a * norm_b)

    def select(self, query_ctx: str, k: int = 3) -> list[dict]:
        """Return the *k* training examples most similar to *query_ctx*."""
        q_vec = self._query_vec(query_ctx)
        scored = [
            (i, self._cosine(q_vec, self._tfidf_vecs[i]))
            for i in range(len(self.train))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [self.train[i] for i, _ in scored[:k]]

    def build_few_shot_block(
        self,
        query_ctx: str,
        k: int = 3,
        template_optimizer: PromptTemplateOptimizer | None = None,
    ) -> str:
        """Build a few-shot prefix from the top-k similar training examples."""
        examples = self.select(query_ctx, k=k)
        if not examples:
            return ""

        opt = template_optimizer or PromptTemplateOptimizer("direct")
        labels = ["A", "B", "C", "D"]
        blocks: list[str] = []

        for ex in examples:
            choices_str = opt.format_choices(ex["endings"])
            gold = labels[ex["label"]] if 0 <= ex["label"] <= 3 else "A"
            blocks.append(
                f"Context: {ex['ctx']}\n"
                f"{choices_str}\n"
                f"Answer: {gold}\n"
            )

        return "\n".join(blocks) + "\n"


# ---------------------------------------------------------------------------
# 3. Chain-of-Thought Prompting
# ---------------------------------------------------------------------------

COT_SUFFIXES: dict[str, str] = {
    "step_by_step": (
        " Let's think step by step about which continuation is most plausible.\n\n"
    ),
    "reasoning": (
        " First, I will reason about the context and each option, "
        "then select the best one.\n\n"
    ),
    "eliminate": (
        " Let me eliminate the unlikely options one by one.\n\n"
    ),
}

DEFAULT_COT = "step_by_step"


class ChainOfThoughtPrompter:
    """Wrap a prompt with chain-of-thought reasoning instructions."""

    def __init__(self, cot_style: str = DEFAULT_COT):
        if cot_style not in COT_SUFFIXES:
            raise ValueError(
                f"Unknown CoT style '{cot_style}'. "
                f"Available: {list(COT_SUFFIXES.keys())}"
            )
        self.cot_style = cot_style
        self.suffix = COT_SUFFIXES[cot_style]

    def wrap(self, prompt: str) -> str:
        """Insert the CoT instruction before the final answer line."""
        # Find the last line that looks like an answer prompt
        # (e.g., "Answer:", "The correct answer is:", etc.)
        lines = prompt.rstrip().split("\n")

        # Find the last line that is an answer cue
        answer_idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip().lower()
            if any(
                kw in stripped
                for kw in ["answer", "correct", "best continuation", "option:"]
            ):
                answer_idx = i
                break

        # Insert CoT before the answer cue
        before = "\n".join(lines[:answer_idx])
        after = "\n".join(lines[answer_idx:])
        return before + "\n" + self.suffix + after

    @staticmethod
    def extract_answer(response: str) -> str:
        """Extract the final answer letter from a CoT response.

        The model may reason at length before concluding.  We look for the
        last occurrence of a single letter A-D, or patterns like
        "the answer is B".
        """
        text = response.strip()

        # Pattern: "the answer is X" or "Answer: X"
        m = re.search(
            r"(?:the\s+(?:correct\s+)?answer\s+is|answer\s*:)\s*([A-Da-d])",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

        # Pattern: "option X" or "(X)"
        m = re.search(r"(?:option\s+|[\(\[])\s*([A-Da-d])\s*[\)\]]?", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # Fallback: last standalone letter A-D in the response
        letters = re.findall(r"\b([A-Da-d])\b", text)
        if letters:
            return letters[-1].upper()

        # Last resort: first character
        for ch in text:
            if ch.upper() in "ABCD":
                return ch.upper()

        return "A"  # default


# ---------------------------------------------------------------------------
# 4. Self-Consistency Decoding (Majority Voting)
# ---------------------------------------------------------------------------

class SelfConsistencyDecoder:
    """Generate k samples at temperature > 0 and return the majority vote."""

    def __init__(self, k: int = 5, temperature: float = 0.7):
        self.k = k
        self.temperature = temperature

    def aggregate(self, responses: list[str]) -> str:
        """Given k response strings, extract answer letters and majority-vote.

        Returns the most common answer letter (A-D).
        """
        cot = ChainOfThoughtPrompter()
        votes: list[str] = []
        for resp in responses:
            letter = cot.extract_answer(resp)
            votes.append(letter)

        if not votes:
            return "A"

        counter = Counter(votes)
        # Tie-breaking: prefer the first answer that appeared
        best_count = counter.most_common(1)[0][1]
        for v in votes:
            if counter[v] == best_count:
                return v

        return counter.most_common(1)[0][0]

    def get_generation_configs(self) -> list[dict]:
        """Return k generation config dicts (varied seeds for diversity)."""
        configs = []
        for i in range(self.k):
            configs.append({
                "temperature": self.temperature,
                "seed": 42 + i * 7,
                "top_p": 0.95,
                "top_k": 50,
            })
        return configs


# ---------------------------------------------------------------------------
# 5. Prompt Ensembling
# ---------------------------------------------------------------------------

class PromptEnsembler:
    """Try multiple prompt templates and aggregate their answers."""

    def __init__(self, template_names: list[str] | None = None):
        if template_names is None:
            template_names = list(PROMPT_TEMPLATES.keys())
        self.template_names = template_names
        self.optimizers = [
            PromptTemplateOptimizer(name) for name in template_names
        ]

    def build_prompts(
        self,
        example: dict,
        few_shot_block: str = "",
    ) -> list[tuple[str, str]]:
        """Build one prompt per template.

        Returns list of (template_name, prompt_string).
        """
        prompts = []
        for name, opt in zip(self.template_names, self.optimizers):
            prompts.append((name, opt.build_prompt(example, few_shot_block)))
        return prompts

    def aggregate(self, responses: list[str]) -> str:
        """Majority-vote across ensemble responses."""
        cot = ChainOfThoughtPrompter()
        votes = [cot.extract_answer(r) for r in responses]
        if not votes:
            return "A"
        counter = Counter(votes)
        return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Convenience: build an optimized prompt combining multiple strategies
# ---------------------------------------------------------------------------

def build_optimized_prompt(
    example: dict,
    template_name: str = "direct",
    few_shot_block: str = "",
    use_cot: bool = True,
    cot_style: str = "step_by_step",
) -> str:
    """Compose template + few-shot + CoT into a single prompt."""
    opt = PromptTemplateOptimizer(template_name)
    prompt = opt.build_prompt(example, few_shot_block=few_shot_block)

    if use_cot:
        cot = ChainOfThoughtPrompter(cot_style)
        prompt = cot.wrap(prompt)

    return prompt


# ---------------------------------------------------------------------------
# Answer normalization utilities
# ---------------------------------------------------------------------------

def normalize_answer(raw: str) -> str:
    """Map a raw model response to one of A/B/C/D.

    Handles various output formats:
      - Single letter: "B"
      - With punctuation: "B."
      - Full text: "The answer is B"
      - Ordinal: "2" -> "B"
      - CoT reasoning followed by answer
    """
    text = raw.strip()

    # Try structured patterns first
    cot = ChainOfThoughtPrompter()
    letter = cot.extract_answer(text)
    if letter in "ABCD":
        return letter

    # Try numeric: 0->A, 1->B, 2->C, 3->D
    m = re.search(r"\b([0-3])\b", text)
    if m:
        return "ABCD"[int(m.group(1))]

    # Try ordinal words
    ordinal_map = {
        "first": "A", "second": "B", "third": "C", "fourth": "D",
        "1st": "A", "2nd": "B", "3rd": "C", "4th": "D",
    }
    text_lower = text.lower()
    for word, ltr in ordinal_map.items():
        if word in text_lower:
            return ltr

    return "A"  # safe default


# ---------------------------------------------------------------------------
# CLI: quick sanity check
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test optimization strategies.")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["template", "fewshot", "cot", "selfcon", "ensemble", "all"],
    )
    parser.add_argument("--sample", type=int, default=3)
    args = parser.parse_args()

    # Load sample data if available
    data_path = Path(__file__).resolve().parent / "data" / "hellaswag_val.json"
    if not data_path.exists():
        print(f"No data at {data_path}.  Run prepare_data.py first.")
        # Create a synthetic example for demonstration
        examples = [{
            "ctx": "A person is standing in a kitchen. They pick up a knife and",
            "endings": [
                "begin to chop vegetables on the cutting board.",
                "throw it at the wall angrily.",
                "use it to write a letter.",
                "place it inside the refrigerator carefully.",
            ],
            "label": 0,
            "activity": "Cooking",
            "ind": 0,
        }]
    else:
        with open(data_path) as f:
            examples = json.load(f)

    examples = examples[: args.sample]

    if args.strategy in ("template", "all"):
        print("=" * 60)
        print("STRATEGY 1: Prompt Template Optimization")
        print("=" * 60)
        for name in PromptTemplateOptimizer.available_templates():
            opt = PromptTemplateOptimizer(name)
            prompt = opt.build_prompt(examples[0])
            print(f"\n--- Template: {name} ---")
            print(prompt[:300] + "..." if len(prompt) > 300 else prompt)

    if args.strategy in ("cot", "all"):
        print("\n" + "=" * 60)
        print("STRATEGY 3: Chain-of-Thought")
        print("=" * 60)
        opt = PromptTemplateOptimizer("direct")
        base_prompt = opt.build_prompt(examples[0])
        for style in COT_SUFFIXES:
            cot = ChainOfThoughtPrompter(style)
            wrapped = cot.wrap(base_prompt)
            print(f"\n--- CoT style: {style} ---")
            print(wrapped[:400] + "..." if len(wrapped) > 400 else wrapped)

    if args.strategy in ("selfcon", "all"):
        print("\n" + "=" * 60)
        print("STRATEGY 4: Self-Consistency")
        print("=" * 60)
        sc = SelfConsistencyDecoder(k=5, temperature=0.7)
        fake_responses = [
            "Let me think... The answer is B",
            "B. The second option makes sense.",
            "I think A is correct.",
            "Option B seems right.",
            "The answer is B.",
        ]
        print(f"Responses: {fake_responses}")
        print(f"Majority vote: {sc.aggregate(fake_responses)}")

    if args.strategy in ("ensemble", "all"):
        print("\n" + "=" * 60)
        print("STRATEGY 5: Prompt Ensembling")
        print("=" * 60)
        ens = PromptEnsembler()
        prompts = ens.build_prompts(examples[0])
        print(f"Generated {len(prompts)} prompt variants:")
        for name, _ in prompts:
            print(f"  - {name}")

    print("\n[optimize_prompt] Strategy demonstration complete.")


if __name__ == "__main__":
    main()
