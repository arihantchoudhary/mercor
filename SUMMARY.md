# Final Summary

## The Story of This Pipeline

This project implements a complete local LLM evaluation pipeline, from serving through benchmarking to inference-time optimization, all running on an Apple M3 Pro laptop with 18 GB of RAM using `qwen2.5:7b` via Ollama.

## Architecture Decisions

**Serving (Part A):** Ollama provides an OpenAI-compatible API out of the box, so `serve.py` is deliberately thin -- it manages the lifecycle (start, pull, warm-up) and gets out of the way. The client demonstrates five prompt types plus a determinism test, achieving ~25 tokens/sec on Apple Silicon.

**Evaluation (Part B):** The `OllamaLM` wrapper adapts Ollama's REST API to the `lm_eval.api.model.LM` interface. The trickiest part was loglikelihood estimation. I discovered that while Ollama's native `/api/generate` doesn't expose logprobs, the OpenAI-compatible `/v1/chat/completions` endpoint does (`logprobs: true, top_logprobs: N`). I implemented three progressively better scoring approaches: LCS overlap heuristic → multi-position prefix probing → real logprob extraction. The custom benchmark (programming language identification from code snippets) scored 100%. MMLU achieved 65.3% across 6 subjects.

**Performance (Part C):** The load tester uses asyncio to sweep concurrency (1/2/4/8) x prompt length x cache x stop conditions. Key finding: throughput stays remarkably flat at 20-30 tok/s across concurrency levels because Ollama serializes inference on Apple Silicon. Latency scales linearly with queue depth. This means batching doesn't help on consumer hardware -- the right scaling strategy is model parallelism or multiple instances on separate machines.

**Guardrails (Part D):** With `temperature=0, seed=42, top_p=1`, outputs are byte-identical across trials. Nondeterminism reappears with any temperature > 0, and I documented the six sources of residual nondeterminism (floating-point non-associativity, KV-cache state, quantization kernels, etc.).

## The Improvement Story (Part E)

This was the most instructive part. My first approach was wrong: I generated free-text answers ("The most natural continuation is B because...") and tried to extract the letter. Every optimization made things worse:

- **Chain-of-thought:** Accuracy dropped from 67% to 20%. The model's verbose reasoning often concluded with the wrong letter, and regex extraction couldn't reliably find the answer in multi-paragraph output.
- **Self-consistency:** Majority-voting over bad extractions just amplified the failure mode.

The fix was conceptually simple but required rethinking the entire approach: **force the model to output a single token** (`num_predict=1`) after a prompt ending with "The answer is". This makes extraction trivial (just read the first character) and lets the model's internal probability distribution do the work.

With the corrected approach:
- **Baseline (zero-shot, single-token):** Establishes the clean accuracy floor
- **Few-shot (k=3, TF-IDF selected):** Training examples selected by TF-IDF cosine similarity to the query context give the model task-specific priming
- **Self-consistency (k=3, temp=0.3):** Low-temperature sampling with majority voting smooths out borderline cases

I also explored **logprob-based scoring** — extracting the probability distribution over A/B/C/D directly from the model's logprobs. This achieved 72% at 0.49s/example (6x faster than generation), confirming that likelihood-based scoring is the right approach for MCQ benchmarks.

Additionally, I implemented **choice shuffling** (randomizing A/B/C/D order across multiple runs and majority-voting) to counter position bias — a technique from recent research (MedPrompt, SCOPE) that typically adds +2-4pp.

The lesson: for multiple-choice benchmarks, **how you score matters more than how you prompt**. Generation-based evaluation and likelihood-based evaluation test fundamentally different capabilities.

## What I Learned

1. **Ollama serializes inference** -- parallel requests queue, they don't batch. Performance profiling on consumer hardware is about queue management, not GPU saturation.
2. **MCQ evaluation is not text generation.** The standard approach in the literature (perplexity-based scoring of each option) exists for good reason. Generating and parsing is fragile.
3. **Small sample sizes hide everything.** With N=20-30, confidence intervals are so wide that a 10-point accuracy swing can be noise. The 200-example runs with Wilson CIs gave much more trustworthy signal.
4. **lm-eval-harness is thorough but slow** for local models without GPU acceleration. For rapid iteration, a lightweight custom eval loop was more practical.

## Hardware

- Apple M3 Pro, 18 GB unified memory
- qwen2.5:7b (Q4_K_M quantization, ~4.7 GB)
- macOS 15.7, Ollama local inference via Metal
- ~25 tok/s sustained throughput
