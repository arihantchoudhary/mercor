# LLM Evaluation Pipeline

> **Assignment:** Architect and refine a scaled-down version of an internal LLM evaluation pipeline using Ollama and lm-evaluation-harness.

All five parts running locally on Apple M3 Pro (18 GB) with **qwen2.5:7b**.

## Quick Start

```bash
make setup    # install deps + pull model
make serve    # one-line startup (starts Ollama, pulls model, warms up)
```

Then in a second terminal:

```bash
make client       # Part A: sample generations
make eval         # Part B: MMLU + HellaSwag benchmarks
make eval-custom  # Part B: custom benchmark
make perf         # Part C: load test + metrics
make validate     # Part D: determinism + guardrails
make improve      # Part E: HellaSwag improvement pipeline
```

---

## Part A: Serving

**Deliverables:** `serve/serve.py`, `serve/client.py`, one-line startup via `make serve`

`serve.py` manages the full Ollama lifecycle: starts the daemon if not running, pulls the model if missing, warms it into GPU memory, and prints API endpoints. `client.py` runs 5 diverse prompt generations (factual, reasoning, creative, code, instruction-following), a multi-turn chat conversation, and a determinism verification test.

```bash
make serve    # start server
make client   # run sample generations
```

---

## Part B: Evaluation

**Deliverables:** `eval_runner/model.py` (wrapper), `eval_runner/run_eval.py` (runner), `eval_runner/results/`

`model.py` subclasses `lm_eval.api.model.LM` to bridge Ollama's REST API with the lm-evaluation-harness. Implements `loglikelihood` (token-by-token logprob scoring via `/v1/chat/completions` with `logprobs=true`), `generate_until`, and `loglikelihood_rolling`. All requests are cached by SHA256(prompt + params) for determinism and efficiency.

**Custom benchmark:** 18-item programming language identification — given a code snippet (Rust, Python, Java, Go, JS, C, C++, TypeScript, Ruby, Swift, Kotlin, PHP, SQL, Elixir, Scala, Haskell, Bash, C#), identify the language. Defined via `custom_task.yaml` for lm-eval integration.

| Benchmark | Accuracy | N | 95% CI (Wilson) |
|-----------|:--------:|:-:|:---------------:|
| **MMLU** (6 subjects) | 65.3% | 300 | [59.8%, 70.5%] |
| **HellaSwag** (lm-eval) | 38.0% | 50 | [25.8%, 51.8%] |
| **Custom** (prog_lang_id) | 100.0% | 18 | -- |

---

## Part C: Performance & Scaling

**Deliverables:** `perf/load_test.py`, `perf/metrics.csv`, `perf/analysis.ipynb`

Async load tester (`aiohttp`) sweeps a 32-configuration matrix:
- **Concurrency:** 1, 2, 4, 8
- **Prompt length:** short vs long
- **KV-cache:** on vs off
- **Stop sequences:** none vs newline

Collects TTFT, tokens/sec, end-to-end latency, and P50/P95/P99 percentiles. 240 data rows in `metrics.csv`. Notebook has 5 rendered plots with commentary.

| Concurrency | Prompt | Tokens/sec | Avg Latency |
|:-----------:|:------:|:----------:|:-----------:|
| 1 | short | 31.7 | 3.2s |
| 1 | long | 26.2 | 13.0s |
| 4 | short | 26.2 | 13.7s |
| 8 | long | 19.3 | 114.4s |

**Key finding:** Ollama serializes inference on Apple Silicon. Throughput stays flat at ~20-32 tok/s regardless of concurrency — latency scales linearly with queue depth.

---

## Part D: Guardrails & Determinism

**Deliverables:** `guardrails/validate.py`, `guardrails/README.md`

**Determinism:** 5 prompts x 5 trials with `temperature=0, seed=42, top_p=1` — all outputs byte-identical. **5/5 passed.**

**Validation:** Programming language identification outputs checked with regex (single language name format), schema (against allow-list of 40+ languages), and correctness. **5/5 passed.**

`README.md` documents 6 sources of residual nondeterminism: floating-point non-associativity, KV-cache state, batching order, quantization kernels, model loading, and hardware differences.

---

## Part E: Benchmark Improvement

**Deliverables:** `improve/prepare_data.py`, `improve/optimize_prompt.py`, `improve/infer.py`, `improve/eval.sh`, `improve/report.md`

**Benchmark:** HellaSwag. **Target:** +3.0 pp. **Achieved: +3.4 pp (p = 0.041 < 0.05).**

### The approach

My first attempt — free-text generation with regex extraction — failed. CoT dropped accuracy from 67% to 20% because verbose reasoning corrupted answer extraction. Self-consistency amplified the problem.

**The fix:** Force single-token output (`num_predict=1`) after a prompt ending with `"The answer is"`. The model outputs one character — no extraction needed. This alone yielded +5.1 pp.

Then stacked two more optimizations:
- **TF-IDF few-shot selection (k=3):** Select the 3 most similar training examples by cosine similarity on TF-IDF vectors. Gives the model task-specific priming.
- **Self-consistency (k=3, temp=0.3):** Generate 3 answers with different seeds, majority vote. Works because single-token output makes extraction trivial.

| Approach | Accuracy | N | p-value |
|----------|:--------:|:-:|:-------:|
| Free-text baseline | 66.7% | 20 | -- |
| Single-token baseline | 71.8% | 500 | -- |
| **+ Few-shot + self-consistency** | **75.2%** | **500** | **0.041** |

McNemar's test: 51 discordant pairs favoring optimized vs 34 favoring baseline. One-sided exact binomial p = 0.041.

See `improve/report.md` for full analysis with 13 before/after examples, ablation study, cost/latency trade-offs, and exact configuration.

See `improve/APPROACH.md` for a detailed writeup of why forced single-token scoring works and the HyDE alternative we considered.

---

## Project Structure

```
serve/                  Part A — serving + client demo
eval_runner/            Part B — lm-eval wrapper + benchmarks
  model.py              OllamaLM adapter (logprob scoring, caching)
  run_eval.py           Benchmark runner
  custom_benchmark.json 18-language code identification dataset
  results/              MMLU, HellaSwag, custom benchmark outputs
perf/                   Part C — load testing + analysis
  load_test.py          Async load generator
  metrics.csv           240 rows of latency/throughput data
  analysis.ipynb        5 rendered plots with commentary
guardrails/             Part D — determinism + validation
  validate.py           Determinism + output format checks
  README.md             Nondeterminism sources documentation
improve/                Part E — HellaSwag improvement
  prepare_data.py       Downloads HellaSwag from HuggingFace
  optimize_prompt.py    5 optimization strategies (templates, TF-IDF, CoT, SC, ensemble)
  infer.py              Forced single-token MCQ inference
  eval.sh               End-to-end pipeline
  report.md             668-word analysis (CIs, ablation, 13 examples)
  APPROACH.md           Detailed writeup of the forced single-token approach
Makefile                11 targets (serve, eval, perf, validate, improve, etc.)
SUMMARY.md              Final summary — the story of the improvement
```

## Final Summary

The most important lesson: **for multiple-choice benchmarks, how you score matters more than how you prompt.**

Free-text generation + regex extraction is fundamentally fragile for MCQ. Every prompt optimization (CoT, few-shot, self-consistency) made accuracy *worse* because it made extraction harder. The breakthrough was forced single-token output — constraining the model to emit a single answer token, making extraction trivial and directly accessing the model's probability distribution.

This mirrors how HellaSwag is evaluated in the literature: likelihood-based scoring, not generation-based. Rediscovering this through failure was the most valuable part of this project.

## Hardware

- Apple M3 Pro, 18 GB unified memory
- qwen2.5:7b (Q4_K_M quantization, ~4.7 GB)
- macOS 15.7, Ollama local inference via Metal
- ~25 tok/s sustained throughput
