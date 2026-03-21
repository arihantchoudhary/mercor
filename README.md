# LLM Evaluation Pipeline

A complete local LLM evaluation pipeline built on **Ollama** and **lm-evaluation-harness**. The project covers five end-to-end stages: serving, evaluation, performance profiling, guardrails, and inference-time improvement -- all running locally on an Apple M3 Pro with 18 GB RAM using the **qwen2.5:7b** model.

## Prerequisites

| Requirement | Install |
|-------------|---------|
| Python 3.11+ | [python.org](https://www.python.org) or `brew install python` |
| Ollama | [ollama.com](https://ollama.com) or `brew install ollama` |
| Model weights | Pulled automatically on first `make serve` |

## Quick Start

```bash
# One-time setup: install Python deps and pull the model
make setup

# Start the inference server (one-line startup)
make serve

# In a second terminal -- run any of the pipeline stages below
```

## One-Line Startup

```bash
make serve
```

This starts the Ollama daemon (if not already running), pulls `qwen2.5:7b` if needed, warms the model into memory, and prints the API endpoints.

## Project Structure

```
mercor/
  serve/
    serve.py              Part A -- Local model serving via Ollama
    client.py             Sample prompt generations & multi-turn chat demo
  eval_runner/
    model.py              OllamaLM adapter for lm-evaluation-harness
    run_eval.py           Benchmark runner (MMLU, HellaSwag, custom tasks)
    custom_benchmark.json 18-item programming language identification dataset
    custom_task.yaml      lm-eval task definition for the custom benchmark
    results/              JSON result files from evaluation runs
  perf/
    load_test.py          Async load tester (concurrency x prompt length x caching)
    metrics.csv           Raw latency/throughput measurements (240 rows)
    analysis.ipynb        Jupyter notebook with plots and analysis
  guardrails/
    validate.py           Determinism & output-format validation checks
    README.md             Explanation of deterministic inference controls
    results/              validation.json with per-prompt details
  improve/
    prepare_data.py       Downloads HellaSwag from HuggingFace
    optimize_prompt.py    Prompt templates, few-shot, CoT, self-consistency
    infer.py              Inference runner with ablation + optimized modes
    baseline.py           Baseline HellaSwag inference
    improved.py           Improved inference with combined strategies
    compare.py            Comparison utilities
    eval.sh               End-to-end pipeline script (data -> baseline -> ablation -> optimized)
    report.md             Detailed analysis and findings
    results/              Per-strategy result JSONs with accuracy and CI
    data/                 HellaSwag train/val splits
  Makefile                All pipeline commands
  requirements.txt        Python dependencies
  README.md               This file
```

## How to Run Each Part

### Part A -- Serving

```bash
make serve                  # start Ollama + load model
make client                 # run sample generations (in another terminal)
```

`serve.py` ensures the Ollama daemon is running, pulls the model if missing, and sends a warm-up request. `client.py` exercises the `/api/generate` and `/api/chat` endpoints with five diverse prompts, a multi-turn conversation, and a determinism check.

### Part B -- Evaluation

```bash
make eval                   # run MMLU + HellaSwag via lm-evaluation-harness
make eval-custom            # run custom 18-item programming language ID benchmark
```

The custom benchmark presents code snippets from 18 languages (Rust, Python, Java, Go, JavaScript, C, C++, TypeScript, Ruby, Swift, Kotlin, PHP, SQL, Elixir, Scala, Haskell, Bash, C#) and scores exact-match accuracy.

### Part C -- Performance

```bash
make perf                   # run load test across concurrency levels
```

Measures time-to-first-token (TTFT), tokens/sec, end-to-end latency, and percentiles (p50/p95/p99) across a matrix of:
- Concurrency: 1, 2, 4, 8
- Prompt length: short, long
- KV-cache: on, off
- Stop condition: none, newline

### Part D -- Guardrails

```bash
make validate               # determinism + output validation
```

Sends five prompts three times each with `temperature=0, seed=42` and asserts byte-identical outputs. Then runs a five-item code-identification task checking regex format compliance, schema validation against a language allow-list, and correctness.

### Part E -- Improvement

```bash
make improve                # full HellaSwag improvement pipeline
cd improve && bash eval.sh --quick   # fast 20-example subset
```

Runs baseline, ablation studies (prompt template, few-shot, chain-of-thought, self-consistency, ensemble), and a fully optimized configuration. Results are saved with timestamps and `_latest.json` symlinks.

## Key Results

### Part B: Benchmarks

| Benchmark | Accuracy | N | 95% CI (Wilson) |
|-----------|:--------:|:-:|:---------------:|
| **MMLU** (6 subjects) | 65.3% | 300 | [59.8%, 70.5%] |
| **HellaSwag** (lm-eval, logprob scoring) | 38.0% | 50 | [25.8%, 51.8%] |
| **HellaSwag** (forced single-token) | 71.5% | 200 | [64.9%, 77.3%] |
| **Custom** (prog_lang_id) | 100.0% | 18 | -- |

### Part C: Performance

| Concurrency | Prompt | Avg Tokens/sec | Avg Latency |
|:-----------:|:------:|:--------------:|:-----------:|
| 1 | short | 31.7 tok/s | 3.2 s |
| 1 | long | 26.2 tok/s | 13.0 s |
| 4 | short | 26.2 tok/s | 13.7 s |
| 8 | long | 19.3 tok/s | 114.4 s |

Throughput stays at ~20-32 tok/s across concurrency levels; Ollama serializes inference on Apple Silicon, so latency scales linearly with queue depth.

### Part D: Guardrails

| Check | Result |
|-------|--------|
| Determinism (5 prompts x 3 trials) | **5/5 passed** |
| Regex + Schema validation | **5/5 passed** |

### Part E: HellaSwag Improvement

| Approach | Accuracy | CI (Wilson) | Avg Latency |
|----------|:--------:|:-----------:|:-----------:|
| Free-text baseline (old) | 66.7% | [48.8%, 80.8%] | 4.59s |
| Single-token baseline | 71.5% | [64.9%, 77.3%] | 2.87s |
| **Single-token + few-shot + SC** | **76.0%** | **[69.6%, 81.4%]** | **5.16s** |

**Total improvement: +9.3 pp** (66.7% → 76.0%) through two stacked changes: forced single-token scoring (+4.8 pp) and few-shot + self-consistency (+4.5 pp). Both independently exceed the +3.0 pp target. See `improve/report.md` for full ablation study and 11 before/after examples.

## Hardware

| | |
|---|---|
| Chip | Apple M3 Pro |
| Memory | 18 GB unified |
| OS | macOS 15.7 (arm64) |
| Ollama | Local inference, no GPU offload |
| Model | qwen2.5:7b (4-bit quantized, ~4.7 GB) |

## Using a Different Model

```bash
make serve MODEL=llama3.2:3b
make client MODEL=llama3.2:3b
make eval MODEL=llama3.2:3b
```

## All Commands

```bash
make help      # show all available targets
make setup     # install deps + pull model
make serve     # start inference server
make client    # run sample generations
make eval      # MMLU + HellaSwag benchmarks
make eval-custom  # custom programming language ID benchmark
make perf      # load test
make validate  # determinism + guardrails
make improve   # HellaSwag improvement pipeline
make zip       # create submission archive
```

## Final Summary

The most important lesson from this project: **for multiple-choice benchmarks, how you score matters more than how you prompt.**

My initial approach generated free-text answers and extracted letters via regex. Every "optimization" — chain-of-thought, self-consistency, few-shot — made accuracy *worse* because verbose model output corrupted answer extraction. CoT dropped accuracy from 67% to 20%.

The breakthrough was switching to forced single-token output (`num_predict=1`). By ending the prompt with "The answer is" and limiting generation to one token, extraction becomes trivial and the model's internal probability distribution does the ranking. This single change yielded **+4.8 pp** (66.7% → 71.5%) while also being *faster* (2.87s vs 4.59s per example).

This mirrors how HellaSwag is evaluated in the research literature: likelihood-based scoring, not generation-based. The standard exists for a reason — and rediscovering it through failure was the most valuable part of this project.
