# Part D — Determinism and Output Validation

Guardrail checks that verify an LLM produces reproducible outputs and that
those outputs conform to an expected format.

## What is tested

| Test | Description |
|------|-------------|
| **Determinism** | Five diverse prompts are each sent `--trials` times (default 5) with identical sampling parameters. All trials for a given prompt must produce byte-identical output. |
| **Output validation** | A programming-language-identification benchmark (five code snippets) is run. Each response is checked with *regex validation* (output is a single language name) and *schema validation* (name appears in an allow-list of known languages). |

## How deterministic mode works

Ollama exposes sampling options per request. To maximise reproducibility the
script pins three parameters:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `temperature` | `0` | Greedy decoding — always pick the highest-probability token. |
| `top_p` | `1` | No nucleus-sampling truncation (combined with temperature=0 this is a no-op, but set explicitly for clarity). |
| `seed` | `42` | Fixes the PRNG seed inside the inference engine so any remaining stochastic tie-breaking is resolved identically. |

With these settings, the same prompt sent to the same model on the same
hardware should yield the same token sequence every time.

## Where nondeterminism can persist

Even with the above controls, outputs may still differ in some situations:

* **Batching** — When multiple requests are batched together the order of
  floating-point accumulations can change, producing different logits.
* **Model loading** — A freshly loaded model may initialise KV-cache or
  scratch buffers differently from a warm model, causing subtle numerical
  divergence.
* **Floating-point non-associativity** — GPU thread scheduling can reorder
  parallel reductions, and IEEE 754 floating-point addition is not
  associative, so partial sums may differ across runs.
* **KV-cache state** — Residual state from a previous request in the
  key/value cache can subtly alter attention computations for a subsequent
  request.
* **Quantisation kernels** — Different quantisation back-ends or GPU
  drivers may use non-deterministic kernel implementations.
* **Hardware differences** — Running the same model on a different GPU (or
  CPU vs GPU) changes numerical results due to differing instruction sets.

## Running

```bash
# Default: 5 trials, qwen2.5:7b, localhost:11434
make validate

# Custom
python guardrails/validate.py --model qwen2.5:7b --trials 10 --base-url http://localhost:11434
```

## Results format

Results are saved to `guardrails/results/validation.json` with the following
structure:

```json
{
  "model": "qwen2.5:7b",
  "trials": 5,
  "timestamp": "2026-03-20T14:30:00",
  "determinism": {
    "total_prompts": 5,
    "deterministic_count": 5,
    "details": [ ... ]
  },
  "validation": {
    "total_items": 5,
    "regex_passed": 5,
    "schema_passed": 5,
    "correct": 5,
    "details": [ ... ]
  }
}
```

The script exits with code 0 if all checks pass, or 1 if any check fails.
