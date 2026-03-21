MODEL ?= qwen2.5:7b
OLLAMA_URL ?= http://localhost:11434

# ── Part A: Serving ──────────────────────────────────────────
.PHONY: serve client

serve:  ## Start ollama and serve the model
	python serve/serve.py --model $(MODEL)

client:  ## Run sample prompt generations
	python serve/client.py --model $(MODEL)

# ── Part B: Evaluation ───────────────────────────────────────
.PHONY: eval eval-custom

eval:  ## Run official benchmarks (MMLU + HellaSwag)
	python eval_runner/run_eval.py --model $(MODEL) --tasks mmlu hellaswag

eval-custom:  ## Run custom benchmark
	python eval_runner/run_eval.py --model $(MODEL) --tasks custom

# ── Part C: Performance ──────────────────────────────────────
.PHONY: perf

perf:  ## Run load test and collect metrics
	python perf/load_test.py --model $(MODEL)

# ── Part D: Guardrails ───────────────────────────────────────
.PHONY: validate

validate:  ## Run determinism & guardrail checks
	python guardrails/validate.py --model $(MODEL)

# ── Part E: Improvement ──────────────────────────────────────
.PHONY: improve

improve:  ## Run benchmark improvement pipeline
	cd improve && bash eval.sh

# ── Utilities ────────────────────────────────────────────────
.PHONY: install setup zip help

install:  ## Install Python dependencies
	pip install -r requirements.txt

setup: install  ## Full setup: install deps + pull model
	ollama pull $(MODEL)

zip:  ## Create submission archive
	zip -r mercor-submission.zip . \
		-x ".venv/*" "env/*" "*/__pycache__/*" ".git/*" "node_modules/*" \
		   ".venv/**" "env/**" ".git/**" "node_modules/**" \
		   "*.pyc" ".DS_Store"

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
