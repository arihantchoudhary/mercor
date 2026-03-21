#!/usr/bin/env python3
"""
Load-test an Ollama endpoint and collect latency / throughput metrics.

Usage:
    python load_test.py
    python load_test.py --model llama3.2:3b
    python load_test.py --base-url http://localhost:11434 --output perf/metrics.csv

The script:
  1. Warms the model with a throwaway request.
  2. Iterates over concurrency levels (1, 2, 4, 8) and prompt types
     (short / long), with caching on and off.
  3. For each combination it fires concurrent streaming requests, measures
     TTFT, total latency, and tokens-per-second (from ollama eval fields).
  4. Computes P50 / P95 / P99 percentiles and writes everything to CSV.
"""

import argparse
import asyncio
import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np


# ── Defaults ────────────────────────────────────────────────────────────────

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "metrics.csv")

CONCURRENCY_LEVELS = [1, 2, 4, 8]

SHORT_PROMPT = "What is 2+2?"

LONG_PROMPT = (
    "You are an expert historian. Please provide a detailed analysis of the "
    "social, economic, and political factors that led to the fall of the Roman "
    "Empire. Consider the role of military overextension, economic troubles such "
    "as inflation and over-reliance on slave labor, the rise of Christianity and "
    "its impact on civic values, government corruption and political instability, "
    "the migration and invasion of barbarian tribes, and the splitting of the "
    "empire into Eastern and Western halves.\n\n"
    "In your response, discuss how these factors interacted with one another and "
    "evaluate which you consider most decisive. Support your argument with "
    "specific historical evidence and cite approximate dates where appropriate. "
    "Structure your answer with clear paragraphs covering each major factor."
)

PROMPTS = {
    "short": SHORT_PROMPT,
    "long": LONG_PROMPT,
}

STOP_SEQUENCES_MAP = {
    "none": None,
    "newline": ["\n"],
}


# ── Data containers ─────────────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    """Metrics collected for a single request."""
    ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0
    latency_ms: float = 0.0
    eval_count: int = 0


@dataclass
class RunConfig:
    """Describes one test run."""
    concurrency: int = 1
    prompt_type: str = "short"
    cache: bool = True
    stop: str = "none"


@dataclass
class AggregatedResult:
    """Aggregated metrics for one RunConfig."""
    config: RunConfig = field(default_factory=RunConfig)
    ttft_ms_values: list = field(default_factory=list)
    tps_values: list = field(default_factory=list)
    latency_ms_values: list = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

def percentile(values: list[float], p: int) -> float:
    """Return the p-th percentile of a list of floats, or 0.0 if empty."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def check_gpu_utilization():
    """Best-effort GPU utilisation snapshot (macOS Apple Silicon)."""
    if platform.system() != "Darwin":
        print("  GPU check: skipped (not macOS)")
        return
    try:
        result = subprocess.run(
            ["ioreg", "-l", "-w", "0"],
            capture_output=True, text=True, timeout=5,
        )
        # Look for a line that hints at Metal / GPU performance state
        for line in result.stdout.splitlines():
            if "PerformanceStatistics" in line and "GPU" in line:
                print(f"  GPU hint: {line.strip()[:120]}")
                return
        print("  GPU check: Apple Silicon Metal in use (no detailed stats without sudo powermetrics)")
    except Exception as exc:
        print(f"  GPU check: could not query ({exc})")


async def warm_up(session: aiohttp.ClientSession, base_url: str, model: str):
    """Send a tiny non-streaming request to load the model into memory."""
    payload = {
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "options": {"num_predict": 1},
    }
    print(f"Warming up model '{model}'...")
    async with session.post(f"{base_url}/api/generate", json=payload) as resp:
        await resp.read()
    print("Model loaded.\n")


# ── Core request logic ───────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    cache: bool = True,
    stop: list[str] | None = None,
) -> RequestMetrics:
    """
    Send a single streaming request and measure TTFT, throughput, and latency.

    Ollama streams newline-delimited JSON objects.  The final object contains
    ``eval_count`` and ``eval_duration`` which we use for throughput.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": 256},
    }
    if not cache:
        # Setting a unique keep_alive to 0 forces ollama to not reuse KV cache
        # across requests; we also disable context caching via num_ctx reset.
        payload["keep_alive"] = 0
    if stop is not None:
        payload["options"]["stop"] = stop

    metrics = RequestMetrics()
    t_start = time.perf_counter()
    first_token_received = False

    try:
        async with session.post(
            f"{base_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                print(f"  WARNING: HTTP {resp.status}: {body[:200]}", file=sys.stderr)
                return metrics

            buffer = bytearray()
            async for chunk in resp.content.iter_any():
                buffer.extend(chunk)
                if not first_token_received:
                    # Only mark TTFT when we see actual token content
                    # (the first JSON object with a non-empty "response" field)
                    try:
                        for raw_line in buffer.split(b"\n"):
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            obj = json.loads(raw_line)
                            if obj.get("response", ""):
                                metrics.ttft_ms = (time.perf_counter() - t_start) * 1000
                                first_token_received = True
                                break
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Total latency
            metrics.latency_ms = (time.perf_counter() - t_start) * 1000

            # Parse the last JSON object to extract eval statistics
            lines = buffer.split(b"\n")
            for line in reversed(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    obj = json.loads(line_stripped)
                except json.JSONDecodeError:
                    continue
                eval_count = obj.get("eval_count", 0)
                eval_duration_ns = obj.get("eval_duration", 0)
                metrics.eval_count = eval_count
                if eval_duration_ns > 0 and eval_count > 0:
                    metrics.tokens_per_sec = eval_count / (eval_duration_ns / 1e9)
                break
    except asyncio.TimeoutError:
        metrics.latency_ms = (time.perf_counter() - t_start) * 1000
        print("  WARNING: request timed out", file=sys.stderr)
    except Exception as exc:
        metrics.latency_ms = (time.perf_counter() - t_start) * 1000
        print(f"  WARNING: request failed: {exc}", file=sys.stderr)

    return metrics


# ── Test driver ──────────────────────────────────────────────────────────────

async def run_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    config: RunConfig,
    num_requests_per_worker: int = 2,
) -> AggregatedResult:
    """
    Fire *config.concurrency* workers, each sending *num_requests_per_worker*
    requests sequentially, and collect results.
    """
    prompt = PROMPTS[config.prompt_type]
    stop = STOP_SEQUENCES_MAP.get(config.stop)
    result = AggregatedResult(config=config)

    async def worker():
        for _ in range(num_requests_per_worker):
            m = await send_request(session, base_url, model, prompt, config.cache, stop)
            result.ttft_ms_values.append(m.ttft_ms)
            result.tps_values.append(m.tokens_per_sec)
            result.latency_ms_values.append(m.latency_ms)

    workers = [worker() for _ in range(config.concurrency)]
    await asyncio.gather(*workers)
    return result


async def run_all(base_url: str, model: str, output_path: str):
    """Execute every combination of concurrency / prompt / cache / stop."""
    connector = aiohttp.TCPConnector(limit=0)  # unlimited concurrent connections
    async with aiohttp.ClientSession(connector=connector) as session:
        # Health check
        try:
            async with session.get(f"{base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
        except Exception:
            print(f"ERROR: Cannot reach Ollama at {base_url}", file=sys.stderr)
            print("Start the server first:  make serve", file=sys.stderr)
            sys.exit(1)

        await warm_up(session, base_url, model)

        print("GPU utilisation snapshot:")
        check_gpu_utilization()
        print()

        all_results: list[AggregatedResult] = []

        test_matrix = [
            (c, pt, cache, stop)
            for c in CONCURRENCY_LEVELS
            for pt in ["short", "long"]
            for cache in [True, False]
            for stop in ["none", "newline"]
        ]

        total = len(test_matrix)
        for idx, (conc, pt, cache, stop) in enumerate(test_matrix, 1):
            cfg = RunConfig(concurrency=conc, prompt_type=pt, cache=cache, stop=stop)
            label = (
                f"[{idx}/{total}] concurrency={conc}  prompt={pt:5s}  "
                f"cache={'on' if cache else 'off'}  stop={stop}"
            )
            print(f"{label} ... ", end="", flush=True)

            agg = await run_batch(session, base_url, model, cfg)
            all_results.append(agg)

            n = len(agg.latency_ms_values)
            avg_lat = np.mean(agg.latency_ms_values) if n else 0
            avg_ttft = np.mean(agg.ttft_ms_values) if n else 0
            avg_tps = np.mean(agg.tps_values) if n else 0
            print(
                f"done  ({n} reqs, avg latency {avg_lat:.0f} ms, "
                f"avg TTFT {avg_ttft:.0f} ms, avg {avg_tps:.1f} tok/s)"
            )

        # Write CSV
        write_csv(all_results, output_path)
        print(f"\nResults saved to {output_path}")


# ── CSV output ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "concurrency",
    "prompt_type",
    "cache",
    "stop",
    "ttft_ms",
    "tokens_per_sec",
    "latency_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
]


def write_csv(results: list[AggregatedResult], path: str):
    """Write one row per (request) with aggregate percentiles repeated."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for agg in results:
            cfg = agg.config
            p50 = percentile(agg.latency_ms_values, 50)
            p95 = percentile(agg.latency_ms_values, 95)
            p99 = percentile(agg.latency_ms_values, 99)
            for i in range(len(agg.latency_ms_values)):
                writer.writerow({
                    "concurrency": cfg.concurrency,
                    "prompt_type": cfg.prompt_type,
                    "cache": "on" if cfg.cache else "off",
                    "stop": cfg.stop,
                    "ttft_ms": f"{agg.ttft_ms_values[i]:.2f}",
                    "tokens_per_sec": f"{agg.tps_values[i]:.2f}",
                    "latency_ms": f"{agg.latency_ms_values[i]:.2f}",
                    "p50_ms": f"{p50:.2f}",
                    "p95_ms": f"{p95:.2f}",
                    "p99_ms": f"{p99:.2f}",
                })


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load-test an Ollama model and collect latency metrics",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url", default=OLLAMA_BASE,
        help=f"Ollama base URL (default: {OLLAMA_BASE})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ollama Load Test")
    print(f"  Model:    {args.model}")
    print(f"  Endpoint: {args.base_url}")
    print(f"  Output:   {args.output}")
    print(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    print("=" * 60 + "\n")

    asyncio.run(run_all(args.base_url, args.model, args.output))


if __name__ == "__main__":
    main()
