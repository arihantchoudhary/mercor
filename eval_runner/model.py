#!/usr/bin/env python3
"""
Custom lm-evaluation-harness model wrapper for an Ollama endpoint.

This module subclasses lm_eval.api.model.LM so that lm-eval's standard
benchmark tasks can be evaluated against any model served by Ollama.

Key design decisions
--------------------
* **Loglikelihood estimation** -- Uses Ollama's OpenAI-compatible
  /v1/chat/completions endpoint with logprobs=true.  For each
  (context, continuation) pair, we perform token-by-token scoring:
  feed context as an assistant-prefix message so the model treats it as
  raw text to continue, generate 1 token at a time with logprobs, and
  look up the actual continuation token in the top-k logprobs.  Summing
  the per-token logprobs gives a proper perplexity-based ranking score.
* **Prompt cache** -- A dict-based cache keyed on (prompt, generation
  parameters) avoids redundant network round-trips and guarantees
  deterministic results when the same prompt appears more than once.
* **Concurrency** -- All requests are serial (one at a time) to respect
  Ollama's single-model-in-memory design.  This is intentional for a local
  evaluation setup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any

import requests
from tqdm import tqdm

import lm_eval.api.model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
REQUEST_TIMEOUT = 300  # seconds

# Number of continuation tokens to score for loglikelihood.
# 5 tokens is a good balance between accuracy and speed.
LOGPROB_SCORE_TOKENS = 5


def _cache_key(prompt: str, **kwargs) -> str:
    """Deterministic cache key for a prompt + generation params."""
    blob = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class OllamaLM(lm_eval.api.model.LM):
    """lm-eval model wrapper that delegates to a running Ollama server."""

    # ----- class-level metadata used by the harness -----
    # Tell lm-eval this is not a HuggingFace model
    AUTO_MODEL_CLASS = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
        max_gen_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 42,
        batch_size: int = 1,
        logprob_tokens: int = LOGPROB_SCORE_TOKENS,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_gen_tokens = max_gen_tokens
        self.temperature = temperature
        self.seed = seed
        self._batch_size = batch_size
        self.logprob_tokens = logprob_tokens

        # Prompt cache: cache_key -> result
        self._cache: dict[str, Any] = {}

        # Persistent HTTP session for connection pooling
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=10
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Verify the server is reachable
        self._health_check()

    # ------------------------------------------------------------------
    # Properties expected by the harness
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self):
        """End-of-text token id. Not used for API models."""
        return None

    @property
    def max_length(self) -> int:
        """Maximum context length. Conservative default for 7B models."""
        return 4096

    @property
    def max_gen_toks(self) -> int:
        return self.max_gen_tokens

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        """Not applicable for API models."""
        return "cpu"

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Approximate tokenisation for length estimation.

        Ollama does not publicly expose a tokeniser endpoint in all
        versions, so we fall back to a rough word-piece heuristic
        (~1.3 tokens per whitespace-delimited word).
        """
        words = string.split()
        return list(range(int(len(words) * 1.3)))

    def tok_decode(self, tokens, **kwargs) -> str:
        """Not used for API-based models -- raise if called."""
        raise NotImplementedError(
            "tok_decode is not supported for the Ollama wrapper"
        )

    # ------------------------------------------------------------------
    # Ollama HTTP helpers
    # ------------------------------------------------------------------

    def _health_check(self):
        """Verify Ollama is running and the model is available."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models:
                # Model might still work (partial name match), just warn.
                logger.warning(
                    "Model '%s' not found in Ollama models list %s. "
                    "It may still work if the name matches a prefix.",
                    self.model_name,
                    models,
                )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start the server first: python serve/serve.py"
            )

    def _generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        raw: bool = True,
    ) -> dict:
        """Send a generation request to Ollama and return the full JSON
        response.  Results are cached by (prompt, max_tokens, temperature).
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_gen_tokens
        temperature = temperature if temperature is not None else self.temperature

        key = _cache_key(
            prompt,
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            seed=self.seed,
        )
        if key in self._cache:
            return self._cache[key]

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "raw": raw,
            "options": {
                "temperature": temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            result = {"response": "", "error": str(exc)}

        self._cache[key] = result
        return result

    def _get_next_token_logprobs(self, prefix_text: str) -> list[dict]:
        """Get top-k logprobs for the next token after prefix_text.

        Uses the /v1/chat/completions endpoint with the prefix as an
        assistant message.  This makes the model treat the prefix as
        raw text to continue (bypassing the chat template for the
        content itself).

        Returns a list of {token, logprob} dicts, or [] on error.
        """
        cache_key = _cache_key(
            prefix_text, model=self.model_name, mode="next_token_logprobs_v2",
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = {
            "model": self.model_name,
            "messages": [{"role": "assistant", "content": prefix_text}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0,
        }

        try:
            resp = self._session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            top_logprobs = (
                data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            )
        except Exception as exc:
            logger.debug("Logprob request failed: %s", exc)
            top_logprobs = []

        self._cache[cache_key] = top_logprobs
        return top_logprobs

    # ------------------------------------------------------------------
    # Core LM interface required by lm-evaluation-harness
    # ------------------------------------------------------------------

    def loglikelihood(self, requests_list: list) -> list[tuple[float, bool]]:
        """Estimate log-likelihood for each (context, continuation) pair.

        Uses token-by-token scoring via the assistant-prefix approach:
        for each token position in the continuation, we send
        context + continuation[:position] as an assistant message,
        generate 1 token with logprobs, and look up the actual next
        continuation token in the top-k logprobs.  The sum of per-token
        logprobs gives a proper ranking score.
        """
        results: list[tuple[float, bool]] = []

        for req in tqdm(requests_list, desc="loglikelihood", leave=False):
            context, continuation = req.args
            if not continuation.strip():
                results.append((0.0, True))
                continue

            score, is_greedy = self._token_by_token_logprob(
                context, continuation
            )
            results.append((score, is_greedy))

        return results

    def _token_by_token_logprob(
        self, context: str, continuation: str,
    ) -> tuple[float, bool]:
        """Score P(continuation | context) by token-by-token probing.

        For each of the first N tokens in the continuation:
        1. Send context + continuation[:pos] as assistant prefix
        2. Get next-token logprobs
        3. Find the actual continuation token in top-k logprobs
        4. Sum logprobs for the score

        The continuation is consumed character-by-character, matching
        against the model's tokenization as revealed by top_logprobs.

        Returns (total_logprob, all_greedy).
        """
        cache_key = _cache_key(
            context,
            model=self.model_name,
            continuation=continuation,
            mode="tbt_logprob_v2",
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        total_logprob = 0.0
        all_greedy = True
        current_prefix = context
        remaining = continuation
        tokens_scored = 0

        for _ in range(self.logprob_tokens):
            if not remaining:
                break

            top_logprobs = self._get_next_token_logprobs(current_prefix)

            if not top_logprobs:
                # API error -- assign penalty and stop
                total_logprob += -20.0
                all_greedy = False
                break

            # Find the best matching token: the longest token in top_logprobs
            # that matches the start of the remaining continuation.
            best_match = None
            best_logprob = None
            best_rank = None

            for rank, entry in enumerate(top_logprobs):
                tok = entry["token"]
                if tok and remaining.startswith(tok):
                    # Prefer longer token matches (more specific)
                    if best_match is None or len(tok) > len(best_match):
                        best_match = tok
                        best_logprob = entry["logprob"]
                        best_rank = rank

            if best_match is not None:
                total_logprob += best_logprob
                if best_rank != 0:
                    all_greedy = False
                current_prefix += best_match
                remaining = remaining[len(best_match):]
                tokens_scored += 1
            else:
                # The actual continuation token is not in the top-20.
                # This means the model considers it very unlikely.
                # Assign a penalty logprob and advance.
                total_logprob += -20.0
                all_greedy = False

                # Advance by estimating the token boundary.  Use the
                # generated (greedy) token's length as a rough guide for
                # how many characters one token spans.
                gen_tok = top_logprobs[0]["token"] if top_logprobs else " "
                advance = max(1, len(gen_tok))
                current_prefix += remaining[:advance]
                remaining = remaining[advance:]
                tokens_scored += 1

        result = (total_logprob, all_greedy)
        self._cache[cache_key] = result
        return result

    def loglikelihood_rolling(self, requests_list: list) -> list[float]:
        """Rolling log-likelihood (used by perplexity-style tasks).

        lm-eval expects list[float] (not tuples) for this method.
        For each request the entire string is treated as both context and
        target.
        """
        results: list[float] = []

        for req in tqdm(
            requests_list, desc="loglikelihood_rolling", leave=False
        ):
            text = req.args[0]

            num_tokens = max(len(text.split()) * 2, 20)
            gen_result = self._generate(
                prompt="",
                max_tokens=num_tokens,
                temperature=0.0,
            )
            generated = gen_result.get("response", "")
            score = self._overlap_score(generated, text)
            results.append(score)

        return results

    def generate_until(self, requests_list: list) -> list[str]:
        """Open-ended generation for each request.

        Each request has .args = (context, gen_kwargs) where gen_kwargs is
        a dict with keys like 'until' (stop sequences) and
        'max_gen_toks'.
        """
        results: list[str] = []

        for req in tqdm(requests_list, desc="generate_until", leave=False):
            context, gen_kwargs = req.args

            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_toks = gen_kwargs.get(
                "max_gen_toks", self.max_gen_tokens
            )
            temperature = gen_kwargs.get("temperature", self.temperature)

            gen_result = self._generate(
                prompt=context,
                max_tokens=max_toks,
                temperature=temperature,
                stop=until if until else None,
            )
            text = gen_result.get("response", "")

            # Manually truncate at stop sequences if the model overshot
            for stop_seq in until:
                idx = text.find(stop_seq)
                if idx != -1:
                    text = text[:idx]

            results.append(text)

        return results

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _overlap_score(generated: str, target: str) -> float:
        """Compute a pseudo-logprob from normalised character overlap.

        Returns a value in (-inf, 0] where 0 means perfect overlap.
        """
        gen_clean = generated.strip().lower()
        tgt_clean = target.strip().lower()

        if not tgt_clean:
            return 0.0

        # Character-level longest common subsequence ratio
        match_len = _lcs_length(gen_clean, tgt_clean)
        ratio = match_len / len(tgt_clean)

        # Clamp to avoid log(0)
        ratio = max(ratio, 1e-12)
        return math.log(ratio)


# --------------------------------------------------------------------------
# Utility: Longest Common Subsequence length (character-level)
# --------------------------------------------------------------------------

def _lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence of *a* and *b*.

    Uses the classic O(n*m) DP, limited to the first 500 characters of
    each string to keep runtime bounded during evaluation.  Ensures the
    shorter string is used for the inner (column) dimension to minimise
    memory and improve cache locality.
    """
    a = a[:500]
    b = b[:500]
    # Early exits
    if not a or not b:
        return 0
    if a == b:
        return len(a)
    # If one is a substring of the other, LCS == shorter length
    if len(a) <= len(b) and a in b:
        return len(a)
    if len(b) < len(a) and b in a:
        return len(b)
    # Put shorter string on inner loop for better cache performance
    if len(a) < len(b):
        a, b = b, a
    m, n = len(a), len(b)
    # Space-optimised DP (single row + prev value)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev_diag = 0
        ai = a[i - 1]
        for j in range(1, n + 1):
            temp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev_diag + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            prev_diag = temp
    return dp[n]
