"""
Concrete OpenAI/ChatGPT adapter.

Only this file imports the openai SDK.

Handles:
- Authentication via OPENAI_API_KEY environment variable.
- Rate limiting: enforces minimum interval between calls based on
  rate_limit_rpm from api_config.
- Retry with exponential backoff on transient errors (RateLimitError,
  InternalServerError, APIConnectionError).
- Configurable timeout per request.
- Cost estimation from average token counts and per-1k-token pricing.
- Graceful degradation: on permanent failure (all retries exhausted),
  returns an LLMResponse with finish_reason="error" rather than raising.

Required api_config keys:
    timeout_seconds: int — per-request timeout.
    max_retries: int — maximum retry attempts on transient errors.
    base_backoff_seconds: float — initial backoff duration.
    max_backoff_seconds: float — ceiling for exponential backoff.
    rate_limit_rpm: int — requests per minute limit.
    avg_input_tokens: int — average input tokens (for cost estimation).
    avg_output_tokens: int — average output tokens (for cost estimation).
    price_per_1k_input: float — USD per 1k input tokens.
    price_per_1k_output: float — USD per 1k output tokens.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import openai

from src.adapters.base_adapter import BaseAdapter, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """Adapter for the OpenAI Chat Completions API (GPT family models)."""

    # Map OpenAI finish reasons to our standardized finish_reason values.
    _FINISH_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "end_turn",
    }

    def __init__(self, model_name: str, max_tokens: int, api_config: dict) -> None:
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._api_config = api_config

        # Validate required config keys upfront so missing keys fail immediately.
        required_keys = [
            "timeout_seconds",
            "max_retries",
            "base_backoff_seconds",
            "max_backoff_seconds",
            "rate_limit_rpm",
            "avg_input_tokens",
            "avg_output_tokens",
            "price_per_1k_input",
            "price_per_1k_output",
        ]
        missing = [k for k in required_keys if k not in api_config]
        if missing:
            raise ValueError(
                f"OpenAIAdapter api_config missing required keys: {missing}"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before initializing the OpenAIAdapter."
            )

        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=float(api_config["timeout_seconds"]),
        )

        # Rate limiting state: enforce minimum interval between calls.
        self._min_interval_seconds = 60.0 / api_config["rate_limit_rpm"]
        self._last_call_time: Optional[float] = None

    def generate(self, prompt: str, temperature: float) -> LLMResponse:
        """Send a prompt to the OpenAI Chat API and return a standardized LLMResponse.

        Implements rate limiting, retry with exponential backoff, and
        graceful error handling. Never raises — returns an error
        LLMResponse on permanent failure.
        """
        max_retries = self._api_config["max_retries"]
        base_backoff = self._api_config["base_backoff_seconds"]
        max_backoff = self._api_config["max_backoff_seconds"]

        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            # --- Rate limiting ---
            self._enforce_rate_limit()

            try:
                start_time = time.perf_counter()

                response = self._client.chat.completions.create(
                    model=self._model_name,
                    max_completion_tokens=self._max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )

                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self._last_call_time = time.time()

                text = response.choices[0].message.content or ""
                raw_finish = response.choices[0].finish_reason
                finish_reason = self._FINISH_REASON_MAP.get(raw_finish, "end_turn")

                return LLMResponse(
                    text=text,
                    model=response.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    finish_reason=finish_reason,
                    latency_ms=elapsed_ms,
                    raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
                )

            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as exc:
                last_exception = exc
                self._last_call_time = time.time()

                if attempt < max_retries:
                    backoff = min(
                        base_backoff * (2 ** attempt),
                        max_backoff,
                    )
                    logger.warning(
                        "OpenAI API transient error (attempt %d/%d): %s. "
                        "Retrying in %.1fs.",
                        attempt + 1,
                        max_retries + 1,
                        type(exc).__name__,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "OpenAI API error after %d attempts: %s — %s",
                        max_retries + 1,
                        type(exc).__name__,
                        exc,
                    )

            except openai.APIError as exc:
                # Non-transient API errors (auth, bad request, etc.) —
                # retrying won't help, so fail immediately.
                logger.error(
                    "OpenAI API permanent error: %s — %s",
                    type(exc).__name__,
                    exc,
                )
                last_exception = exc
                break

        # All retries exhausted or permanent error — return error response.
        error_text = (
            f"[ERROR] {type(last_exception).__name__}: {last_exception}"
            if last_exception
            else "[ERROR] Unknown failure"
        )
        return LLMResponse(
            text=error_text,
            model=self._model_name,
            input_tokens=0,
            output_tokens=0,
            finish_reason="error",
            latency_ms=0.0,
            raw_response=None,
        )

    def estimate_cost(self, n_calls: int) -> float:
        """Estimate total USD cost for n_calls based on average token usage."""
        avg_in = self._api_config["avg_input_tokens"]
        avg_out = self._api_config["avg_output_tokens"]
        price_in = self._api_config["price_per_1k_input"]
        price_out = self._api_config["price_per_1k_output"]

        input_cost = n_calls * avg_in / 1000.0 * price_in
        output_cost = n_calls * avg_out / 1000.0 * price_out

        return input_cost + output_cost

    def provider_name(self) -> str:
        """Return 'openai'."""
        return "openai"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """Sleep if necessary to respect the configured RPM limit."""
        if self._last_call_time is not None:
            elapsed = time.time() - self._last_call_time
            wait = self._min_interval_seconds - elapsed
            if wait > 0:
                logger.debug("Rate limit: sleeping %.2fs", wait)
                time.sleep(wait)
