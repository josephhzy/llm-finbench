"""
Abstract base class for LLM provider adapters.

The adapter pattern is load-bearing architecture in this project. ALL
provider-specific code (SDK imports, auth, request formatting, response
parsing) lives exclusively in concrete adapter subclasses. The engine,
scorer, and reporter never import provider-specific SDKs — they interact
only with BaseAdapter and LLMResponse.

Why this matters: the evaluation harness must be provider-agnostic so that
the same pipeline can compare Claude, GPT-4, Gemini, or local models
without any changes to the measurement logic. If you find yourself importing
a provider SDK outside of adapters/, you are violating this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    Every adapter must return this exact structure regardless of the
    provider's native response format. This is the only type the rest
    of the pipeline ever sees.

    Attributes:
        text: The generated text content.
        model: The exact model identifier used (e.g. "claude-sonnet-4-20250514").
        input_tokens: Number of input tokens consumed (for cost tracking).
        output_tokens: Number of output tokens generated.
        finish_reason: One of "end_turn", "max_tokens", or "error".
            "end_turn" means normal completion. "max_tokens" means the
            response was truncated. "error" means all retries were
            exhausted and no valid response was obtained.
        latency_ms: Wall-clock time for the API call in milliseconds.
        raw_response: Optional dict of the provider's full response for
            debugging. Not used by the scoring pipeline.
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    latency_ms: float
    raw_response: Optional[dict] = field(default=None, repr=False)


class BaseAdapter(ABC):
    """Abstract base class that all LLM provider adapters must implement.

    Concrete adapters handle authentication, request formatting, retry
    logic, rate limiting, and response normalization. The rest of the
    pipeline only calls the methods defined here.
    """

    @abstractmethod
    def __init__(self, model_name: str, max_tokens: int, api_config: dict) -> None:
        """Initialize the adapter with model and configuration.

        Args:
            model_name: Provider-specific model identifier
                (e.g. "claude-sonnet-4-20250514").
            max_tokens: Maximum tokens to generate per call.
            api_config: Provider-specific configuration dict. At minimum
                should contain timeout, retry, and cost estimation
                parameters. See each adapter's docstring for required keys.
        """
        ...

    @abstractmethod
    def generate(self, prompt: str, temperature: float) -> LLMResponse:
        """Generate a single response for the given prompt.

        This method must handle retries, rate limiting, and error
        recovery internally. The caller should never need to catch
        provider-specific exceptions.

        Args:
            prompt: The full prompt text to send to the model.
            temperature: Sampling temperature (0.0 = deterministic,
                1.0 = maximum randomness).

        Returns:
            LLMResponse with the result. On permanent failure (all
            retries exhausted), returns an LLMResponse with
            finish_reason="error" and empty text — never raises.
        """
        ...

    @abstractmethod
    def estimate_cost(self, n_calls: int) -> float:
        """Estimate the total USD cost for n_calls API calls.

        Uses average token counts from api_config to project cost.
        This is called before execution so the user can approve spend.

        Args:
            n_calls: Number of API calls planned.

        Returns:
            Estimated cost in USD.
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical provider name (e.g. 'anthropic', 'openai').

        Used for logging, report metadata, and adapter registry lookups.
        """
        ...
