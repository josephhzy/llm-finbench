"""Tests for the adapter registry and base adapter classes.

Covers adapter registry lookup, error handling for unknown providers,
LLMResponse dataclass fields, retry behaviour (mocked), and rate limiting.
No real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.adapters import (
    BaseAdapter,
    LLMResponse,
    get_adapter,
    register_adapter,
    _REGISTRY,
    _LAZY_REGISTRY,
)


# ===========================================================================
# LLMResponse dataclass
# ===========================================================================


class TestLLMResponse:
    """Test that LLMResponse has all expected fields."""

    def test_all_required_fields(self):
        resp = LLMResponse(
            text="Hello",
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            finish_reason="end_turn",
            latency_ms=50.0,
        )
        assert resp.text == "Hello"
        assert resp.model == "test-model"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert resp.finish_reason == "end_turn"
        assert resp.latency_ms == 50.0
        assert resp.raw_response is None  # optional, default None

    def test_raw_response_optional(self):
        resp = LLMResponse(
            text="test",
            model="m",
            input_tokens=0,
            output_tokens=0,
            finish_reason="end_turn",
            latency_ms=0.0,
            raw_response={"id": "test-123"},
        )
        assert resp.raw_response == {"id": "test-123"}

    def test_finish_reason_error(self):
        resp = LLMResponse(
            text="[ERROR] RateLimitError",
            model="m",
            input_tokens=0,
            output_tokens=0,
            finish_reason="error",
            latency_ms=0.0,
        )
        assert resp.finish_reason == "error"

    def test_finish_reason_max_tokens(self):
        resp = LLMResponse(
            text="truncated...",
            model="m",
            input_tokens=100,
            output_tokens=1024,
            finish_reason="max_tokens",
            latency_ms=200.0,
        )
        assert resp.finish_reason == "max_tokens"


# ===========================================================================
# Adapter registry
# ===========================================================================


class TestAdapterRegistry:
    """Test the adapter registry's get_adapter and register_adapter."""

    def test_get_adapter_returns_correct_type_for_known_providers(self):
        """The lazy registry should list 'anthropic' and 'openai'."""
        assert "anthropic" in _LAZY_REGISTRY
        assert "openai" in _LAZY_REGISTRY

    def test_get_adapter_raises_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_provider_xyz")

    def test_unknown_adapter_error_message_includes_available(self):
        """The error message should list available adapters."""
        with pytest.raises(ValueError, match="anthropic"):
            get_adapter("nonexistent_provider_xyz")

    def test_register_adapter_rejects_non_subclass(self):
        """Cannot register a class that doesn't subclass BaseAdapter."""
        with pytest.raises(TypeError, match="subclass of BaseAdapter"):
            register_adapter("bad_adapter", str)  # type: ignore

    def test_register_adapter_rejects_duplicate(self):
        """Cannot re-register an already-registered adapter name."""

        class DummyAdapter(BaseAdapter):
            def __init__(self, model_name, max_tokens, api_config):
                pass

            def generate(self, prompt, temperature):
                pass

            def estimate_cost(self, n_calls):
                return 0.0

            def provider_name(self):
                return "dummy"

        # Use a unique name that won't collide
        name = "_test_duplicate_adapter_"
        try:
            # Ensure clean state
            _REGISTRY.pop(name, None)
            register_adapter(name, DummyAdapter)
            with pytest.raises(ValueError, match="already registered"):
                register_adapter(name, DummyAdapter)
        finally:
            _REGISTRY.pop(name, None)

    def test_lazy_registry_entries_have_module_and_class(self):
        """Each lazy registry entry should be a (module_path, class_name) tuple."""
        for name, entry in _LAZY_REGISTRY.items():
            assert isinstance(entry, tuple), f"Entry for '{name}' is not a tuple"
            assert len(entry) == 2, f"Entry for '{name}' should have 2 elements"
            module_path, class_name = entry
            assert isinstance(module_path, str)
            assert isinstance(class_name, str)


# ===========================================================================
# Retry behaviour (mocked adapter)
# ===========================================================================


class TestRetryBehaviour:
    """Test retry logic using a mock that simulates transient failures.

    We test the OpenAI adapter's retry logic by patching only the OpenAI
    client, not the entire openai module. This lets the real exception
    classes (RateLimitError, etc.) be used in except clauses.
    """

    def _make_api_config(self) -> dict:
        return {
            "timeout_seconds": 5,
            "max_retries": 2,
            "base_backoff_seconds": 0.01,  # tiny backoff for fast tests
            "max_backoff_seconds": 0.05,
            "rate_limit_rpm": 6000,  # high RPM to avoid rate limit sleeps
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "price_per_1k_input": 0.001,
            "price_per_1k_output": 0.002,
        }

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai.OpenAI")
    def test_retry_on_transient_error_then_succeed(self, mock_openai_cls):
        """Adapter should retry on RateLimitError and succeed on next attempt."""
        import openai as real_openai
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Create a mock response for the success case
        mock_choice = MagicMock()
        mock_choice.message.content = "2.14%"
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 20
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "test-model"

        # Build a real RateLimitError using the openai SDK's exception class
        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}
        rate_error = real_openai.RateLimitError(
            "rate limited",
            response=error_response,
            body=None,
        )

        # First call raises, second succeeds
        mock_client.chat.completions.create.side_effect = [
            rate_error,
            mock_response,
        ]

        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=self._make_api_config(),
        )

        response = adapter.generate("test prompt", 0.0)

        # Should have been called twice (first fail, then success)
        assert mock_client.chat.completions.create.call_count == 2
        assert response.text == "2.14%"
        assert response.finish_reason == "end_turn"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai.OpenAI")
    def test_all_retries_exhausted_returns_error(self, mock_openai_cls):
        """After all retries, adapter should return error LLMResponse."""
        import openai as real_openai
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Build a real RateLimitError
        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}
        rate_error = real_openai.RateLimitError(
            "rate limited",
            response=error_response,
            body=None,
        )

        mock_client.chat.completions.create.side_effect = rate_error

        api_config = self._make_api_config()
        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=api_config,
        )

        response = adapter.generate("test prompt", 0.0)

        # max_retries=2 means 3 total attempts (initial + 2 retries)
        assert mock_client.chat.completions.create.call_count == 3
        assert response.finish_reason == "error"
        assert "[ERROR]" in response.text


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting logic."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai")
    def test_rate_limit_interval_calculation(self, mock_openai_module):
        """Rate limit should compute min interval from RPM."""
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_openai_module.OpenAI.return_value = MagicMock()

        api_config = {
            "timeout_seconds": 5,
            "max_retries": 0,
            "base_backoff_seconds": 1.0,
            "max_backoff_seconds": 60.0,
            "rate_limit_rpm": 60,  # 60 RPM = 1 req/sec = 1.0s interval
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "price_per_1k_input": 0.001,
            "price_per_1k_output": 0.002,
        }

        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=api_config,
        )

        # 60 RPM = 1 request per second
        assert abs(adapter._min_interval_seconds - 1.0) < 1e-9

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai")
    def test_rate_limit_rpm_120(self, mock_openai_module):
        """120 RPM = 0.5s interval."""
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_openai_module.OpenAI.return_value = MagicMock()

        api_config = {
            "timeout_seconds": 5,
            "max_retries": 0,
            "base_backoff_seconds": 1.0,
            "max_backoff_seconds": 60.0,
            "rate_limit_rpm": 120,
            "avg_input_tokens": 100,
            "avg_output_tokens": 50,
            "price_per_1k_input": 0.001,
            "price_per_1k_output": 0.002,
        }

        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=api_config,
        )

        assert abs(adapter._min_interval_seconds - 0.5) < 1e-9


# ===========================================================================
# Cost estimation
# ===========================================================================


class TestAdapterCostEstimation:
    """Test adapter-level cost estimation."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai")
    def test_cost_estimation(self, mock_openai_module):
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_openai_module.OpenAI.return_value = MagicMock()

        api_config = {
            "timeout_seconds": 5,
            "max_retries": 0,
            "base_backoff_seconds": 1.0,
            "max_backoff_seconds": 60.0,
            "rate_limit_rpm": 60,
            "avg_input_tokens": 200,
            "avg_output_tokens": 300,
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
        }

        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=api_config,
        )

        cost = adapter.estimate_cost(100)
        # input: 100 * 200/1000 * 0.003 = 0.06
        # output: 100 * 300/1000 * 0.015 = 0.45
        expected = 0.06 + 0.45
        assert abs(cost - expected) < 1e-9

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai")
    def test_cost_zero_calls(self, mock_openai_module):
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_openai_module.OpenAI.return_value = MagicMock()

        api_config = {
            "timeout_seconds": 5,
            "max_retries": 0,
            "base_backoff_seconds": 1.0,
            "max_backoff_seconds": 60.0,
            "rate_limit_rpm": 60,
            "avg_input_tokens": 200,
            "avg_output_tokens": 300,
            "price_per_1k_input": 0.003,
            "price_per_1k_output": 0.015,
        }

        adapter = OpenAIAdapter(
            model_name="test-model",
            max_tokens=512,
            api_config=api_config,
        )

        assert adapter.estimate_cost(0) == 0.0


# ===========================================================================
# Missing API config keys
# ===========================================================================


class TestAdapterConfigValidation:
    """Test that adapters reject incomplete api_config."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"})
    @patch("src.adapters.openai_adapter.openai")
    def test_missing_config_key_raises(self, mock_openai_module):
        from src.adapters.openai_adapter import OpenAIAdapter

        mock_openai_module.OpenAI.return_value = MagicMock()

        with pytest.raises(ValueError, match="missing required keys"):
            OpenAIAdapter(
                model_name="test",
                max_tokens=512,
                api_config={"timeout_seconds": 5},  # missing many keys
            )
