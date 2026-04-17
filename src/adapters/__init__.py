"""
Adapter registry with factory pattern.

Provides a central registry for LLM provider adapters so the engine can
instantiate any adapter by name (from config.yaml) without importing
provider-specific modules directly.

Usage:
    from src.adapters import get_adapter, LLMResponse

    AdapterClass = get_adapter("openai")
    adapter = AdapterClass(model_name="gpt-5-nano", max_tokens=1024, api_config={...})
    response: LLMResponse = adapter.generate("What is DBS NIM?", temperature=0.0)

To add a new provider:
    1. Create src/adapters/<provider>_adapter.py implementing BaseAdapter.
    2. Add a lazy entry in _LAZY_REGISTRY mapping name -> (module_path, class_name).
    That's it. No other files need to change.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Tuple, Type

from src.adapters.base_adapter import BaseAdapter, LLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseAdapter]] = {}

# Lazy registry: maps provider name -> (module_path, class_name).
# The actual import only happens when get_adapter() is called, so the
# package remains importable even if a provider's SDK is not installed.
_LAZY_REGISTRY: Dict[str, Tuple[str, str]] = {
    "anthropic": ("src.adapters.anthropic_adapter", "AnthropicAdapter"),
    "openai": ("src.adapters.openai_adapter", "OpenAIAdapter"),
}


def register_adapter(name: str, cls: Type[BaseAdapter]) -> None:
    """Register an adapter class under the given provider name.

    Args:
        name: Canonical provider name (lowercase, e.g. "anthropic").
        cls: Adapter class that implements BaseAdapter.

    Raises:
        TypeError: If cls is not a subclass of BaseAdapter.
        ValueError: If name is already registered (prevents silent overwrites).
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseAdapter)):
        raise TypeError(f"Cannot register {cls!r}: must be a subclass of BaseAdapter.")
    if name in _REGISTRY:
        raise ValueError(
            f"Adapter '{name}' is already registered as {_REGISTRY[name]!r}. "
            f"Unregister it first or use a different name."
        )
    _REGISTRY[name] = cls


def get_adapter(name: str) -> Type[BaseAdapter]:
    """Look up a registered adapter class by provider name.

    If the adapter hasn't been imported yet but exists in the lazy
    registry, it will be imported and registered on first access.

    Args:
        name: Canonical provider name (e.g. "anthropic").

    Returns:
        The adapter class (not an instance -- caller is responsible for
        instantiation with model_name, max_tokens, and api_config).

    Raises:
        ValueError: If no adapter is registered under that name.
        ImportError: If the adapter's provider SDK is not installed.
    """
    # Fast path: already registered.
    if name in _REGISTRY:
        return _REGISTRY[name]

    # Lazy import path.
    if name in _LAZY_REGISTRY:
        module_path, class_name = _LAZY_REGISTRY[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot load adapter '{name}': failed to import "
                f"'{module_path}'. Is the provider SDK installed? "
                f"Original error: {exc}"
            ) from exc
        cls = getattr(module, class_name)
        register_adapter(name, cls)
        return cls

    available = (
        ", ".join(sorted(set(_REGISTRY.keys()) | set(_LAZY_REGISTRY.keys())))
        or "(none)"
    )
    raise ValueError(f"Unknown adapter '{name}'. Available adapters: {available}")


# Re-export for convenience so callers can do:
#   from src.adapters import BaseAdapter, LLMResponse, get_adapter
__all__ = [
    "BaseAdapter",
    "LLMResponse",
    "register_adapter",
    "get_adapter",
]
