"""Configuration loader and validator for LLM Financial Stability Bench.

Loads config.yaml into frozen dataclasses for typed, immutable access to all
evaluation parameters. The frozen constraint prevents accidental mutation
mid-run, which would silently compromise reproducibility.

Key design decisions:
- Frozen dataclasses over pydantic: zero external deps beyond PyYAML, and we
  need serialisation control (snapshot_config) more than we need schema DSLs.
- validate_config is separate from construction so callers can inspect a
  config object before validation (useful for debugging bad YAML).
- quick mode replaces evaluation fields wholesale rather than merging, because
  partial overrides create ambiguous semantics (e.g. does quick_mode inherit
  the full template list if it omits the key?).
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Model Pricing Registry
# ---------------------------------------------------------------------------
# Prices in USD per 1,000 tokens.
# Sources:
#   Anthropic — https://docs.anthropic.com/en/docs/about-claude/models
#   OpenAI    — https://openai.com/api/pricing/
#
# Keys use the model name prefix so date-suffixed variants like
# "claude-haiku-4-5-20251001" match "claude-haiku-4-5" automatically.

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # ── Anthropic Claude ──────────────────────────────────────────────────
    "claude-opus-4-6":   {"input": 0.005,    "output": 0.025},   # $5/$25 per MTok
    "claude-opus-4-5":   {"input": 0.005,    "output": 0.025},
    "claude-opus-4-1":   {"input": 0.015,    "output": 0.075},   # $15/$75 per MTok
    "claude-opus-4":     {"input": 0.015,    "output": 0.075},
    "claude-opus-3":     {"input": 0.015,    "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003,    "output": 0.015},   # $3/$15 per MTok
    "claude-sonnet-4-5": {"input": 0.003,    "output": 0.015},
    "claude-sonnet-4":   {"input": 0.003,    "output": 0.015},
    "claude-sonnet-3-7": {"input": 0.003,    "output": 0.015},
    "claude-haiku-4-5":  {"input": 0.001,    "output": 0.005},   # $1/$5 per MTok
    "claude-haiku-3-5":  {"input": 0.0008,   "output": 0.004},   # $0.80/$4 per MTok
    "claude-haiku-3":    {"input": 0.00025,  "output": 0.00125}, # $0.25/$1.25 per MTok
    # ── OpenAI ───────────────────────────────────────────────────────────
    "gpt-5.4":           {"input": 0.0025,   "output": 0.015},   # $2.50/$15 per MTok
    "gpt-5.4-mini":      {"input": 0.00075,  "output": 0.0045},  # $0.75/$4.50 per MTok
    "gpt-5.4-nano":      {"input": 0.0002,   "output": 0.00125}, # $0.20/$1.25 per MTok
}


def pricing_for_model(model_name: str) -> Dict[str, float]:
    """Look up per-1K-token pricing for a given model name.

    Matches by prefix so date-suffixed variants like
    ``claude-haiku-4-5-20251001`` resolve to ``claude-haiku-4-5``.

    Returns
    -------
    dict with ``"input"`` and ``"output"`` keys (USD per 1K tokens).

    Raises
    ------
    ValueError
        If no matching entry is found in MODEL_PRICING.
    """
    model_lower = model_name.lower()
    # Exact match first
    if model_lower in MODEL_PRICING:
        return MODEL_PRICING[model_lower]
    # Prefix match — handles date suffixes (e.g. -20251001)
    for key, pricing in MODEL_PRICING.items():
        if model_lower.startswith(key):
            return pricing
    raise ValueError(
        f"No pricing found for model {model_name!r}. "
        f"Add it to MODEL_PRICING in src/config.py, or set explicit "
        f"price_per_1k_input / price_per_1k_output in config.yaml."
    )


# ---------------------------------------------------------------------------
# Config section dataclasses (all frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Which LLM provider and model to evaluate."""
    provider: str
    name: str
    max_tokens: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Parameters controlling the evaluation sweep.

    Each fact is tested across every (template x temperature) combination,
    repeated runs_per_combination times. This gives N = |templates| x
    |temperatures| x runs_per_combination data points per fact — enough to
    compute meaningful variance when runs_per_combination >= 10.
    """
    temperatures: List[float]
    runs_per_combination: int
    templates: List[str]


@dataclass(frozen=True)
class QuickModeConfig:
    """Reduced-scope overrides for development iteration.

    Quick mode exists so you never accidentally burn $15 while debugging a
    prompt template. It constrains the sweep to a single temperature, fewer
    runs, and a subset of facts.
    """
    temperatures: List[float]
    runs_per_combination: int
    max_facts: int
    templates: List[str]


@dataclass(frozen=True)
class ScoringConfig:
    """How generated responses are scored.

    - embedding_model: sentence-transformers model for semantic similarity.
      Must stay constant across an entire evaluation run — mixing models
      makes cosine similarities incomparable.
    - hallucination_tolerance: relative tolerance for numeric comparison.
      0.05 means 5% — so 2.14 vs 2.24 is within tolerance (|diff|/expected
      = 0.047), but 2.14 vs 3.41 is a hallucination.
    - composite_weights: how the three sub-scores combine into one stability
      score. Factual consistency gets the highest weight because getting the
      number right matters more than phrasing consistency.
    """
    embedding_model: str
    hallucination_tolerance: float
    composite_weights: Dict[str, float]


@dataclass(frozen=True)
class FlaggingConfig:
    """Thresholds for traffic-light flagging in reports.

    green_threshold > yellow_threshold is enforced by validation.
    Scores >= green are stable, >= yellow are caution, below yellow are
    flagged as unreliable.
    """
    green_threshold: float
    yellow_threshold: float


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpointing to survive crashes during long evaluation runs.

    save_interval: persist results every N API calls. 50 is a good default —
    frequent enough to limit data loss, infrequent enough to avoid I/O
    overhead dominating runtime.
    """
    save_interval: int
    directory: str


@dataclass(frozen=True)
class CostConfig:
    """API cost estimation parameters.

    Token counts are calibrated from the actual benchmark run (6,600 calls):
      avg_input_tokens  ≈ 36  (short factual question, no context padding)
      avg_output_tokens ≈ 35  (model returns a number, not a paragraph)

    Prices are auto-filled from MODEL_PRICING based on model.name if
    price_per_1k_input / price_per_1k_output are absent or zero in
    config.yaml. Set them explicitly in config.yaml to override.
    """
    avg_input_tokens: int
    avg_output_tokens: int
    price_per_1k_input: float
    price_per_1k_output: float
    confirmation_threshold: float


@dataclass(frozen=True)
class ApiConfig:
    """Resilience settings for API calls.

    Exponential backoff: wait = min(base_backoff * 2^attempt, max_backoff).
    rate_limit_rpm caps request rate to stay within provider quotas.
    """
    timeout_seconds: int
    max_retries: int
    base_backoff_seconds: float
    max_backoff_seconds: float
    rate_limit_rpm: int


@dataclass(frozen=True)
class AppConfig:
    """Top-level container for all configuration sections."""
    model: ModelConfig
    evaluation: EvaluationConfig
    quick_mode: QuickModeConfig
    scoring: ScoringConfig
    flagging: FlaggingConfig
    checkpoint: CheckpointConfig
    cost: CostConfig
    api: ApiConfig


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _build_model(raw: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        provider=str(raw["provider"]),
        name=str(raw["name"]),
        max_tokens=int(raw["max_tokens"]),
    )


def _build_evaluation(raw: Dict[str, Any]) -> EvaluationConfig:
    return EvaluationConfig(
        temperatures=[float(t) for t in raw["temperatures"]],
        runs_per_combination=int(raw["runs_per_combination"]),
        templates=[str(t) for t in raw["templates"]],
    )


def _build_quick_mode(raw: Dict[str, Any]) -> QuickModeConfig:
    return QuickModeConfig(
        temperatures=[float(t) for t in raw["temperatures"]],
        runs_per_combination=int(raw["runs_per_combination"]),
        max_facts=int(raw["max_facts"]),
        templates=[str(t) for t in raw["templates"]],
    )


def _build_scoring(raw: Dict[str, Any]) -> ScoringConfig:
    return ScoringConfig(
        embedding_model=str(raw["embedding_model"]),
        hallucination_tolerance=float(raw["hallucination_tolerance"]),
        composite_weights={str(k): float(v) for k, v in raw["composite_weights"].items()},
    )


def _build_flagging(raw: Dict[str, Any]) -> FlaggingConfig:
    return FlaggingConfig(
        green_threshold=float(raw["green_threshold"]),
        yellow_threshold=float(raw["yellow_threshold"]),
    )


def _build_checkpoint(raw: Dict[str, Any]) -> CheckpointConfig:
    return CheckpointConfig(
        save_interval=int(raw["save_interval"]),
        directory=str(raw["directory"]),
    )


def _build_cost(raw: Dict[str, Any], model_name: str) -> CostConfig:
    """Build CostConfig, auto-filling prices from MODEL_PRICING if not set.

    If ``price_per_1k_input`` or ``price_per_1k_output`` are absent, zero,
    or the string ``"auto"`` in config.yaml, their values are looked up from
    the MODEL_PRICING registry using ``model_name``.
    """
    def _resolve(key: str, field: str) -> float:
        val = raw.get(key)
        if val is None or str(val).strip().lower() == "auto":
            return pricing_for_model(model_name)[field]
        val = float(val)
        return pricing_for_model(model_name)[field] if val == 0.0 else val

    return CostConfig(
        avg_input_tokens=int(raw["avg_input_tokens"]),
        avg_output_tokens=int(raw["avg_output_tokens"]),
        price_per_1k_input=_resolve("price_per_1k_input", "input"),
        price_per_1k_output=_resolve("price_per_1k_output", "output"),
        confirmation_threshold=float(raw["confirmation_threshold"]),
    )


def _build_api(raw: Dict[str, Any]) -> ApiConfig:
    return ApiConfig(
        timeout_seconds=int(raw["timeout_seconds"]),
        max_retries=int(raw["max_retries"]),
        base_backoff_seconds=float(raw["base_backoff_seconds"]),
        max_backoff_seconds=float(raw["max_backoff_seconds"]),
        rate_limit_rpm=int(raw["rate_limit_rpm"]),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS = [
    "model", "evaluation", "quick_mode", "scoring",
    "flagging", "checkpoint", "cost", "api",
]


def load_config(path: str = "config.yaml", quick: bool = False) -> AppConfig:
    """Load config.yaml and return a validated, frozen AppConfig.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.
    quick : bool
        If True, override evaluation parameters with quick_mode values so
        that development runs are fast and cheap.

    Returns
    -------
    AppConfig
        Fully constructed and validated configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required section or field is missing from the YAML.
    ValueError
        If validation fails (see validate_config).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    missing = [s for s in _REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise KeyError(f"Missing required config sections: {missing}")

    evaluation_cfg = _build_evaluation(raw["evaluation"])
    quick_mode_cfg = _build_quick_mode(raw["quick_mode"])

    # In quick mode, replace evaluation with quick_mode values so the rest
    # of the pipeline doesn't need to know about quick mode at all.
    if quick:
        evaluation_cfg = EvaluationConfig(
            temperatures=quick_mode_cfg.temperatures,
            runs_per_combination=quick_mode_cfg.runs_per_combination,
            templates=quick_mode_cfg.templates,
        )

    config = AppConfig(
        model=_build_model(raw["model"]),
        evaluation=evaluation_cfg,
        quick_mode=quick_mode_cfg,
        scoring=_build_scoring(raw["scoring"]),
        flagging=_build_flagging(raw["flagging"]),
        checkpoint=_build_checkpoint(raw["checkpoint"]),
        cost=_build_cost(raw["cost"], model_name=raw["model"]["name"]),
        api=_build_api(raw["api"]),
    )

    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Validate cross-field constraints that dataclass types alone can't enforce.

    Raises ValueError with a descriptive message on the first violation found.
    Checks are ordered from most likely user error to least likely.

    Parameters
    ----------
    config : AppConfig
        The configuration to validate.

    Raises
    ------
    ValueError
        If any constraint is violated.
    """
    errors: List[str] = []

    # --- Composite weights must sum to ~1.0 ---
    weight_sum = sum(config.scoring.composite_weights.values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
        errors.append(
            f"scoring.composite_weights must sum to 1.0, got {weight_sum:.6f}"
        )

    expected_weight_keys = {"semantic_consistency", "factual_consistency", "hallucination_rate"}
    actual_weight_keys = set(config.scoring.composite_weights.keys())
    if actual_weight_keys != expected_weight_keys:
        errors.append(
            f"scoring.composite_weights keys must be {expected_weight_keys}, "
            f"got {actual_weight_keys}"
        )

    # --- Temperatures in valid range [0, 2] ---
    for temp in config.evaluation.temperatures:
        if not 0.0 <= temp <= 2.0:
            errors.append(
                f"evaluation.temperatures contains {temp}, must be in [0.0, 2.0]"
            )

    for temp in config.quick_mode.temperatures:
        if not 0.0 <= temp <= 2.0:
            errors.append(
                f"quick_mode.temperatures contains {temp}, must be in [0.0, 2.0]"
            )

    # --- Flagging thresholds ordered correctly ---
    if config.flagging.green_threshold <= config.flagging.yellow_threshold:
        errors.append(
            f"flagging.green_threshold ({config.flagging.green_threshold}) must be "
            f"strictly greater than yellow_threshold ({config.flagging.yellow_threshold})"
        )

    # --- Thresholds in [0, 1] ---
    for name, val in [
        ("flagging.green_threshold", config.flagging.green_threshold),
        ("flagging.yellow_threshold", config.flagging.yellow_threshold),
    ]:
        if not 0.0 <= val <= 1.0:
            errors.append(f"{name} must be in [0.0, 1.0], got {val}")

    # --- Positive integers ---
    if config.evaluation.runs_per_combination < 1:
        errors.append("evaluation.runs_per_combination must be >= 1")
    if config.quick_mode.runs_per_combination < 1:
        errors.append("quick_mode.runs_per_combination must be >= 1")
    if config.quick_mode.max_facts < 1:
        errors.append("quick_mode.max_facts must be >= 1")
    if config.model.max_tokens < 1:
        errors.append("model.max_tokens must be >= 1")
    if config.checkpoint.save_interval < 1:
        errors.append("checkpoint.save_interval must be >= 1")

    # --- Non-empty lists ---
    if not config.evaluation.temperatures:
        errors.append("evaluation.temperatures must not be empty")
    if not config.evaluation.templates:
        errors.append("evaluation.templates must not be empty")

    # --- API resilience sanity ---
    if config.api.timeout_seconds < 1:
        errors.append("api.timeout_seconds must be >= 1")
    if config.api.max_retries < 0:
        errors.append("api.max_retries must be >= 0")
    if config.api.base_backoff_seconds <= 0:
        errors.append("api.base_backoff_seconds must be > 0")
    if config.api.max_backoff_seconds < config.api.base_backoff_seconds:
        errors.append(
            f"api.max_backoff_seconds ({config.api.max_backoff_seconds}) must be "
            f">= base_backoff_seconds ({config.api.base_backoff_seconds})"
        )
    if config.api.rate_limit_rpm < 1:
        errors.append("api.rate_limit_rpm must be >= 1")

    # --- Hallucination tolerance ---
    if not 0.0 < config.scoring.hallucination_tolerance < 1.0:
        errors.append(
            f"scoring.hallucination_tolerance must be in (0.0, 1.0), "
            f"got {config.scoring.hallucination_tolerance}"
        )

    # --- Cost fields non-negative ---
    if config.cost.avg_input_tokens < 0:
        errors.append("cost.avg_input_tokens must be >= 0")
    if config.cost.avg_output_tokens < 0:
        errors.append("cost.avg_output_tokens must be >= 0")
    if config.cost.price_per_1k_input < 0:
        errors.append("cost.price_per_1k_input must be >= 0")
    if config.cost.price_per_1k_output < 0:
        errors.append("cost.price_per_1k_output must be >= 0")
    if config.cost.confirmation_threshold < 0:
        errors.append("cost.confirmation_threshold must be >= 0")

    # --- Model fields non-empty ---
    if not config.model.provider.strip():
        errors.append("model.provider must not be empty")
    if not config.model.name.strip():
        errors.append("model.name must not be empty")

    # --- Scoring embedding model non-empty ---
    if not config.scoring.embedding_model.strip():
        errors.append("scoring.embedding_model must not be empty")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )


def snapshot_config(config: AppConfig) -> dict:
    """Serialise the config to a plain dict for reproducibility logging.

    The returned dict is JSON/YAML-serialisable and should be saved alongside
    every evaluation run's results. This lets anyone reconstruct exactly which
    parameters produced a given set of scores — even if config.yaml has since
    been modified.

    Parameters
    ----------
    config : AppConfig
        The configuration to snapshot.

    Returns
    -------
    dict
        A deep-copied, plain-dict representation of the entire config.
    """
    # dataclasses.asdict already does recursive conversion to dicts/lists.
    # We deep-copy to guarantee the caller can't mutate internal state
    # (extra safety given frozen dataclasses).
    return copy.deepcopy(asdict(config))
