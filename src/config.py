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

    Token counts are rough averages — the real cost depends on prompt length
    and response verbosity. These defaults are calibrated for typical
    financial extraction prompts (~200 input tokens including the question
    and a short context snippet, ~300 output tokens for a detailed answer).
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


def _build_cost(raw: Dict[str, Any]) -> CostConfig:
    return CostConfig(
        avg_input_tokens=int(raw["avg_input_tokens"]),
        avg_output_tokens=int(raw["avg_output_tokens"]),
        price_per_1k_input=float(raw["price_per_1k_input"]),
        price_per_1k_output=float(raw["price_per_1k_output"]),
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
        cost=_build_cost(raw["cost"]),
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
