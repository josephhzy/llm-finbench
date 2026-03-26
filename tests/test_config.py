"""Tests for the configuration loader and validator.

Covers loading from the real config.yaml, quick mode overrides,
validation error detection, and config snapshot serialisation.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import pytest
import yaml

from src.config import (
    ApiConfig,
    AppConfig,
    CheckpointConfig,
    CostConfig,
    EvaluationConfig,
    FlaggingConfig,
    ModelConfig,
    QuickModeConfig,
    ScoringConfig,
    load_config,
    snapshot_config,
    validate_config,
)

# Path to the real config.yaml
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "config.yaml"
)


# ===========================================================================
# Loading from real config.yaml
# ===========================================================================


class TestLoadConfig:
    """Test that load_config correctly parses the real config.yaml."""

    def test_loads_successfully(self):
        config = load_config(CONFIG_PATH)
        assert isinstance(config, AppConfig)

    def test_model_section(self):
        config = load_config(CONFIG_PATH)
        assert config.model.provider == "anthropic"
        assert config.model.name == "claude-sonnet-4-20250514"
        assert config.model.max_tokens == 1024

    def test_evaluation_section(self):
        config = load_config(CONFIG_PATH)
        assert config.evaluation.temperatures == [0.0, 0.3, 0.5, 0.7, 1.0]
        assert config.evaluation.runs_per_combination == 10
        assert len(config.evaluation.templates) == 4
        assert "direct_extraction" in config.evaluation.templates

    def test_scoring_section(self):
        config = load_config(CONFIG_PATH)
        assert config.scoring.embedding_model == "all-MiniLM-L6-v2"
        assert config.scoring.hallucination_tolerance == 0.05
        weights = config.scoring.composite_weights
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)

    def test_flagging_section(self):
        config = load_config(CONFIG_PATH)
        assert config.flagging.green_threshold == 0.75
        assert config.flagging.yellow_threshold == 0.50

    def test_checkpoint_section(self):
        config = load_config(CONFIG_PATH)
        assert config.checkpoint.save_interval == 50
        assert config.checkpoint.directory == "results"

    def test_cost_section(self):
        config = load_config(CONFIG_PATH)
        assert config.cost.avg_input_tokens == 200
        assert config.cost.avg_output_tokens == 300
        assert config.cost.confirmation_threshold == 1.0

    def test_api_section(self):
        config = load_config(CONFIG_PATH)
        assert config.api.timeout_seconds == 30
        assert config.api.max_retries == 5
        assert config.api.rate_limit_rpm == 50

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_frozen_immutability(self):
        """Config objects are frozen — attribute assignment should raise."""
        config = load_config(CONFIG_PATH)
        with pytest.raises(AttributeError):
            config.model = None  # type: ignore[misc]


# ===========================================================================
# Quick mode overrides
# ===========================================================================


class TestQuickMode:
    """Test that quick=True replaces evaluation params with quick_mode values."""

    def test_quick_temperatures(self):
        config = load_config(CONFIG_PATH, quick=True)
        assert config.evaluation.temperatures == [0.0]

    def test_quick_runs_per_combination(self):
        config = load_config(CONFIG_PATH, quick=True)
        assert config.evaluation.runs_per_combination == 3

    def test_quick_templates(self):
        config = load_config(CONFIG_PATH, quick=True)
        assert config.evaluation.templates == ["direct_extraction"]

    def test_quick_mode_preserves_other_sections(self):
        """Quick mode only changes evaluation — scoring, flagging, etc. stay."""
        normal = load_config(CONFIG_PATH, quick=False)
        quick = load_config(CONFIG_PATH, quick=True)
        assert quick.scoring == normal.scoring
        assert quick.flagging == normal.flagging
        assert quick.model == normal.model
        assert quick.api == normal.api

    def test_quick_mode_original_still_stored(self):
        """The original quick_mode config is still accessible."""
        config = load_config(CONFIG_PATH, quick=True)
        assert config.quick_mode.max_facts == 6


# ===========================================================================
# Validation: composite weights
# ===========================================================================


class TestValidateWeights:
    """Test that validate_config catches bad composite weights."""

    def _make_config(self, **weight_overrides):
        """Build an AppConfig with custom scoring weights."""
        weights = {
            "semantic_consistency": 0.30,
            "factual_consistency": 0.40,
            "hallucination_rate": 0.30,
        }
        weights.update(weight_overrides)
        return AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[0.0], runs_per_combination=3, templates=["direct_extraction"]
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"]
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights=weights,
            ),
            flagging=FlaggingConfig(green_threshold=0.75, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )

    def test_weights_not_summing_to_one(self):
        config = self._make_config(
            semantic_consistency=0.50,
            factual_consistency=0.40,
            hallucination_rate=0.30,
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_config(config)

    def test_valid_weights_pass(self):
        config = self._make_config()
        # Should not raise
        validate_config(config)

    def test_wrong_weight_keys(self):
        """Weights with wrong key names should fail validation."""
        config = AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[0.0], runs_per_combination=3, templates=["direct_extraction"]
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"]
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights={
                    "wrong_key": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            ),
            flagging=FlaggingConfig(green_threshold=0.75, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )
        with pytest.raises(ValueError, match="keys must be"):
            validate_config(config)


# ===========================================================================
# Validation: temperature range
# ===========================================================================


class TestValidateTemperatures:
    """Test that validate_config catches out-of-range temperatures."""

    def test_temperature_above_two(self):
        """Temperature > 2.0 should fail validation."""
        config = AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[0.0, 2.5],  # 2.5 is out of range
                runs_per_combination=3,
                templates=["direct_extraction"],
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"],
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights={
                    "semantic_consistency": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            ),
            flagging=FlaggingConfig(green_threshold=0.75, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )
        with pytest.raises(ValueError, match="2.5"):
            validate_config(config)

    def test_negative_temperature(self):
        config = AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[-0.1],
                runs_per_combination=3,
                templates=["direct_extraction"],
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"],
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights={
                    "semantic_consistency": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            ),
            flagging=FlaggingConfig(green_threshold=0.75, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )
        with pytest.raises(ValueError, match="-0.1"):
            validate_config(config)


# ===========================================================================
# Validation: threshold ordering
# ===========================================================================


class TestValidateThresholds:
    """Test that green_threshold > yellow_threshold is enforced."""

    def test_green_equals_yellow_fails(self):
        config = AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[0.0], runs_per_combination=3, templates=["direct_extraction"]
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"]
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights={
                    "semantic_consistency": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            ),
            flagging=FlaggingConfig(green_threshold=0.50, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )
        with pytest.raises(ValueError, match="strictly greater"):
            validate_config(config)

    def test_green_less_than_yellow_fails(self):
        config = AppConfig(
            model=ModelConfig(provider="test", name="test-model", max_tokens=512),
            evaluation=EvaluationConfig(
                temperatures=[0.0], runs_per_combination=3, templates=["direct_extraction"]
            ),
            quick_mode=QuickModeConfig(
                temperatures=[0.0], runs_per_combination=3, max_facts=5,
                templates=["direct_extraction"]
            ),
            scoring=ScoringConfig(
                embedding_model="all-MiniLM-L6-v2",
                hallucination_tolerance=0.05,
                composite_weights={
                    "semantic_consistency": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            ),
            flagging=FlaggingConfig(green_threshold=0.40, yellow_threshold=0.50),
            checkpoint=CheckpointConfig(save_interval=50, directory="results"),
            cost=CostConfig(
                avg_input_tokens=200, avg_output_tokens=300,
                price_per_1k_input=0.003, price_per_1k_output=0.015,
                confirmation_threshold=1.0,
            ),
            api=ApiConfig(
                timeout_seconds=30, max_retries=5, base_backoff_seconds=1.0,
                max_backoff_seconds=60.0, rate_limit_rpm=50,
            ),
        )
        with pytest.raises(ValueError, match="strictly greater"):
            validate_config(config)


# ===========================================================================
# snapshot_config
# ===========================================================================


class TestSnapshotConfig:
    """Test that snapshot_config returns a serialisable dict."""

    def test_returns_dict(self):
        config = load_config(CONFIG_PATH)
        snap = snapshot_config(config)
        assert isinstance(snap, dict)

    def test_json_serialisable(self):
        config = load_config(CONFIG_PATH)
        snap = snapshot_config(config)
        # Should not raise
        json_str = json.dumps(snap)
        assert isinstance(json_str, str)

    def test_contains_all_sections(self):
        config = load_config(CONFIG_PATH)
        snap = snapshot_config(config)
        expected_keys = [
            "model", "evaluation", "quick_mode", "scoring",
            "flagging", "checkpoint", "cost", "api",
        ]
        for key in expected_keys:
            assert key in snap, f"Missing section: {key}"

    def test_snapshot_is_deep_copy(self):
        """Mutating the snapshot should not affect the original config."""
        config = load_config(CONFIG_PATH)
        snap = snapshot_config(config)
        snap["model"]["provider"] = "mutated"
        # Original config should be unaffected (and is frozen anyway)
        assert config.model.provider == "anthropic"

    def test_snapshot_preserves_values(self):
        config = load_config(CONFIG_PATH)
        snap = snapshot_config(config)
        assert snap["model"]["name"] == "claude-sonnet-4-20250514"
        assert snap["evaluation"]["runs_per_combination"] == 10
        assert snap["scoring"]["hallucination_tolerance"] == 0.05


# ===========================================================================
# Validation: malformed YAML
# ===========================================================================


class TestMalformedConfig:
    """Test that load_config handles malformed YAML gracefully."""

    def test_empty_yaml_raises(self, tmp_path):
        """An empty YAML file should raise ValueError."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        with pytest.raises((ValueError, TypeError)):
            load_config(str(config_file))

    def test_missing_section_raises(self, tmp_path):
        """YAML missing a required section should raise KeyError."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml.dump({"model": {"provider": "test", "name": "x", "max_tokens": 512}}))
        with pytest.raises(KeyError, match="Missing required"):
            load_config(str(config_file))
