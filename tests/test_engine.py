"""Tests for the evaluation engine.

Covers checkpoint save/load, deduplication key format, dry-run mode,
resume skip behaviour, and cost estimation. All tests mock the adapter
so no real API calls are made.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.adapters.base_adapter import LLMResponse
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
)
from src.engine import (
    CallRecord,
    EvaluationEngine,
    _deserialize_records,
    _make_completed_key,
    _serialize_records,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_test_config(tmp_path: Path) -> AppConfig:
    """Build a minimal valid AppConfig for engine tests."""
    return AppConfig(
        model=ModelConfig(provider="openai", name="test-model", max_tokens=512),
        evaluation=EvaluationConfig(
            temperatures=[0.0],
            runs_per_combination=1,
            templates=["direct_extraction"],
        ),
        quick_mode=QuickModeConfig(
            temperatures=[0.0],
            runs_per_combination=1,
            max_facts=3,
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
        checkpoint=CheckpointConfig(
            save_interval=50,
            directory=str(tmp_path / "results"),
        ),
        cost=CostConfig(
            avg_input_tokens=200,
            avg_output_tokens=300,
            price_per_1k_input=0.003,
            price_per_1k_output=0.015,
            confirmation_threshold=100.0,  # high threshold to avoid prompt
        ),
        api=ApiConfig(
            timeout_seconds=30,
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=60.0,
            rate_limit_rpm=50,
        ),
    )


def _make_test_facts() -> list[dict]:
    """Build minimal facts for engine tests."""
    return [
        {
            "id": "test_fact_1",
            "company": "TestCo",
            "metric": "Net Interest Margin",
            "period": "FY2024",
            "value": 2.14,
            "unit": "percent",
            "category": "profitability",
            "difficulty": "easy",
        },
    ]


def _make_mock_response(text: str = "2.14%") -> LLMResponse:
    """Build a mock LLMResponse."""
    return LLMResponse(
        text=text,
        model="test-model",
        input_tokens=50,
        output_tokens=20,
        finish_reason="end_turn",
        latency_ms=100.0,
    )


# ===========================================================================
# Deduplication key format
# ===========================================================================


class TestMakeCompletedKey:
    """Test that the deduplication key has the correct format."""

    def test_key_format(self):
        key = _make_completed_key("fact_1", "direct_extraction", 0.3, 2)
        assert key == "fact_1|direct_extraction|0.3|2"

    def test_key_format_zero_temp(self):
        key = _make_completed_key("fact_1", "comparative", 0.0, 0)
        assert key == "fact_1|comparative|0.0|0"

    def test_key_uses_pipe_delimiter(self):
        """Key must use pipe delimiters: fact_id|template|temp|run."""
        key = _make_completed_key("abc", "tpl", 1.0, 5)
        parts = key.split("|")
        assert len(parts) == 4
        assert parts[0] == "abc"
        assert parts[1] == "tpl"
        assert parts[2] == "1.0"
        assert parts[3] == "5"

    def test_keys_are_unique_for_different_inputs(self):
        key_a = _make_completed_key("f1", "t1", 0.0, 0)
        key_b = _make_completed_key("f1", "t1", 0.0, 1)
        key_c = _make_completed_key("f1", "t1", 0.3, 0)
        key_d = _make_completed_key("f2", "t1", 0.0, 0)
        assert len({key_a, key_b, key_c, key_d}) == 4


# ===========================================================================
# CallRecord serialization round-trip
# ===========================================================================


class TestRecordSerialization:
    """Test that CallRecords survive JSON round-trip."""

    def test_serialize_deserialize_round_trip(self):
        records = [
            CallRecord(
                fact_id="f1",
                template_name="direct_extraction",
                temperature=0.0,
                run_index=0,
                raw_response="The value is 2.14%.",
                extracted_value=2.14,
                extracted_unit="percent",
                latency_ms=150.0,
                input_tokens=100,
                output_tokens=30,
                finish_reason="end_turn",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        ]
        serialized = _serialize_records(records)
        deserialized = _deserialize_records(serialized)
        assert len(deserialized) == 1
        assert deserialized[0].fact_id == "f1"
        assert deserialized[0].extracted_value == 2.14
        assert deserialized[0].extracted_unit == "percent"

    def test_serialize_none_extracted_value(self):
        records = [
            CallRecord(
                fact_id="f1",
                template_name="t",
                temperature=0.0,
                run_index=0,
                raw_response="no number here",
                extracted_value=None,
                extracted_unit=None,
                latency_ms=50.0,
                input_tokens=10,
                output_tokens=5,
                finish_reason="end_turn",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        ]
        serialized = _serialize_records(records)
        deserialized = _deserialize_records(serialized)
        assert deserialized[0].extracted_value is None
        assert deserialized[0].extracted_unit is None


# ===========================================================================
# Checkpoint save and load
# ===========================================================================


class TestCheckpointing:
    """Test checkpoint save/load using real file system via tmp_path."""

    def test_save_and_load_checkpoint(self, tmp_path):
        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())
        run_id = "test_run_checkpoint"

        records = [
            CallRecord(
                fact_id="f1",
                template_name="direct_extraction",
                temperature=0.0,
                run_index=0,
                raw_response="2.14%",
                extracted_value=2.14,
                extracted_unit="percent",
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=20,
                finish_reason="end_turn",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        ]
        completed_keys = {"f1|direct_extraction|0.0|0"}

        # Save checkpoint
        engine._save_checkpoint(run_id, records, completed_keys)

        # Verify file exists
        checkpoint_path = Path(config.checkpoint.directory) / run_id / "checkpoint.json"
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_records, loaded_keys = engine._load_checkpoint(run_id)
        assert len(loaded_records) == 1
        assert loaded_records[0].fact_id == "f1"
        assert loaded_records[0].extracted_value == 2.14
        assert loaded_keys == completed_keys

    def test_load_checkpoint_nonexistent_returns_empty(self, tmp_path):
        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())

        records, keys = engine._load_checkpoint("nonexistent_run_id")
        assert records == []
        assert keys == set()

    def test_load_corrupt_checkpoint_returns_empty(self, tmp_path):
        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())
        run_id = "corrupt_run"

        # Write corrupt JSON
        run_dir = Path(config.checkpoint.directory) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoint.json").write_text("{ invalid json !!!")

        records, keys = engine._load_checkpoint(run_id)
        assert records == []
        assert keys == set()

    def test_checkpoint_data_structure(self, tmp_path):
        """Verify the on-disk checkpoint has expected fields."""
        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())
        run_id = "test_structure"

        records = [
            CallRecord(
                fact_id="f1", template_name="t", temperature=0.0,
                run_index=0, raw_response="x", extracted_value=1.0,
                extracted_unit="percent", latency_ms=10.0,
                input_tokens=5, output_tokens=3,
                finish_reason="end_turn",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        ]
        engine._save_checkpoint(run_id, records, {"f1|t|0.0|0"})

        checkpoint_path = Path(config.checkpoint.directory) / run_id / "checkpoint.json"
        with open(checkpoint_path) as fh:
            data = json.load(fh)

        assert "run_id" in data
        assert data["run_id"] == run_id
        assert "n_completed" in data
        assert data["n_completed"] == 1
        assert "completed_keys" in data
        assert "records" in data


# ===========================================================================
# Dry run mode
# ===========================================================================


class TestDryRun:
    """Test that dry-run mode makes no API calls."""

    @patch("src.engine.generate_all_prompts")
    def test_dry_run_returns_run_id(self, mock_prompts, tmp_path):
        mock_prompts.return_value = [
            {
                "fact_id": "f1",
                "template_name": "direct_extraction",
                "rendered_prompt": "What was TestCo's NIM?",
                "fact": _make_test_facts()[0],
            },
        ]

        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())

        run_id = engine.run(dry_run=True)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    @patch("src.engine.generate_all_prompts")
    def test_dry_run_does_not_call_adapter(self, mock_prompts, tmp_path):
        mock_prompts.return_value = [
            {
                "fact_id": "f1",
                "template_name": "direct_extraction",
                "rendered_prompt": "What was TestCo's NIM?",
                "fact": _make_test_facts()[0],
            },
        ]

        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())

        # Mock the adapter to detect any calls
        mock_adapter = MagicMock()
        engine._adapter = mock_adapter

        engine.run(dry_run=True)
        mock_adapter.generate.assert_not_called()

    @patch("src.engine.generate_all_prompts")
    def test_dry_run_does_not_create_results_file(self, mock_prompts, tmp_path):
        mock_prompts.return_value = [
            {
                "fact_id": "f1",
                "template_name": "direct_extraction",
                "rendered_prompt": "What was TestCo's NIM?",
                "fact": _make_test_facts()[0],
            },
        ]

        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())

        run_id = engine.run(dry_run=True)

        # No results.json should be written in dry-run
        results_path = Path(config.checkpoint.directory) / run_id / "results.json"
        assert not results_path.exists()


# ===========================================================================
# Resume: skip already-completed combinations
# ===========================================================================


class TestResume:
    """Test that the engine skips already-completed combinations on resume."""

    @patch("src.engine.extract_numeric")
    @patch("src.engine.generate_all_prompts")
    def test_resume_skips_completed(self, mock_prompts, mock_extract, tmp_path):
        """Engine should skip calls whose key is already in the checkpoint."""
        facts = _make_test_facts()
        config = _make_test_config(tmp_path)
        run_id = "resume_test_run"

        mock_prompts.return_value = [
            {
                "fact_id": "test_fact_1",
                "template_name": "direct_extraction",
                "rendered_prompt": "What was TestCo's NIM?",
                "fact": facts[0],
            },
        ]

        # Pre-populate a checkpoint with the one combination already done
        engine = EvaluationEngine(config, facts)
        existing_records = [
            CallRecord(
                fact_id="test_fact_1",
                template_name="direct_extraction",
                temperature=0.0,
                run_index=0,
                raw_response="2.14%",
                extracted_value=2.14,
                extracted_unit="percent",
                latency_ms=100.0,
                input_tokens=50,
                output_tokens=20,
                finish_reason="end_turn",
                timestamp="2026-01-01T00:00:00+00:00",
            ),
        ]
        completed_keys = {"test_fact_1|direct_extraction|0.0|0"}
        engine._save_checkpoint(run_id, existing_records, completed_keys)

        # Mock adapter
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = _make_mock_response()
        engine._adapter = mock_adapter

        # Resume the run
        result_run_id = engine.run(dry_run=False, resume_run_id=run_id)
        assert result_run_id == run_id

        # Adapter should NOT have been called because the only combination
        # was already in the checkpoint
        mock_adapter.generate.assert_not_called()


# ===========================================================================
# Cost estimation
# ===========================================================================


class TestCostEstimation:
    """Test cost estimation calculation."""

    @patch("src.engine.generate_all_prompts")
    def test_estimate_cost_formula(self, mock_prompts, tmp_path):
        """Cost = total_calls * (input_cost + output_cost) per call."""
        mock_prompts.return_value = [
            {"fact_id": "f1", "template_name": "t", "rendered_prompt": "p", "fact": {}},
        ]

        config = _make_test_config(tmp_path)
        # 1 prompt * 1 temperature * 1 run = 1 call
        engine = EvaluationEngine(config, _make_test_facts())

        cost = engine.estimate_cost()

        # input_cost per call = (200 / 1000) * 0.003 = 0.0006
        # output_cost per call = (300 / 1000) * 0.015 = 0.0045
        # total = 1 * (0.0006 + 0.0045) = 0.0051
        expected = 1 * ((200 / 1000) * 0.003 + (300 / 1000) * 0.015)
        assert abs(cost - expected) < 1e-9

    @patch("src.engine.generate_all_prompts")
    def test_estimate_total_calls(self, mock_prompts, tmp_path):
        mock_prompts.return_value = [
            {"fact_id": "f1", "template_name": "t1", "rendered_prompt": "p", "fact": {}},
            {"fact_id": "f1", "template_name": "t2", "rendered_prompt": "p", "fact": {}},
        ]

        config = _make_test_config(tmp_path)
        # 2 prompts * 1 temperature * 1 run = 2 calls
        engine = EvaluationEngine(config, _make_test_facts())
        total = engine.estimate_total_calls()
        assert total == 2

    @patch("src.engine.generate_all_prompts")
    def test_estimate_cost_scales_with_calls(self, mock_prompts, tmp_path):
        mock_prompts.return_value = [
            {"fact_id": f"f{i}", "template_name": "t", "rendered_prompt": "p", "fact": {}}
            for i in range(10)
        ]

        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())
        cost = engine.estimate_cost()

        # 10 prompts * 1 temp * 1 run = 10 calls
        per_call = (200 / 1000) * 0.003 + (300 / 1000) * 0.015
        expected = 10 * per_call
        assert abs(cost - expected) < 1e-9


# ===========================================================================
# Full run with mock adapter
# ===========================================================================


class TestFullRunMocked:
    """Test a complete engine run with a mocked adapter."""

    @patch("src.engine.extract_numeric")
    @patch("src.engine.generate_all_prompts")
    def test_full_run_creates_results_file(self, mock_prompts, mock_extract, tmp_path):
        facts = _make_test_facts()
        config = _make_test_config(tmp_path)

        mock_prompts.return_value = [
            {
                "fact_id": "test_fact_1",
                "template_name": "direct_extraction",
                "rendered_prompt": "What was TestCo's NIM?",
                "fact": facts[0],
            },
        ]

        mock_extract.return_value = MagicMock(value=2.14, unit="percent")

        engine = EvaluationEngine(config, facts)
        mock_adapter = MagicMock()
        mock_adapter.generate.return_value = _make_mock_response()
        engine._adapter = mock_adapter

        run_id = engine.run(dry_run=False)

        # Results file should exist
        results_path = Path(config.checkpoint.directory) / run_id / "results.json"
        assert results_path.exists()

        # Adapter should have been called exactly once
        assert mock_adapter.generate.call_count == 1

        # Load and verify results
        with open(results_path) as fh:
            data = json.load(fh)
        assert data["n_records"] == 1
        assert len(data["call_records"]) == 1
        assert data["call_records"][0]["fact_id"] == "test_fact_1"

    @patch("src.engine.extract_numeric")
    @patch("src.engine.generate_all_prompts")
    def test_run_with_no_prompts_raises(self, mock_prompts, mock_extract, tmp_path):
        """Engine should raise ValueError when no valid prompts are generated."""
        mock_prompts.return_value = []

        config = _make_test_config(tmp_path)
        engine = EvaluationEngine(config, _make_test_facts())

        with pytest.raises(ValueError, match="No valid prompts"):
            engine.run(dry_run=False)
