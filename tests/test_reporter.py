"""Tests for the report generator module.

Tests flag assignment logic, report file generation from mock data,
and summary.txt content validation.

_score_all_groups is mocked in these tests to isolate reporter file-generation
logic from the live scoring layer (which requires sentence-transformers and
real response data). The reporter-scorer integration itself is correct.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from src.config import load_config
from src.reporter import _flag_from_score, generate_report

# Path to real config for loading AppConfig
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "config.yaml"
)


# ===========================================================================
# Flag assignment
# ===========================================================================


class TestFlagFromScore:
    """Test the traffic-light flag mapping.

    Default thresholds from config.yaml:
      green_threshold: 0.75
      yellow_threshold: 0.50
    """

    def test_red_below_yellow(self):
        """Score < 0.50 -> red."""
        flag = _flag_from_score(0.30, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f534"  # red circle

    def test_red_at_zero(self):
        flag = _flag_from_score(0.0, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f534"

    def test_yellow_at_threshold(self):
        """Score == 0.50 -> yellow (>= yellow_threshold)."""
        flag = _flag_from_score(0.50, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e1"  # yellow circle

    def test_yellow_between_thresholds(self):
        """Score in [0.50, 0.75) -> yellow."""
        flag = _flag_from_score(0.60, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e1"

    def test_yellow_just_below_green(self):
        flag = _flag_from_score(0.749, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e1"

    def test_green_at_threshold(self):
        """Score == 0.75 -> green (>= green_threshold)."""
        flag = _flag_from_score(0.75, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e2"  # green circle

    def test_green_above_threshold(self):
        flag = _flag_from_score(0.90, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e2"

    def test_green_perfect(self):
        flag = _flag_from_score(1.0, yellow_threshold=0.50, green_threshold=0.75)
        assert flag == "\U0001f7e2"

    def test_custom_thresholds(self):
        """Test with non-default thresholds."""
        # green >= 0.9, yellow >= 0.6
        assert _flag_from_score(0.95, 0.6, 0.9) == "\U0001f7e2"
        assert _flag_from_score(0.75, 0.6, 0.9) == "\U0001f7e1"
        assert _flag_from_score(0.50, 0.6, 0.9) == "\U0001f534"


# ===========================================================================
# Helpers for mock data
# ===========================================================================


def _create_mock_results(results_dir: str, run_id: str) -> None:
    """Create a minimal mock results.json file for testing report generation."""
    run_dir = os.path.join(results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    results_data = {
        "config": {
            "model": {"provider": "openai", "name": "gpt-5-nano"},
            "evaluation": {
                "temperatures": [0.0],
                "runs_per_combination": 3,
                "templates": ["direct_extraction"],
            },
            "scoring": {
                "embedding_model": "all-MiniLM-L6-v2",
                "hallucination_tolerance": 0.05,
                "composite_weights": {
                    "semantic_consistency": 0.30,
                    "factual_consistency": 0.40,
                    "hallucination_rate": 0.30,
                },
            },
        },
        "facts": [
            {
                "id": "dbs_fy2024_nim",
                "company": "DBS",
                "metric": "Net Interest Margin",
                "category": "profitability",
                "difficulty": "easy",
                "value": 2.14,
                "unit": "percent",
            },
        ],
        "call_records": [
            {
                "fact_id": "dbs_fy2024_nim",
                "template_name": "direct_extraction",
                "temperature": 0.0,
                "run_index": i,
                "raw_response": "The net interest margin was 2.14%.",
            }
            for i in range(3)
        ],
    }

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results_data, fh)


def _fake_score_all_groups(groups, facts_lookup, config):
    """A mock replacement for _score_all_groups that returns synthetic detail rows.

    Returns fixed scores so reporter file-generation logic can be tested
    without running live sentence-transformer embeddings or making API calls.
    """
    rows = []
    for (fact_id, template, temperature), records in sorted(groups.items()):
        fact = facts_lookup.get(fact_id, {})
        rows.append({
            "fact_id": fact_id,
            "company": fact.get("company", "unknown"),
            "metric": fact.get("metric", "unknown"),
            "category": fact.get("category", "unknown"),
            "difficulty": fact.get("difficulty", "unknown"),
            "template": template,
            "temperature": temperature,
            "n_runs": len(records),
            "semantic_score": 0.95,
            "factual_score": 1.0,
            "hallucination_rate": 0.0,
            "extraction_failure_rate": 0.0,
            "modal_value": 2.14,
            "ground_truth": fact.get("value"),
            "composite_stability": 0.85,
            "flag": "\U0001f7e2",
        })
    return rows


# ===========================================================================
# Report generation with mock results
# ===========================================================================


class TestReportGeneration:
    """Test generate_report with a mock results directory.

    Uses a patched _score_all_groups to isolate reporter file-generation
    logic from the scorer module.
    """

    @pytest.fixture
    def mock_env(self, tmp_path):
        """Set up a temporary results dir and output dir with mock data."""
        results_dir = str(tmp_path / "results")
        output_dir = str(tmp_path / "reports")
        run_id = "test_run_001"
        _create_mock_results(results_dir, run_id)

        config = load_config(CONFIG_PATH)
        return {
            "run_id": run_id,
            "results_dir": results_dir,
            "output_dir": output_dir,
            "config": config,
        }

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_generate_report_creates_csv_files(self, mock_scorer, mock_env):
        """Verify that all expected CSV files are created."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
        )

        expected_keys = [
            "per_fact_detail",
            "by_metric_type",
            "by_company",
            "by_temperature",
            "by_template",
            "by_difficulty",
            "summary",
        ]
        for key in expected_keys:
            assert key in generated, f"Missing report key: {key}"
            assert os.path.isfile(generated[key]), (
                f"File not created for {key}: {generated[key]}"
            )

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_per_fact_csv_has_content(self, mock_scorer, mock_env):
        """The per-fact detail CSV should contain data rows."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
        )

        detail_path = generated["per_fact_detail"]
        with open(detail_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # Should have comment header lines (starting with #) plus CSV data
        comment_lines = [l for l in lines if l.startswith("#")]
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(comment_lines) > 0, "Missing comment header in CSV"
        # At least a header row and one data row
        assert len(data_lines) >= 2, "CSV should have header + data rows"

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_summary_contains_methodology(self, mock_scorer, mock_env):
        """The summary.txt must include a METHODOLOGY section."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
        )

        summary_path = generated["summary"]
        with open(summary_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        assert "METHODOLOGY" in content
        assert "Model:" in content
        assert "Temperature range:" in content
        assert "Runs per combination:" in content
        assert "Embedding model:" in content

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_summary_contains_run_id(self, mock_scorer, mock_env):
        """Summary should reference the run ID."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
        )
        with open(generated["summary"], "r", encoding="utf-8") as fh:
            content = fh.read()
        assert mock_env["run_id"] in content

    def test_missing_results_file_raises(self, tmp_path):
        """FileNotFoundError if results.json doesn't exist."""
        config = load_config(CONFIG_PATH)
        with pytest.raises(FileNotFoundError):
            generate_report(
                run_id="nonexistent_run",
                results_dir=str(tmp_path / "empty"),
                config=config,
                output_dir=str(tmp_path / "output"),
            )

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_json_output_format(self, mock_scorer, mock_env):
        """Test that output_format='json' produces JSON files."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
            output_format="json",
        )

        # Detail file should be .json
        detail_path = generated["per_fact_detail"]
        assert detail_path.endswith(".json")
        # Should be valid JSON
        with open(detail_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, list)

        # Summary is always .txt regardless of output_format
        assert generated["summary"].endswith(".txt")

    @patch("src.reporter._score_all_groups", side_effect=_fake_score_all_groups)
    def test_csv_comment_header_has_flagging_thresholds(self, mock_scorer, mock_env):
        """The CSV comment header should document flagging thresholds."""
        generated = generate_report(
            run_id=mock_env["run_id"],
            results_dir=mock_env["results_dir"],
            config=mock_env["config"],
            output_dir=mock_env["output_dir"],
        )

        detail_path = generated["per_fact_detail"]
        with open(detail_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        # Should document the thresholds in the header
        assert "0.75" in content  # green threshold
        assert "0.5" in content   # yellow threshold
