"""Integration tests for the LLM Financial Stability Bench.

Exercises the full pipeline end-to-end in dry-run mode (zero API calls)
and verifies the expected output structure. Also tests the report
generation path with synthetic results data, and the model comparison
module.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.comparison import (
    ComparisonReport,
    RunSummary,
    compare_runs,
    format_comparison_report,
)
from src.config import (
    AppConfig,
    CheckpointConfig,
    load_config,
)
from src.engine import EvaluationEngine
from src.reporter import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FACTS_PATH = PROJECT_ROOT / "ground_truth" / "facts.json"


def _load_facts() -> list[dict]:
    with open(FACTS_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["facts"]


def _make_quick_config(tmp_path: Path) -> AppConfig:
    """Load real config in quick mode with checkpoint dir pointing to tmp."""
    from dataclasses import replace

    config = load_config(str(CONFIG_PATH), quick=True)
    return replace(
        config,
        checkpoint=CheckpointConfig(
            save_interval=50,
            directory=str(tmp_path / "results"),
        ),
        cost=replace(config.cost, confirmation_threshold=999.0),
    )


# ===========================================================================
# End-to-end dry-run test
# ===========================================================================


class TestDryRunEndToEnd:
    """Test that the full pipeline works in dry-run mode without API calls."""

    def test_dry_run_quick_mode_returns_run_id(self, tmp_path):
        """Dry-run in quick mode should produce a run_id and make zero API calls."""
        config = _make_quick_config(tmp_path)
        facts = _load_facts()

        engine = EvaluationEngine(config, facts)
        run_id = engine.run(dry_run=True)

        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_dry_run_does_not_create_results(self, tmp_path):
        """Dry-run should NOT create a results.json file."""
        config = _make_quick_config(tmp_path)
        facts = _load_facts()

        engine = EvaluationEngine(config, facts)
        run_id = engine.run(dry_run=True)

        results_file = Path(config.checkpoint.directory) / run_id / "results.json"
        assert not results_file.exists()

    def test_dry_run_cost_estimate_positive(self, tmp_path):
        """Cost estimate for a non-trivial run should be positive."""
        config = _make_quick_config(tmp_path)
        facts = _load_facts()

        engine = EvaluationEngine(config, facts)
        cost = engine.estimate_cost()

        assert cost > 0.0

    def test_dry_run_total_calls_matches_quick_mode(self, tmp_path):
        """Total calls should reflect quick mode parameters.

        Quick mode uses 1 template (direct_extraction), 1 temperature,
        3 runs per combination. The max_facts capping is done in
        evaluate.py CLI, not in the engine, so the engine sees all facts.
        Total = n_valid_facts * 1 template * 1 temperature * 3 runs.
        """
        config = _make_quick_config(tmp_path)
        facts = _load_facts()

        engine = EvaluationEngine(config, facts)
        total = engine.estimate_total_calls()

        # All facts should be valid for direct_extraction (needs company,
        # metric, period). With 29 facts: 29 * 1 * 1 * 3 = 87.
        # Allow some slack for facts missing required fields.
        n_facts = len(facts)
        max_possible = n_facts * 1 * 1 * 3
        assert 0 < total <= max_possible


# ===========================================================================
# Report generation with synthetic data
# ===========================================================================


class TestReportGenerationEndToEnd:
    """Test report generation from synthetic results data."""

    def test_report_from_synthetic_run(self, tmp_path):
        """Generate reports from a synthetic results.json and verify outputs."""
        config = _make_quick_config(tmp_path)
        run_id = "integration_test_run"

        # Build a synthetic results.json
        results_dir = tmp_path / "results"
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True)

        facts = _load_facts()[:3]
        call_records = []
        for fact in facts:
            for run_idx in range(3):
                call_records.append(
                    {
                        "fact_id": fact["id"],
                        "template_name": "direct_extraction",
                        "temperature": 0.0,
                        "run_index": run_idx,
                        "raw_response": f"The value is {fact['value']}%.",
                        "extracted_value": fact["value"],
                        "extracted_unit": fact.get("unit", "percent"),
                        "latency_ms": 100.0,
                        "input_tokens": 50,
                        "output_tokens": 20,
                        "finish_reason": "end_turn",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                    }
                )

        results_data = {
            "run_id": run_id,
            "config": {
                "model": {"provider": "test", "name": "test-model"},
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
                "flagging": {
                    "green_threshold": 0.75,
                    "yellow_threshold": 0.50,
                },
            },
            "facts": facts,
            "n_records": len(call_records),
            "call_records": call_records,
        }

        with open(run_dir / "results.json", "w") as fh:
            json.dump(results_data, fh)

        output_dir = tmp_path / "reports"
        generated = generate_report(
            run_id=run_id,
            results_dir=str(results_dir),
            config=config,
            output_dir=str(output_dir),
            output_format="csv",
        )

        # Verify expected report files were generated
        assert "per_fact_detail" in generated
        assert "summary" in generated
        assert "by_company" in generated
        assert "by_temperature" in generated
        assert "by_template" in generated

        # Verify files exist on disk
        for name, path in generated.items():
            assert Path(path).exists(), f"Report file missing: {name} -> {path}"

        # Verify summary.txt has content
        summary_path = generated["summary"]
        summary_text = Path(summary_path).read_text()
        assert "Evaluation Summary" in summary_text
        assert run_id in summary_text


# ===========================================================================
# Model comparison integration
# ===========================================================================


class TestComparisonIntegration:
    """Test the model comparison module end-to-end."""

    def test_compare_two_models(self):
        """Compare two synthetic model summaries and verify report structure."""
        summary_a = RunSummary(
            label="Model A",
            model_name="model-a",
            composite_stability=0.75,
            semantic_consistency=0.90,
            factual_consistency=0.70,
            hallucination_rate=0.30,
            extraction_failure_rate=0.10,
            n_groups=100,
            n_green=50,
            n_yellow=30,
            n_red=20,
        )
        summary_b = RunSummary(
            label="Model B",
            model_name="model-b",
            composite_stability=0.60,
            semantic_consistency=0.85,
            factual_consistency=0.55,
            hallucination_rate=0.50,
            extraction_failure_rate=0.15,
            n_groups=100,
            n_green=30,
            n_yellow=25,
            n_red=45,
        )

        report = compare_runs(summary_a, summary_b)

        assert isinstance(report, ComparisonReport)
        assert report.overall_winner == "Model A"
        assert report.wins_a > report.wins_b
        assert len(report.metric_comparisons) == 5
        assert report.recommendation  # non-empty

        # Format should produce a readable string
        text = format_comparison_report(report)
        assert "Model Comparison Report" in text
        assert "Model A" in text
        assert "Model B" in text
        assert "RECOMMENDATION" in text

    def test_compare_tied_models(self):
        """Two identical summaries should produce a tie."""
        summary = RunSummary(
            label="Same Model",
            model_name="model-x",
            composite_stability=0.70,
            semantic_consistency=0.85,
            factual_consistency=0.65,
            hallucination_rate=0.40,
            extraction_failure_rate=0.12,
            n_groups=50,
        )

        report = compare_runs(summary, summary, label_a="Run 1", label_b="Run 2")

        assert report.overall_winner == "tie"
        assert report.wins_a == 0
        assert report.wins_b == 0
        assert report.ties == 5
