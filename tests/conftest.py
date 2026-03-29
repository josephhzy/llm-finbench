"""Shared pytest fixtures for LLM Financial Stability Bench tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FACTS_PATH = PROJECT_ROOT / "ground_truth" / "facts.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    """Load the real config.yaml into an AppConfig object."""
    return load_config(str(CONFIG_PATH))


@pytest.fixture
def sample_facts():
    """Load the full facts list from ground_truth/facts.json."""
    with open(FACTS_PATH, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["facts"]


@pytest.fixture
def sample_fact(sample_facts):
    """Return the first fact from facts.json for single-fact tests."""
    return sample_facts[0]


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal valid AppConfig for testing without loading config.yaml.

    Uses tmp_path for the checkpoint directory so tests do not write to the
    real results/ directory.
    """
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
            confirmation_threshold=100.0,
        ),
        api=ApiConfig(
            timeout_seconds=30,
            max_retries=3,
            base_backoff_seconds=1.0,
            max_backoff_seconds=60.0,
            rate_limit_rpm=50,
        ),
    )


@pytest.fixture
def minimal_facts():
    """Create a small list of sample facts for testing.

    Contains facts with varying difficulty levels and companies to support
    aggregation tests.
    """
    return [
        {
            "id": "test_fact_easy",
            "company": "TestCo",
            "metric": "Net Interest Margin",
            "metric_abbreviation": "NIM",
            "period": "FY2024",
            "value": 2.14,
            "unit": "percent",
            "category": "profitability",
            "difficulty": "easy",
            "context": "NIM was 2.14%.",
        },
        {
            "id": "test_fact_medium",
            "company": "TestCo",
            "metric": "Return on Equity",
            "metric_abbreviation": "ROE",
            "period": "FY2024",
            "value": 15.0,
            "unit": "percent",
            "category": "profitability",
            "difficulty": "medium",
            "context": "ROE was 15.0%.",
        },
        {
            "id": "test_fact_hard",
            "company": "AnotherCo",
            "metric": "CET1 Ratio",
            "metric_abbreviation": "CET1",
            "period": "FY2024",
            "value": 14.8,
            "unit": "percent",
            "category": "capital",
            "difficulty": "hard",
            "context": "CET1 ratio was 14.8%.",
        },
    ]


@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary directory for test outputs.

    Returns a Path object pointing to a fresh temp directory.
    """
    out = tmp_path / "test_output"
    out.mkdir()
    return out
