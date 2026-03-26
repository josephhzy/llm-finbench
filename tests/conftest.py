"""Shared pytest fixtures for LLM Financial Stability Bench tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import load_config

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
