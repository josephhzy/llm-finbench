"""Tests for the scoring engine.

Uses synthetic data to verify scoring logic without requiring real API calls
or sentence-transformers (except for semantic consistency, which is skipped
if sentence-transformers is not installed).
"""

from __future__ import annotations

import pytest

from unittest.mock import patch

from src.scorer import (
    compute_composite_stability,
    compute_factual_consistency,
    compute_hallucination_rate,
    compute_semantic_consistency,
    score_fact,
)

# Check whether sentence_transformers is available for semantic consistency tests
try:
    import sentence_transformers  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ===========================================================================
# Factual consistency
# ===========================================================================


class TestFactualConsistency:
    """Test compute_factual_consistency with synthetic extracted values."""

    def test_ten_identical_values_gives_one(self):
        """10 identical values -> consistency = 1.0."""
        values = [2.14] * 10
        score, modal, fail_rate = compute_factual_consistency(values)
        assert score == pytest.approx(1.0)
        assert modal == pytest.approx(2.14)
        assert fail_rate == pytest.approx(0.0)

    def test_five_and_five_gives_half(self):
        """5 of value A and 5 of value B -> consistency = 0.5."""
        values = [2.14] * 5 + [3.41] * 5
        score, modal, fail_rate = compute_factual_consistency(values)
        assert score == pytest.approx(0.5)
        assert fail_rate == pytest.approx(0.0)
        # Modal should be one of the two values
        assert modal in (2.14, 3.41)

    def test_dominant_mode(self):
        """7 of one value, 3 of another -> consistency = 7/10 = 0.7."""
        values = [2.14] * 7 + [3.41] * 3
        score, modal, fail_rate = compute_factual_consistency(values)
        assert score == pytest.approx(0.7)
        assert modal == pytest.approx(2.14)

    def test_all_none_returns_zero(self):
        """All None -> (0.0, None, 1.0)."""
        values = [None] * 10
        score, modal, fail_rate = compute_factual_consistency(values)
        assert score == pytest.approx(0.0)
        assert modal is None
        assert fail_rate == pytest.approx(1.0)

    def test_empty_list_returns_zero(self):
        """Empty list -> (0.0, None, 1.0)."""
        score, modal, fail_rate = compute_factual_consistency([])
        assert score == pytest.approx(0.0)
        assert modal is None
        assert fail_rate == pytest.approx(1.0)

    def test_mixed_none_and_values(self):
        """Some None, some values -> fail rate reflects None fraction."""
        values = [2.14, 2.14, 2.14, None, None]
        score, modal, fail_rate = compute_factual_consistency(values)
        # Consistency among valid: 3/3 = 1.0
        assert score == pytest.approx(1.0)
        assert modal == pytest.approx(2.14)
        # 2 out of 5 are None
        assert fail_rate == pytest.approx(0.4)

    def test_single_value(self):
        """Single value -> consistency = 1.0."""
        score, modal, fail_rate = compute_factual_consistency([42.0])
        assert score == pytest.approx(1.0)
        assert modal == pytest.approx(42.0)
        assert fail_rate == pytest.approx(0.0)

    def test_floating_point_noise(self):
        """Near-identical floats (rounding noise) should be treated as same."""
        # These differ at the 10th decimal — should round to same value
        values = [2.14, 2.1400000001, 2.14, 2.14]
        score, modal, fail_rate = compute_factual_consistency(values)
        assert score == pytest.approx(1.0)

    def test_three_distinct_values(self):
        """Three modes with different counts."""
        values = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        score, modal, fail_rate = compute_factual_consistency(values)
        # Mode is 1.0 with count 3, total valid 6 -> 3/6 = 0.5
        assert score == pytest.approx(0.5)
        assert modal == pytest.approx(1.0)


# ===========================================================================
# Hallucination rate
# ===========================================================================


class TestHallucinationRate:
    """Test compute_hallucination_rate with known ground truth."""

    def test_all_correct_gives_zero(self):
        """All values match ground truth exactly -> 0.0."""
        values = [2.14] * 10
        rate = compute_hallucination_rate(values, ground_truth=2.14)
        assert rate == pytest.approx(0.0)

    def test_all_wrong_gives_one(self):
        """All values far from ground truth -> 1.0."""
        values = [99.99] * 10
        rate = compute_hallucination_rate(values, ground_truth=2.14)
        assert rate == pytest.approx(1.0)

    def test_within_tolerance_not_hallucination(self):
        """Value within 5% relative tolerance is NOT a hallucination."""
        # 2.14 * 1.05 = 2.247 -> 2.24 is within tolerance
        # 2.14 * 0.95 = 2.033 -> 2.04 is within tolerance
        values = [2.24]
        rate = compute_hallucination_rate(values, ground_truth=2.14, tolerance=0.05)
        assert rate == pytest.approx(0.0)

    def test_outside_tolerance_is_hallucination(self):
        """Value outside 5% relative tolerance IS a hallucination."""
        # 3.41 vs 2.14: relative error = |3.41-2.14|/2.14 = 0.593 >> 0.05
        values = [3.41]
        rate = compute_hallucination_rate(values, ground_truth=2.14, tolerance=0.05)
        assert rate == pytest.approx(1.0)

    def test_none_values_count_as_hallucinations(self):
        """None extractions count as hallucinations."""
        values = [None, None, 2.14, 2.14]
        rate = compute_hallucination_rate(values, ground_truth=2.14, tolerance=0.05)
        # 2 Nones out of 4 = 0.5
        assert rate == pytest.approx(0.5)

    def test_all_none_all_hallucinated(self):
        values = [None] * 5
        rate = compute_hallucination_rate(values, ground_truth=2.14)
        assert rate == pytest.approx(1.0)

    def test_empty_list_gives_zero(self):
        """Empty list -> 0.0 (no data = no hallucinations counted)."""
        rate = compute_hallucination_rate([], ground_truth=2.14)
        assert rate == pytest.approx(0.0)

    def test_ground_truth_zero_uses_absolute_tolerance(self):
        """When ground_truth=0, falls back to absolute tolerance."""
        # abs(0.0005) <= 0.001 -> not a hallucination
        values = [0.0005]
        rate = compute_hallucination_rate(values, ground_truth=0.0)
        assert rate == pytest.approx(0.0)

    def test_ground_truth_zero_large_deviation(self):
        """When ground_truth=0 and value is far from 0, it's a hallucination."""
        values = [5.0]
        rate = compute_hallucination_rate(values, ground_truth=0.0)
        assert rate == pytest.approx(1.0)

    def test_mixed_correct_and_wrong(self):
        """Mix of correct and wrong values."""
        # Ground truth = 10.0, tolerance = 0.05
        # 10.0 -> ok, 10.4 -> |0.4|/10 = 0.04 -> ok, 15.0 -> |5|/10 = 0.5 -> hallucination
        values = [10.0, 10.4, 15.0, None]
        rate = compute_hallucination_rate(values, ground_truth=10.0, tolerance=0.05)
        # 15.0 and None are hallucinations: 2/4 = 0.5
        assert rate == pytest.approx(0.5)

    def test_custom_tolerance(self):
        """Test with a wider tolerance."""
        # 2.14 vs 2.50: relative error = 0.36/2.14 = 0.168
        # With 20% tolerance -> NOT a hallucination
        values = [2.50]
        rate = compute_hallucination_rate(values, ground_truth=2.14, tolerance=0.20)
        assert rate == pytest.approx(0.0)


# ===========================================================================
# Composite stability
# ===========================================================================


class TestCompositeStability:
    """Test compute_composite_stability with known inputs."""

    @pytest.fixture
    def default_weights(self):
        return {
            "semantic_consistency": 0.30,
            "factual_consistency": 0.40,
            "hallucination_rate": 0.30,
        }

    def test_all_perfect_scores(self, default_weights):
        """Semantic=1.0, factual=1.0, hallucination=0.0 -> composite=1.0."""
        composite = compute_composite_stability(
            semantic=1.0,
            factual=1.0,
            hallucination=0.0,
            weights=default_weights,
        )
        assert composite == pytest.approx(1.0)

    def test_all_worst_scores(self, default_weights):
        """Semantic=0.0, factual=0.0, hallucination=1.0 -> composite=0.0."""
        composite = compute_composite_stability(
            semantic=0.0,
            factual=0.0,
            hallucination=1.0,
            weights=default_weights,
        )
        assert composite == pytest.approx(0.0)

    def test_known_values(self, default_weights):
        """Test with specific intermediate values."""
        # composite = 0.8*0.3 + 0.6*0.4 + (1-0.2)*0.3
        #           = 0.24 + 0.24 + 0.24 = 0.72
        composite = compute_composite_stability(
            semantic=0.8,
            factual=0.6,
            hallucination=0.2,
            weights=default_weights,
        )
        assert composite == pytest.approx(0.72)

    def test_hallucination_inverted(self, default_weights):
        """Verify that hallucination rate is inverted (1 - rate) in the formula."""
        # With hallucination=0.5, the contribution is (1-0.5)*0.3 = 0.15
        composite_low = compute_composite_stability(
            semantic=1.0, factual=1.0, hallucination=0.5, weights=default_weights
        )
        composite_high = compute_composite_stability(
            semantic=1.0, factual=1.0, hallucination=0.0, weights=default_weights
        )
        assert composite_high > composite_low

    def test_custom_weights(self):
        """Test with non-default weights."""
        weights = {
            "semantic_consistency": 0.50,
            "factual_consistency": 0.25,
            "hallucination_rate": 0.25,
        }
        # composite = 1.0*0.5 + 0.0*0.25 + (1-0.0)*0.25 = 0.5 + 0 + 0.25 = 0.75
        composite = compute_composite_stability(
            semantic=1.0,
            factual=0.0,
            hallucination=0.0,
            weights=weights,
        )
        assert composite == pytest.approx(0.75)

    def test_clamped_to_zero_one(self, default_weights):
        """Composite should never go below 0 or above 1."""
        # Both extremes should be clamped
        composite = compute_composite_stability(
            semantic=1.0, factual=1.0, hallucination=0.0, weights=default_weights
        )
        assert 0.0 <= composite <= 1.0

        composite = compute_composite_stability(
            semantic=0.0, factual=0.0, hallucination=1.0, weights=default_weights
        )
        assert 0.0 <= composite <= 1.0


# ===========================================================================
# Semantic consistency (requires sentence-transformers)
# ===========================================================================


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not installed",
)
class TestSemanticConsistency:
    """Test compute_semantic_consistency (needs sentence-transformers)."""

    def test_identical_responses_high_similarity(self):
        """10 identical strings should yield similarity very close to 1.0."""
        responses = ["The NIM was 2.14%."] * 10
        score = compute_semantic_consistency(responses, "all-MiniLM-L6-v2")
        assert score >= 0.99

    def test_single_response_returns_one(self):
        """Fewer than 2 responses -> 1.0 by convention."""
        score = compute_semantic_consistency(["Only one."], "all-MiniLM-L6-v2")
        assert score == pytest.approx(1.0)

    def test_empty_responses_returns_one(self):
        """Empty list -> 1.0 (nothing to compare)."""
        score = compute_semantic_consistency([], "all-MiniLM-L6-v2")
        assert score == pytest.approx(1.0)

    def test_dissimilar_responses_lower_score(self):
        """Very different responses should score lower than identical ones."""
        identical = ["The NIM was 2.14%."] * 5
        diverse = [
            "The NIM was 2.14%.",
            "I like pizza and chocolate.",
            "The weather is sunny today.",
            "Python is a programming language.",
            "Singapore is in Southeast Asia.",
        ]
        score_identical = compute_semantic_consistency(identical, "all-MiniLM-L6-v2")
        score_diverse = compute_semantic_consistency(diverse, "all-MiniLM-L6-v2")
        assert score_identical > score_diverse


# ===========================================================================
# score_fact integration (mocks sentence-transformers to avoid dependency)
# ===========================================================================


class TestScoreFact:
    """Test the score_fact() integrated pipeline with mocked semantic consistency."""

    @pytest.fixture
    def default_weights(self):
        return {
            "semantic_consistency": 0.30,
            "factual_consistency": 0.40,
            "hallucination_rate": 0.30,
        }

    @patch("src.scorer.compute_semantic_consistency", return_value=0.95)
    def test_score_fact_basic(self, mock_semantic, default_weights):
        """score_fact returns a FactScores with correct composite calculation."""
        responses = ["The NIM was 2.14%."] * 5
        result = score_fact(
            fact_id="dbs_fy2024_nim",
            template="direct_extraction",
            temperature=0.0,
            response_texts=responses,
            ground_truth_value=2.14,
            expected_unit="percent",
            embedding_model_name="all-MiniLM-L6-v2",
            hallucination_tolerance=0.05,
            composite_weights=default_weights,
        )
        assert result.fact_id == "dbs_fy2024_nim"
        assert result.template == "direct_extraction"
        assert result.temperature == 0.0
        assert result.n_runs == 5
        assert result.semantic_consistency == pytest.approx(0.95)
        # All extractions should produce 2.14
        assert result.factual_consistency == pytest.approx(1.0)
        assert result.hallucination_rate == pytest.approx(0.0)
        assert result.modal_value == pytest.approx(2.14)
        assert result.extraction_failure_rate == pytest.approx(0.0)
        assert result.ground_truth == pytest.approx(2.14)
        # composite = 0.95*0.3 + 1.0*0.4 + (1-0.0)*0.3 = 0.285 + 0.4 + 0.3 = 0.985
        assert result.composite_stability == pytest.approx(0.985)
        assert 0.0 <= result.composite_stability <= 1.0

    @patch("src.scorer.compute_semantic_consistency", return_value=1.0)
    def test_score_fact_all_hallucinated(self, mock_semantic, default_weights):
        """score_fact correctly flags all-wrong responses as hallucinations."""
        responses = ["The value was 99.99%."] * 3
        result = score_fact(
            fact_id="test_fact",
            template="direct_extraction",
            temperature=0.5,
            response_texts=responses,
            ground_truth_value=2.14,
            expected_unit="percent",
            embedding_model_name="all-MiniLM-L6-v2",
            hallucination_tolerance=0.05,
            composite_weights=default_weights,
        )
        assert result.hallucination_rate == pytest.approx(1.0)
        assert result.factual_consistency == pytest.approx(1.0)  # all same wrong value
        # composite = 1.0*0.3 + 1.0*0.4 + (1-1.0)*0.3 = 0.3 + 0.4 + 0.0 = 0.7
        assert result.composite_stability == pytest.approx(0.7)

    @patch("src.scorer.compute_semantic_consistency", return_value=0.8)
    def test_score_fact_preserves_raw_data(self, mock_semantic, default_weights):
        """score_fact stores response_texts and extracted_values for reproducibility."""
        responses = ["Revenue was 10.5 billion.", "Revenue was 10.5 billion."]
        result = score_fact(
            fact_id="test_fact",
            template="direct_extraction",
            temperature=0.0,
            response_texts=responses,
            ground_truth_value=10.5,
            expected_unit="billions",
            embedding_model_name="all-MiniLM-L6-v2",
            hallucination_tolerance=0.05,
            composite_weights=default_weights,
        )
        assert result.response_texts == responses
        assert len(result.extracted_values) == 2
