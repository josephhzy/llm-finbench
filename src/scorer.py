"""Scoring engine for LLM Financial Stability Bench.

Computes four independent metrics for each fact-template-temperature combination,
then combines them into a single composite stability score:

1. **Semantic consistency** — do repeated LLM responses say the same thing,
   regardless of whether the thing is correct?  Measured via pairwise cosine
   similarity of sentence-transformer embeddings.  This detects prompt
   sensitivity: a model that says "2.14%" five times and "the NIM was around
   two percent" five times is less semantically consistent than one that
   always uses the same phrasing.

2. **Factual consistency** — do the extracted numeric values agree with each
   other across runs?  Uses the *modal* (most common) value, not the mean,
   because mean hides bimodal distributions.  If a model alternates between
   "2.14%" and "3.41%", the mean (2.78%) misrepresents both modes.  The modal
   approach surfaces this: consistency = count_of_mode / total_non_none.

3. **Hallucination rate** — what fraction of extracted values deviate from
   ground truth beyond a configurable relative tolerance (default 5%)?  None
   extractions (where the extractor couldn't parse a number) count as
   hallucinations because the model failed to produce a parseable answer.

4. **Composite stability** — a weighted combination of the three scores above.
   Weights are loaded from config.yaml (always loaded from config) so they can be tuned
   without code changes.  Hallucination rate is inverted (1 - rate) so that
   higher composite = better.

Critical integrity rules:
- Semantic similarity uses sentence-transformers, NEVER the model under evaluation.
- Factual extraction uses regex (via src.extractor), NEVER the model under evaluation.
- The model under evaluation only generates the responses being scored.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.extractor import extract_numeric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

# facts.json stores monetary values in relative units (e.g. 22297 for
# "SGD 22,297 million"). The extractor always multiplies out to absolute
# values ("22,297 million" → 22_297_000_000). Without this table the
# hallucination comparison becomes 22_297_000_000 vs 22_297 and every
# monetary fact is flagged as a hallucination regardless of model quality.
_UNIT_DIVISORS: dict = {
    "sgd_millions": 1_000_000,
    "usd_millions": 1_000_000,
    "hkd_millions": 1_000_000,
    "myr_millions": 1_000_000,
    "sgd_billions": 1_000_000_000,
    "usd_billions": 1_000_000_000,
    "hkd_billions": 1_000_000_000,
}


def _normalise_to_facts_scale(
    value: Optional[float],
    expected_unit: Optional[str],
) -> Optional[float]:
    """Divide an extracted absolute value back to the scale used in facts.json.

    Examples
    --------
    "SGD 22,297 million" is extracted as 22_297_000_000 (absolute).
    With expected_unit="sgd_millions" this returns 22_297.0, which matches
    the ground truth stored in facts.json directly.

    Percent, ratio, and plain-dollar facts have no entry in _UNIT_DIVISORS
    so they pass through unchanged (divisor = 1.0).
    """
    if value is None or not expected_unit:
        return value
    divisor = _UNIT_DIVISORS.get(expected_unit.lower().strip(), 1.0)
    return value / divisor


# ---------------------------------------------------------------------------
# Lazy-loaded embedding model
# ---------------------------------------------------------------------------
# Sentence-transformers models are large (~80 MB for MiniLM).  We load once
# on first use and cache globally so that scoring hundreds of facts doesn't
# reload the model each time.  The global is module-private; callers go
# through _get_embedding_model().

_embedding_model = None
_embedding_model_name: Optional[str] = None


def _get_embedding_model(model_name: str):
    """Return a cached SentenceTransformer instance, loading on first call.

    Uses a global singleton because:
    - The model is stateless (no mutable internal state across encode() calls).
    - Loading takes 2-5 seconds; we want to pay that cost exactly once.
    - The model must be the SAME across the entire evaluation run (mixing
      embedding models makes cosine similarities incomparable).

    The cache is keyed by model_name so that switching models in the same
    Python process (e.g. during experimentation) loads the new model rather
    than silently reusing the old one.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. "all-MiniLM-L6-v2".

    Returns
    -------
    SentenceTransformer
        The loaded model, ready for .encode().
    """
    global _embedding_model, _embedding_model_name
    if _embedding_model is None or _embedding_model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
        _embedding_model_name = model_name
    return _embedding_model


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class FactScores:
    """Complete scoring results for one (fact, template, temperature) combination.

    Stores both the computed scores and the raw data that produced them,
    so downstream analysis can re-slice without re-running the pipeline.

    Attributes
    ----------
    fact_id : str
        Identifier for the ground-truth fact being evaluated.
    template : str
        Which prompt template was used (e.g. "direct_extraction").
    temperature : float
        The sampling temperature used for this batch of runs.
    n_runs : int
        Number of LLM calls in this batch.
    semantic_consistency : float
        Mean pairwise cosine similarity of response embeddings. Range [0, 1].
    factual_consistency : float
        Fraction of non-None extractions matching the modal value. Range [0, 1].
    modal_value : Optional[float]
        The most common extracted numeric value, or None if all extractions failed.
    extraction_failure_rate : float
        Fraction of responses where the extractor returned None. Range [0, 1].
    hallucination_rate : float
        Fraction of responses that deviate from ground truth beyond tolerance.
        None extractions count as hallucinations. Range [0, 1].
    ground_truth : float
        The known-correct value from facts.json.
    composite_stability : float
        Weighted combination of the three sub-scores. Range [0, 1].
    extracted_values : List[Optional[float]]
        Raw extracted values (None where extraction failed). Preserved for
        downstream re-analysis without re-running extraction.
    response_texts : List[str]
        The original LLM response texts. Preserved for reproducibility — you
        can re-run scoring from these without making new API calls.
    """
    fact_id: str
    template: str
    temperature: float
    n_runs: int
    semantic_consistency: float
    factual_consistency: float
    modal_value: Optional[float]
    extraction_failure_rate: float
    hallucination_rate: float
    ground_truth: float
    composite_stability: float
    extracted_values: List[Optional[float]]
    response_texts: List[str]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_semantic_consistency(responses: List[str], model_name: str) -> float:
    """Compute semantic consistency as mean pairwise cosine similarity.

    Embeds all N responses using a sentence-transformer model, computes the
    full N x N cosine similarity matrix, and returns the mean of the upper
    triangle (excluding the diagonal, which is always 1.0).

    Why cosine similarity over, say, BLEU or ROUGE?  Because we want to
    measure *semantic* agreement, not lexical overlap.  "The NIM was 2.14%"
    and "Net Interest Margin stood at 2.14 percent" should score highly
    despite low token overlap.

    Parameters
    ----------
    responses : List[str]
        The N response texts to compare.
    model_name : str
        HuggingFace model identifier for the sentence-transformer.

    Returns
    -------
    float
        Mean pairwise cosine similarity in [0, 1].
        Returns 1.0 if fewer than 2 responses (nothing to compare).
    """
    if len(responses) < 2:
        return 1.0

    model = _get_embedding_model(model_name)

    # Encode all responses to dense vectors
    # normalize_embeddings=True makes dot product equivalent to cosine similarity
    embeddings = model.encode(responses, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings)

    # Cosine similarity matrix: since embeddings are L2-normalized,
    # cosine_sim(a, b) = dot(a, b)
    sim_matrix = embeddings @ embeddings.T

    # Extract upper triangle (excluding diagonal)
    n = len(responses)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    # Mean of upper triangle
    score = float(np.mean(pairwise_sims))

    # Clamp to [0, 1] — cosine similarity of normalized embeddings can
    # theoretically go slightly negative for very dissimilar texts, but
    # our score semantics require [0, 1].
    return max(0.0, min(1.0, score))


def compute_factual_consistency(
    extracted_values: List[Optional[float]],
) -> Tuple[float, Optional[float], float]:
    """Compute factual consistency using the modal (most common) extracted value.

    Why modal, not mean?  Consider a model that returns "2.14%" five times and
    "3.41%" five times.  The mean is 2.78% — a value NO run actually produced.
    The mode reveals the bimodal distribution: consistency = 5/10 = 0.5, which
    accurately reflects that the model is split between two answers.

    Parameters
    ----------
    extracted_values : List[Optional[float]]
        Extracted numeric values from each run.  None indicates extraction
        failure (couldn't parse a number from the response).

    Returns
    -------
    tuple of (consistency_score, modal_value, extraction_failure_rate)
        - consistency_score: count_of_mode / count_of_non_none. Range [0, 1].
        - modal_value: the most common non-None value, or None if all failed.
        - extraction_failure_rate: count_of_none / total. Range [0, 1].
        If all values are None, returns (0.0, None, 1.0).
    """
    total = len(extracted_values)
    if total == 0:
        return (0.0, None, 1.0)

    # Separate valid extractions from failures
    valid_values = [v for v in extracted_values if v is not None]
    n_valid = len(valid_values)
    n_none = total - n_valid

    extraction_failure_rate = n_none / total

    if n_valid == 0:
        return (0.0, None, 1.0)

    # Find the mode using Counter
    # For floating point comparison, we round to 6 decimal places to handle
    # trivial floating-point noise (e.g., 2.1400000001 vs 2.14)
    rounded = [round(v, 6) for v in valid_values]
    counter = Counter(rounded)
    modal_value_rounded, mode_count = counter.most_common(1)[0]

    # Map back to the original (un-rounded) value for the modal_value.
    # Use the first occurrence that matches the rounded mode.
    modal_value: Optional[float] = None
    for v in valid_values:
        if round(v, 6) == modal_value_rounded:
            modal_value = v
            break

    consistency_score = mode_count / n_valid

    return (consistency_score, modal_value, extraction_failure_rate)


def compute_hallucination_rate(
    extracted_values: List[Optional[float]],
    ground_truth: float,
    tolerance: float = 0.05,
) -> float:
    """Compute the fraction of extracted values that are hallucinations.

    A value is a hallucination if:
    - It is None (extraction failed — the model didn't produce a parseable answer).
    - Its relative deviation from ground truth exceeds the tolerance:
      |extracted - truth| / |truth| > tolerance.

    The 5% default tolerance distinguishes rounding differences (2.14% vs 2.15%)
    from genuine hallucinations (2.14% vs 3.41%).

    Special case: when ground_truth == 0, relative error is undefined, so we
    fall back to an absolute tolerance of 0.001.

    Parameters
    ----------
    extracted_values : List[Optional[float]]
        Extracted numeric values (None = extraction failure).
    ground_truth : float
        The known-correct value from the source financial report.
    tolerance : float
        Relative tolerance threshold (default 0.05 = 5%).

    Returns
    -------
    float
        Hallucination rate in [0, 1].  0.0 = all correct, 1.0 = all hallucinated.
    """
    total = len(extracted_values)
    if total == 0:
        return 0.0

    n_hallucinations = 0
    abs_tolerance = 0.001  # Fallback for ground_truth == 0

    for value in extracted_values:
        if value is None:
            # None extraction counts as hallucination — the model failed to
            # produce a parseable numeric answer.
            n_hallucinations += 1
        elif ground_truth == 0.0:
            # Special case: can't compute relative error when truth is 0.
            if abs(value) > abs_tolerance:
                n_hallucinations += 1
        else:
            relative_error = abs(value - ground_truth) / abs(ground_truth)
            if relative_error > tolerance:
                n_hallucinations += 1

    return n_hallucinations / total


def compute_composite_stability(
    semantic: float,
    factual: float,
    hallucination: float,
    weights: Dict[str, float],
) -> float:
    """Combine sub-scores into a single composite stability score.

    The composite is a weighted sum where hallucination rate is inverted
    (1 - rate) so that higher composite always means better.  Weights are
    loaded from config.yaml, always loaded from config, so they can be tuned for
    different analysis needs without code changes.

    Default weights from config.yaml:
      semantic_consistency: 0.30
      factual_consistency:  0.40
      hallucination_rate:   0.30

    Factual consistency gets the highest default weight because getting the
    number right matters more than consistent phrasing in financial contexts.

    Parameters
    ----------
    semantic : float
        Semantic consistency score in [0, 1].
    factual : float
        Factual consistency score in [0, 1].
    hallucination : float
        Hallucination rate in [0, 1] (higher = worse).
    weights : dict
        Keys: "semantic_consistency", "factual_consistency", "hallucination_rate".
        Values must sum to 1.0 (enforced by config validation).

    Returns
    -------
    float
        Composite stability score in [0, 1].
    """
    composite = (
        semantic * weights["semantic_consistency"]
        + factual * weights["factual_consistency"]
        + (1.0 - hallucination) * weights["hallucination_rate"]
    )

    # Clamp to [0, 1] for safety (should already be in range if inputs are valid)
    return max(0.0, min(1.0, composite))


# ---------------------------------------------------------------------------
# Full scoring pipeline
# ---------------------------------------------------------------------------

def score_fact(
    fact_id: str,
    template: str,
    temperature: float,
    response_texts: List[str],
    ground_truth_value: float,
    expected_unit: Optional[str],
    embedding_model_name: str,
    hallucination_tolerance: float,
    composite_weights: Dict[str, float],
) -> FactScores:
    """Run the full scoring pipeline for one (fact, template, temperature) batch.

    This is the main entry point for scoring.  Given N response texts from
    repeated LLM calls, it:
    1. Extracts numeric values from each response using regex (src.extractor).
    2. Computes semantic consistency via sentence-transformer embeddings.
    3. Computes factual consistency using the modal extracted value.
    4. Computes hallucination rate against the known ground truth.
    5. Combines sub-scores into a composite stability score.

    Parameters
    ----------
    fact_id : str
        Identifier for the fact being evaluated (e.g. "dbs_fy2024_nim").
    template : str
        Which prompt template was used (e.g. "direct_extraction").
    temperature : float
        Sampling temperature for this batch.
    response_texts : List[str]
        The N raw LLM response texts.
    ground_truth_value : float
        The known-correct numeric value.
    expected_unit : Optional[str]
        Unit hint for the extractor (e.g. "percent", "billions").
    embedding_model_name : str
        HuggingFace model name for sentence-transformer embeddings.
    hallucination_tolerance : float
        Relative tolerance for hallucination detection (e.g. 0.05 for 5%).
    composite_weights : Dict[str, float]
        Weights for the composite score.  Keys: "semantic_consistency",
        "factual_consistency", "hallucination_rate".

    Returns
    -------
    FactScores
        Complete scoring results including raw data for reproducibility.
    """
    n_runs = len(response_texts)

    # Step 1: Extract numeric values from each response
    extracted_values: List[Optional[float]] = []
    for i, text in enumerate(response_texts):
        result = extract_numeric(text, expected_unit=expected_unit)
        extracted_values.append(result.value)
        if result.value is None:
            logger.debug(
                "Extraction failed for fact=%s template=%s temp=%.1f run=%d",
                fact_id, template, temperature, i,
            )

    # Step 1b: Normalise extracted absolute values to the same scale as
    # facts.json. The extractor expands scale words to absolute values
    # ("22,297 million" → 22_297_000_000) but facts.json stores relative
    # values (22297.0 in sgd_millions). Dividing by the unit's scale factor
    # brings both sides to the same denominator before any comparison.
    extracted_values = [
        _normalise_to_facts_scale(v, expected_unit) for v in extracted_values
    ]

    # Step 2: Compute semantic consistency
    semantic_consistency = compute_semantic_consistency(
        response_texts, embedding_model_name
    )

    # Step 3: Compute factual consistency (modal value)
    factual_consistency, modal_value, extraction_failure_rate = (
        compute_factual_consistency(extracted_values)
    )

    # Step 4: Compute hallucination rate
    hallucination_rate = compute_hallucination_rate(
        extracted_values, ground_truth_value, tolerance=hallucination_tolerance
    )

    # Step 5: Compute composite stability
    composite_stability = compute_composite_stability(
        semantic_consistency, factual_consistency, hallucination_rate,
        composite_weights,
    )

    logger.info(
        "Scored fact=%s template=%s temp=%.1f: "
        "semantic=%.3f factual=%.3f hallucination=%.3f composite=%.3f "
        "(modal=%s, extraction_failures=%d/%d)",
        fact_id, template, temperature,
        semantic_consistency, factual_consistency, hallucination_rate,
        composite_stability,
        modal_value, int(extraction_failure_rate * n_runs), n_runs,
    )

    return FactScores(
        fact_id=fact_id,
        template=template,
        temperature=temperature,
        n_runs=n_runs,
        semantic_consistency=semantic_consistency,
        factual_consistency=factual_consistency,
        modal_value=modal_value,
        extraction_failure_rate=extraction_failure_rate,
        hallucination_rate=hallucination_rate,
        ground_truth=ground_truth_value,
        composite_stability=composite_stability,
        extracted_values=extracted_values,
        response_texts=response_texts,
    )
