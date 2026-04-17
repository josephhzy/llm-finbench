"""Report generator for the LLM Financial Stability Bench.

Takes raw evaluation results from the engine and produces structured reports
at multiple levels of aggregation. This is the final stage of the pipeline:
engine produces raw responses, scorer computes per-group metrics, reporter
assembles everything into human-readable and machine-parseable outputs.

Why both CSV and summary text? CSVs are for downstream analysis (pandas,
Excel, R) — they must be self-contained with column descriptions so a
reader doesn't need to cross-reference documentation. The summary text is
for quick scanning: methodology, traffic-light breakdown, best/worst facts.
A researcher should be able to open summary.txt and know within 30 seconds
whether this run produced interesting findings.

Why comment headers in CSVs? A CSV with bare numbers is useless six months
later. The comment block (lines starting with #) documents what each column
means, what thresholds were used, and when the file was generated. pandas
and most tools skip comment lines by default.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import AppConfig, FlaggingConfig, ScoringConfig
from src.scorer import FactScores, score_fact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column ordering for the per-fact detail report. Explicit ordering prevents
# pandas from alphabetising columns, which would scatter related fields.
_DETAIL_COLUMNS = [
    "fact_id",
    "company",
    "metric",
    "category",
    "difficulty",
    "template",
    "temperature",
    "n_runs",
    "semantic_score",
    "factual_score",
    "hallucination_rate",
    "extraction_failure_rate",
    "modal_value",
    "ground_truth",
    "composite_stability",
    "flag",
]

# Columns that carry numeric scores, used for aggregation. These are the
# columns we compute mean values over when grouping by company, metric, etc.
_SCORE_COLUMNS = [
    "semantic_score",
    "factual_score",
    "hallucination_rate",
    "extraction_failure_rate",
    "composite_stability",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flag_from_score(
    composite: float,
    yellow_threshold: float,
    green_threshold: float,
) -> str:
    """Map a composite stability score to a traffic-light flag.

    The thresholds come from config.yaml's flagging section, always loaded from config.
    This function centralises the mapping so that detail rows, aggregations,
    and summary text all use identical logic.
    """
    if composite >= green_threshold:
        return "\U0001f7e2"  # green circle
    elif composite >= yellow_threshold:
        return "\U0001f7e1"  # yellow circle
    else:
        return "\U0001f534"  # red circle


def _build_facts_lookup(results_data: dict) -> Dict[str, dict]:
    """Build a lookup from fact_id to fact metadata.

    The results.json stores facts alongside call records. We extract them
    once into a dict for O(1) access during scoring.
    """
    lookup: Dict[str, dict] = {}
    for fact in results_data.get("facts", []):
        fact_id = fact.get("id")
        if fact_id is not None:
            lookup[fact_id] = fact
    return lookup


def _group_call_records(
    call_records: List[dict],
) -> Dict[Tuple[str, str, float], List[dict]]:
    """Group call records by (fact_id, template, temperature).

    Each group represents one evaluation condition. The scorer needs all
    responses for a single condition to compute consistency metrics.
    Returns a dict mapping the grouping key to the list of call records.
    """
    groups: Dict[Tuple[str, str, float], List[dict]] = defaultdict(list)
    for record in call_records:
        key = (
            record["fact_id"],
            record.get("template_name", record.get("template", "unknown")),
            record["temperature"],
        )
        groups[key].append(record)
    return groups


def _extract_response_texts(call_records: List[dict]) -> List[str]:
    """Extract the response text from each call record in a group.

    Records with missing or empty text are included as empty strings so
    the scorer can count them as extraction failures rather than silently
    dropping data points.
    """
    return [r.get("raw_response", "") or "" for r in call_records]


def _score_all_groups(
    groups: Dict[Tuple[str, str, float], List[dict]],
    facts_lookup: Dict[str, dict],
    config: AppConfig,
) -> List[dict]:
    """Score every (fact_id, template, temperature) group and return detail rows.

    Each row is a dict ready to become a DataFrame row in the per-fact detail
    report. Scoring failures are logged and the row is still emitted with
    null scores so the failure is visible in the report rather than silently
    lost.
    """
    rows: List[dict] = []

    for (fact_id, template, temperature), records in sorted(groups.items()):
        fact = facts_lookup.get(fact_id, {})
        ground_truth = fact.get("value")
        expected_unit = fact.get("unit")

        # Guard: ground_truth=None causes a TypeError inside compute_hallucination_rate
        # (abs(value - None) fails). Log clearly and emit a null-scored row rather
        # than silently recording all-zeros via the broad except below.
        if ground_truth is None:
            logger.warning(
                "No ground truth value for fact_id=%s (template=%s, temp=%.1f); "
                "skipping scoring — check that facts.json has a 'value' field for this fact.",
                fact_id, template, temperature,
            )
            rows.append({
                "fact_id": fact_id,
                "company": fact.get("company", "unknown"),
                "metric": fact.get("metric", "unknown"),
                "category": fact.get("category", "unknown"),
                "difficulty": fact.get("difficulty", "unknown"),
                "template": template,
                "temperature": temperature,
                "n_runs": len(records),
                "semantic_score": None,
                "factual_score": None,
                "hallucination_rate": None,
                "extraction_failure_rate": None,
                "modal_value": None,
                "ground_truth": None,
                "composite_stability": None,
                "flag": "\U0001f534",
            })
            continue

        response_texts = _extract_response_texts(records)

        # Score this group via the scorer module
        try:
            scores: FactScores = score_fact(
                fact_id=fact_id,
                template=template,
                temperature=temperature,
                response_texts=response_texts,
                ground_truth_value=ground_truth,
                expected_unit=expected_unit,
                embedding_model_name=config.scoring.embedding_model,
                hallucination_tolerance=config.scoring.hallucination_tolerance,
                composite_weights=config.scoring.composite_weights,
            )
        except Exception:
            logger.exception(
                "Scoring failed for fact_id=%s template=%s temp=%.1f; "
                "recording null scores.",
                fact_id,
                template,
                temperature,
            )
            scores = FactScores(
                fact_id=fact_id,
                template=template,
                temperature=temperature,
                n_runs=len(records),
                semantic_consistency=0.0,
                factual_consistency=0.0,
                modal_value=None,
                extraction_failure_rate=1.0,
                hallucination_rate=1.0,
                ground_truth=ground_truth or 0.0,
                composite_stability=0.0,
                extracted_values=[],
                response_texts=response_texts,
            )

        # Compute the flag from composite stability
        composite = scores.composite_stability
        if composite is not None:
            flag = _flag_from_score(
                composite,
                config.flagging.yellow_threshold,
                config.flagging.green_threshold,
            )
        else:
            flag = "\U0001f534"  # red — unknown scores are flagged as unstable

        rows.append({
            "fact_id": fact_id,
            "company": fact.get("company", "unknown"),
            "metric": fact.get("metric", "unknown"),
            "category": fact.get("category", "unknown"),
            "difficulty": fact.get("difficulty", "unknown"),
            "template": template,
            "temperature": temperature,
            "n_runs": len(records),
            "semantic_score": scores.semantic_consistency,
            "factual_score": scores.factual_consistency,
            "hallucination_rate": scores.hallucination_rate,
            "extraction_failure_rate": scores.extraction_failure_rate,
            "modal_value": scores.modal_value,
            "ground_truth": ground_truth,
            "composite_stability": composite,
            "flag": flag,
        })

    return rows


# ---------------------------------------------------------------------------
# CSV writing with comment headers
# ---------------------------------------------------------------------------

def _write_csv_with_header(
    df: pd.DataFrame,
    path: str,
    header_lines: List[str],
) -> None:
    """Write a DataFrame to CSV with a comment-line header block.

    The header block is a sequence of lines prefixed with '#'. pandas and
    most CSV readers skip these by default (comment='#'), so they serve as
    inline documentation without breaking tooling.
    """
    with open(path, "w", encoding="utf-8") as fh:
        for line in header_lines:
            fh.write(f"# {line}\n")
        fh.write("#\n")
        df.to_csv(fh, index=False)


def _detail_csv_header(config: AppConfig, run_id: str) -> List[str]:
    """Generate the comment header for per_fact_report.csv."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return [
        f"Per-Fact Detail Report — Run: {run_id}",
        f"Generated: {now}",
        "",
        "Column Descriptions:",
        "  fact_id                — Unique identifier for the ground truth fact",
        "  company                — Company name (DBS, OCBC, UOB, Singtel, CapitaLand)",
        "  metric                 — Financial metric being tested",
        "  category               — Metric category (profitability, capital, etc.)",
        "  difficulty             — Fact difficulty level (easy, medium, hard)",
        "  template               — Prompt template used for this evaluation group",
        "  temperature            — Sampling temperature used",
        "  n_runs                 — Number of API calls in this evaluation group",
        "  semantic_score         — Semantic consistency (0-1): cosine similarity of response embeddings",
        "  factual_score          — Factual consistency (0-1): fraction of runs matching modal extracted value",
        "  hallucination_rate     — Fraction of extracted values outside tolerance of ground truth",
        "  extraction_failure_rate — Fraction of responses where numeric extraction returned null",
        "  modal_value            — Most commonly extracted numeric value across runs",
        "  ground_truth           — Expected value from verified annual report",
        "  composite_stability    — Weighted combination of scores (see config for weights)",
        "  flag                   — Traffic-light indicator based on composite_stability",
        "",
        "Flagging Thresholds:",
        f"  \U0001f7e2 Green  — composite_stability >= {config.flagging.green_threshold}",
        f"  \U0001f7e1 Yellow — composite_stability >= {config.flagging.yellow_threshold}",
        f"  \U0001f534 Red    — composite_stability <  {config.flagging.yellow_threshold}",
        "",
        "Scoring Weights:",
        f"  semantic_consistency: {config.scoring.composite_weights.get('semantic_consistency', 'N/A')}",
        f"  factual_consistency:  {config.scoring.composite_weights.get('factual_consistency', 'N/A')}",
        f"  hallucination_rate:   {config.scoring.composite_weights.get('hallucination_rate', 'N/A')}",
        f"  hallucination_tolerance: {config.scoring.hallucination_tolerance} "
        f"({config.scoring.hallucination_tolerance * 100:.0f}% relative)",
    ]


def _aggregated_csv_header(
    title: str,
    group_col: str,
    config: AppConfig,
    run_id: str,
) -> List[str]:
    """Generate the comment header for any aggregated report CSV."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return [
        f"{title} — Run: {run_id}",
        f"Generated: {now}",
        "",
        f"Aggregated by: {group_col}",
        "Each row is the mean of all per-fact scores within that group.",
        "",
        "Column Descriptions:",
        f"  {group_col}              — Grouping key",
        "  semantic_score         — Mean semantic consistency across group",
        "  factual_score          — Mean factual consistency across group",
        "  hallucination_rate     — Mean hallucination rate across group",
        "  extraction_failure_rate — Mean extraction failure rate across group",
        "  composite_stability    — Mean composite stability across group",
        "  n_facts                — Number of per-fact rows in this group",
        "",
        "Flagging Thresholds (applied to mean composite_stability):",
        f"  \U0001f7e2 Green  — >= {config.flagging.green_threshold}",
        f"  \U0001f7e1 Yellow — >= {config.flagging.yellow_threshold}",
        f"  \U0001f534 Red    — <  {config.flagging.yellow_threshold}",
    ]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_by(
    detail_df: pd.DataFrame,
    group_col: str,
    config: AppConfig,
) -> pd.DataFrame:
    """Aggregate detail rows by a single column, computing mean scores.

    Also adds n_facts (count of rows per group) and a flag column based
    on the mean composite_stability.
    """
    if detail_df.empty:
        return pd.DataFrame()

    # Only aggregate columns that exist and have data
    agg_cols = [c for c in _SCORE_COLUMNS if c in detail_df.columns]

    grouped = detail_df.groupby(group_col, sort=True)[agg_cols].mean()
    grouped["n_facts"] = detail_df.groupby(group_col).size()

    # Add flag based on mean composite_stability
    if "composite_stability" in grouped.columns:
        grouped["flag"] = grouped["composite_stability"].apply(
            lambda x: _flag_from_score(
                x if pd.notna(x) else 0.0,
                config.flagging.yellow_threshold,
                config.flagging.green_threshold,
            )
        )
    else:
        grouped["flag"] = "\U0001f534"

    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Summary text generation
# ---------------------------------------------------------------------------

def _generate_summary(
    detail_df: pd.DataFrame,
    config: AppConfig,
    run_id: str,
    results_data: dict,
) -> str:
    """Generate the summary.txt content.

    Includes methodology, overall statistics, top/bottom facts, and
    auto-generated observations. This is meant to be scannable — a
    researcher should get the key findings within 30 seconds.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: List[str] = []

    # ---- Title ----
    lines.append("=" * 72)
    lines.append("  LLM Financial Stability Bench — Evaluation Summary")
    lines.append("=" * 72)
    lines.append(f"Run ID:    {run_id}")
    lines.append(f"Generated: {now}")
    lines.append("")

    # ---- Methodology ----
    lines.append("-" * 72)
    lines.append("  METHODOLOGY")
    lines.append("-" * 72)

    # Pull model info from config or results metadata
    run_config = results_data.get("config", {})
    model_name = run_config.get("model", {}).get("name", config.model.name)
    model_provider = run_config.get("model", {}).get("provider", config.model.provider)

    eval_cfg = run_config.get("evaluation", {})
    temperatures = eval_cfg.get("temperatures", config.evaluation.temperatures)
    runs_per = eval_cfg.get("runs_per_combination", config.evaluation.runs_per_combination)
    templates = eval_cfg.get("templates", config.evaluation.templates)

    scoring_cfg = run_config.get("scoring", {})
    embedding_model = scoring_cfg.get("embedding_model", config.scoring.embedding_model)
    hallucination_tol = scoring_cfg.get(
        "hallucination_tolerance", config.scoring.hallucination_tolerance
    )
    composite_weights = scoring_cfg.get(
        "composite_weights", config.scoring.composite_weights
    )

    lines.append(f"Model:                {model_provider}/{model_name}")
    lines.append(f"Temperature range:    {temperatures}")
    lines.append(f"Runs per combination: {runs_per}")
    lines.append(f"Templates:            {templates}")
    lines.append(f"Embedding model:      {embedding_model}")
    lines.append(f"Hallucination tol.:   {hallucination_tol} ({hallucination_tol * 100:.0f}% relative)")
    lines.append(f"Composite weights:    {composite_weights}")
    lines.append(f"Scoring method:       Modal value for factual consistency, "
                 f"cosine similarity for semantic consistency")
    lines.append("")

    # ---- Overall Statistics ----
    lines.append("-" * 72)
    lines.append("  OVERALL STATISTICS")
    lines.append("-" * 72)

    if detail_df.empty:
        lines.append("No scored results available.")
        lines.append("")
    else:
        total_rows = len(detail_df)
        valid_composite = detail_df["composite_stability"].dropna()

        if not valid_composite.empty:
            mean_stability = valid_composite.mean()
            median_stability = valid_composite.median()
            std_stability = valid_composite.std()
            min_stability = valid_composite.min()
            max_stability = valid_composite.max()
        else:
            mean_stability = 0.0
            median_stability = 0.0
            std_stability = 0.0
            min_stability = 0.0
            max_stability = 0.0

        # Count flags
        n_green = (detail_df["flag"] == "\U0001f7e2").sum()
        n_yellow = (detail_df["flag"] == "\U0001f7e1").sum()
        n_red = (detail_df["flag"] == "\U0001f534").sum()

        pct_green = (n_green / total_rows * 100) if total_rows > 0 else 0
        pct_yellow = (n_yellow / total_rows * 100) if total_rows > 0 else 0
        pct_red = (n_red / total_rows * 100) if total_rows > 0 else 0

        lines.append(f"Total evaluation groups: {total_rows}")
        lines.append(f"Mean composite stability:   {mean_stability:.4f}")
        lines.append(f"Median composite stability: {median_stability:.4f}")
        lines.append(f"Std dev:                    {std_stability:.4f}")
        lines.append(f"Range:                      [{min_stability:.4f}, {max_stability:.4f}]")
        lines.append("")
        lines.append(f"Flag distribution:")
        lines.append(f"  \U0001f7e2 Green  (>= {config.flagging.green_threshold}): "
                     f"{n_green:>4d} ({pct_green:5.1f}%)")
        lines.append(f"  \U0001f7e1 Yellow (>= {config.flagging.yellow_threshold}): "
                     f"{n_yellow:>4d} ({pct_yellow:5.1f}%)")
        lines.append(f"  \U0001f534 Red    (<  {config.flagging.yellow_threshold}): "
                     f"{n_red:>4d} ({pct_red:5.1f}%)")
        lines.append("")

        # Mean scores breakdown
        lines.append("Mean scores across all groups:")
        for col in _SCORE_COLUMNS:
            if col in detail_df.columns:
                valid = detail_df[col].dropna()
                if not valid.empty:
                    lines.append(f"  {col:30s} {valid.mean():.4f}")
                else:
                    lines.append(f"  {col:30s} N/A")
        lines.append("")

    # ---- Top 5 Most Stable ----
    lines.append("-" * 72)
    lines.append("  TOP 5 MOST STABLE FACTS")
    lines.append("-" * 72)

    if not detail_df.empty and "composite_stability" in detail_df.columns:
        # Aggregate per fact_id for ranking (a fact appears in multiple template/temp combos)
        per_fact = detail_df.groupby("fact_id").agg({
            "composite_stability": "mean",
            "company": "first",
            "metric": "first",
        }).sort_values("composite_stability", ascending=False)

        for i, (fact_id, row) in enumerate(per_fact.head(5).iterrows()):
            score = row["composite_stability"]
            flag = _flag_from_score(
                score if pd.notna(score) else 0.0,
                config.flagging.yellow_threshold,
                config.flagging.green_threshold,
            )
            lines.append(
                f"  {i + 1}. {flag} {fact_id:40s} "
                f"{row['company']:12s} {row['metric']:30s} "
                f"stability={score:.4f}"
            )
    else:
        lines.append("  No data available.")
    lines.append("")

    # ---- Top 5 Least Stable ----
    lines.append("-" * 72)
    lines.append("  TOP 5 LEAST STABLE FACTS")
    lines.append("-" * 72)

    if not detail_df.empty and "composite_stability" in detail_df.columns:
        per_fact = detail_df.groupby("fact_id").agg({
            "composite_stability": "mean",
            "company": "first",
            "metric": "first",
        }).sort_values("composite_stability", ascending=True)

        for i, (fact_id, row) in enumerate(per_fact.head(5).iterrows()):
            score = row["composite_stability"]
            flag = _flag_from_score(
                score if pd.notna(score) else 0.0,
                config.flagging.yellow_threshold,
                config.flagging.green_threshold,
            )
            lines.append(
                f"  {i + 1}. {flag} {fact_id:40s} "
                f"{row['company']:12s} {row['metric']:30s} "
                f"stability={score:.4f}"
            )
    else:
        lines.append("  No data available.")
    lines.append("")

    # ---- Key Findings ----
    lines.append("-" * 72)
    lines.append("  KEY FINDINGS (auto-generated)")
    lines.append("-" * 72)

    findings = _generate_findings(detail_df, config)
    if findings:
        for finding in findings:
            lines.append(f"  * {finding}")
    else:
        lines.append("  No findings could be generated (insufficient data).")
    lines.append("")

    lines.append("=" * 72)
    lines.append("  End of Summary")
    lines.append("=" * 72)

    return "\n".join(lines)


def _generate_findings(detail_df: pd.DataFrame, config: AppConfig) -> List[str]:
    """Auto-generate key observations from the scored data.

    These are simple heuristic observations — not interpretations. They
    highlight patterns (temperature sensitivity, company differences, etc.)
    that a researcher should investigate further.
    """
    findings: List[str] = []

    if detail_df.empty:
        return findings

    # Finding 1: Temperature sensitivity
    if "temperature" in detail_df.columns and detail_df["temperature"].nunique() > 1:
        temp_means = detail_df.groupby("temperature")["composite_stability"].mean()
        if not temp_means.empty:
            best_temp = temp_means.idxmax()
            worst_temp = temp_means.idxmin()
            spread = temp_means.max() - temp_means.min()
            findings.append(
                f"Temperature sensitivity: best stability at T={best_temp} "
                f"({temp_means[best_temp]:.4f}), worst at T={worst_temp} "
                f"({temp_means[worst_temp]:.4f}), spread={spread:.4f}."
            )

    # Finding 2: Template sensitivity
    if "template" in detail_df.columns and detail_df["template"].nunique() > 1:
        tpl_means = detail_df.groupby("template")["composite_stability"].mean()
        if not tpl_means.empty:
            best_tpl = tpl_means.idxmax()
            worst_tpl = tpl_means.idxmin()
            findings.append(
                f"Template sensitivity: '{best_tpl}' most stable "
                f"({tpl_means[best_tpl]:.4f}), '{worst_tpl}' least stable "
                f"({tpl_means[worst_tpl]:.4f})."
            )

    # Finding 3: Company-level differences
    if "company" in detail_df.columns and detail_df["company"].nunique() > 1:
        co_means = detail_df.groupby("company")["composite_stability"].mean()
        if not co_means.empty:
            best_co = co_means.idxmax()
            worst_co = co_means.idxmin()
            spread = co_means.max() - co_means.min()
            if spread > 0.05:
                findings.append(
                    f"Company variance: '{best_co}' has highest mean stability "
                    f"({co_means[best_co]:.4f}), '{worst_co}' has lowest "
                    f"({co_means[worst_co]:.4f}). Spread of {spread:.4f} — "
                    f"investigate whether this reflects model familiarity or "
                    f"metric complexity."
                )

    # Finding 4: Hallucination hotspots
    if "hallucination_rate" in detail_df.columns:
        high_halluc = detail_df[detail_df["hallucination_rate"] > 0.5]
        if not high_halluc.empty:
            n_high = len(high_halluc)
            total = len(detail_df)
            worst_facts = high_halluc.nsmallest(
                min(3, len(high_halluc)), "composite_stability"
            )["fact_id"].tolist()
            findings.append(
                f"Hallucination hotspots: {n_high}/{total} groups have "
                f">50% hallucination rate. Worst offenders: {worst_facts}."
            )

    # Finding 5: Extraction failure rate
    if "extraction_failure_rate" in detail_df.columns:
        mean_fail = detail_df["extraction_failure_rate"].mean()
        if pd.notna(mean_fail) and mean_fail > 0.1:
            findings.append(
                f"Extraction issues: mean extraction failure rate is "
                f"{mean_fail:.1%} — review prompt templates to encourage "
                f"more parseable numeric responses."
            )

    # Finding 6: Overall quality assessment
    if "composite_stability" in detail_df.columns:
        valid = detail_df["composite_stability"].dropna()
        if not valid.empty:
            overall = valid.mean()
            if overall >= config.flagging.green_threshold:
                findings.append(
                    f"Overall assessment: mean stability {overall:.4f} is above "
                    f"green threshold ({config.flagging.green_threshold}). "
                    f"Model shows strong consistency on this fact set."
                )
            elif overall >= config.flagging.yellow_threshold:
                findings.append(
                    f"Overall assessment: mean stability {overall:.4f} is in the "
                    f"yellow zone. Model is moderately consistent but has room "
                    f"for improvement."
                )
            else:
                findings.append(
                    f"Overall assessment: mean stability {overall:.4f} is below "
                    f"yellow threshold ({config.flagging.yellow_threshold}). "
                    f"Model shows concerning inconsistency on this fact set."
                )

    return findings


# ---------------------------------------------------------------------------
# Bootstrap CI side-car
# ---------------------------------------------------------------------------

# Metrics to bootstrap. Kept in sync with evaluate_with_ci.DEFAULT_METRICS —
# duplicated here rather than imported so the reporter does not take a
# dependency on a top-level CLI script.
_CI_METRICS: Tuple[str, ...] = (
    "composite_stability",
    "semantic_score",
    "factual_score",
    "hallucination_rate",
    "extraction_failure_rate",
)


def _emit_bootstrap_ci_summary(
    detail_df: pd.DataFrame,
    out_path: Path,
    run_id: str,
    bootstrap_iter: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Optional[str]:
    """Write reports/ci_summary.csv with bootstrap 95% CIs per metric.

    Fact-level CIs are the primary number — resample FACT IDs with
    replacement, compute the mean of fact-means per resample, report the
    2.5th and 97.5th percentiles. Group-level CIs resample rows
    independently; they are narrower and should be treated as a
    diagnostic, not as a model-comparison number. See
    ``docs/STATISTICAL_RIGOR.md`` and ``evaluate_with_ci.py``.

    This function is intentionally defensive — a failure here must NOT
    crash report generation. Returns the written path on success, or
    ``None`` if the CI summary was skipped (empty detail, missing
    columns, or numpy import failure).
    """
    if detail_df is None or detail_df.empty:
        logger.info("CI summary skipped: detail_df is empty.")
        return None
    if "fact_id" not in detail_df.columns:
        logger.info("CI summary skipped: detail_df has no 'fact_id' column.")
        return None

    available = [m for m in _CI_METRICS if m in detail_df.columns]
    if not available:
        logger.info(
            "CI summary skipped: none of %s present in detail_df.",
            list(_CI_METRICS),
        )
        return None

    try:
        import numpy as np  # Local import — report.py already pulls numpy transitively.
    except ImportError as exc:
        logger.warning("CI summary skipped (numpy unavailable): %s", exc)
        return None

    rng = np.random.default_rng(seed)
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    def _bootstrap(values: "np.ndarray") -> Tuple[float, float]:
        n = len(values)
        if n == 0:
            return (float("nan"), float("nan"))
        idx = rng.integers(low=0, high=n, size=(bootstrap_iter, n))
        means = values[idx].mean(axis=1)
        return (
            float(np.percentile(means, lo_pct)),
            float(np.percentile(means, hi_pct)),
        )

    rows: List[Dict[str, object]] = []
    for metric in available:
        per_fact = (
            detail_df.groupby("fact_id")[metric].mean().dropna().to_numpy()
        )
        group_values = detail_df[metric].dropna().to_numpy()
        if per_fact.size == 0 or group_values.size == 0:
            continue
        fact_point = float(per_fact.mean())
        fact_lo, fact_hi = _bootstrap(per_fact)
        group_point = float(group_values.mean())
        group_lo, group_hi = _bootstrap(group_values)
        rows.append({
            "metric": metric,
            "fact_point": round(fact_point, 6),
            "fact_ci_lower": round(fact_lo, 6),
            "fact_ci_upper": round(fact_hi, 6),
            "n_facts": int(per_fact.size),
            "group_point": round(group_point, 6),
            "group_ci_lower": round(group_lo, 6),
            "group_ci_upper": round(group_hi, 6),
            "n_groups": int(group_values.size),
        })

    if not rows:
        logger.info("CI summary skipped: no bootstrappable rows found.")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ci_df = pd.DataFrame(rows)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "# Bootstrap 95% CIs over per_fact_report.csv\n"
            f"# Run ID:      {run_id}\n"
            f"# Bootstrap:   {bootstrap_iter} iterations, seed={seed}, alpha={alpha}\n"
            "# Primary:     fact_point / fact_ci_lower / fact_ci_upper  (resample facts)\n"
            "# Secondary:   group_point / group_ci_lower / group_ci_upper (resample rows)\n"
            "# See docs/STATISTICAL_RIGOR.md for why fact-level is the citable CI.\n"
            "#\n"
        )
        ci_df.to_csv(fh, index=False)
    logger.info("Wrote bootstrap CI summary: %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    run_id: str,
    results_dir: str,
    config: AppConfig,
    output_dir: str,
    output_format: str = "csv",
) -> dict:
    """Generate all report outputs from a completed evaluation run.

    Loads raw results from results/{run_id}/results.json, scores each
    (fact_id, template, temperature) group via scorer.score_fact(), and
    produces:
      - per_fact_report.csv   — one row per evaluation group
      - by_metric_type.csv    — aggregated by fact category
      - by_company.csv        — aggregated by company
      - by_temperature.csv    — aggregated by temperature
      - by_template.csv       — aggregated by template
      - by_difficulty.csv     — aggregated by fact difficulty (easy/medium/hard)
      - summary.txt           — human-readable overview with methodology

    Parameters
    ----------
    run_id : str
        Identifier for the evaluation run. Used to locate the results file
        and label output files.
    results_dir : str
        Base directory containing run results. The function looks for
        {results_dir}/{run_id}/results.json.
    config : AppConfig
        The application configuration. Used for scoring parameters,
        flagging thresholds, and methodology metadata.
    output_dir : str
        Directory where report files will be written. Created if it does
        not exist.
    output_format : str
        Output format for tabular data. Currently supports "csv" (default)
        and "json". The summary.txt is always generated regardless of this
        setting.

    Returns
    -------
    dict
        A mapping from report name to the absolute file path written.
        Keys: "per_fact_detail", "by_metric_type", "by_company",
        "by_temperature", "by_template", "by_difficulty", "summary".

    Raises
    ------
    FileNotFoundError
        If the results file does not exist at the expected path.
    json.JSONDecodeError
        If the results file contains invalid JSON.
    """
    # --- Load raw results ---
    results_path = Path(results_dir) / run_id / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path.resolve()}. "
            f"Ensure the evaluation run '{run_id}' completed successfully."
        )

    logger.info("Loading results from %s", results_path)
    with open(results_path, "r", encoding="utf-8") as fh:
        results_data = json.load(fh)

    # --- Restore scoring/flagging params from the run's saved config snapshot ---
    # report.py loads the current config.yaml, but scoring and flagging parameters
    # must match the original run to produce reproducible scores. The run always
    # saves its full config in results.json; use it here if present.
    saved = results_data.get("config", {})
    saved_scoring = saved.get("scoring", {})
    saved_flagging = saved.get("flagging", {})
    if saved_scoring:
        config = replace(config, scoring=ScoringConfig(
            embedding_model=saved_scoring["embedding_model"],
            hallucination_tolerance=saved_scoring["hallucination_tolerance"],
            composite_weights=saved_scoring["composite_weights"],
        ))
        logger.info("Using scoring config from run snapshot (embedding=%s, tol=%.2f).",
                    config.scoring.embedding_model, config.scoring.hallucination_tolerance)
    if saved_flagging:
        config = replace(config, flagging=FlaggingConfig(
            green_threshold=saved_flagging["green_threshold"],
            yellow_threshold=saved_flagging["yellow_threshold"],
        ))

    # --- Prepare output directory ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Build facts lookup and group call records ---
    facts_lookup = _build_facts_lookup(results_data)
    call_records = results_data.get("call_records", [])

    if not call_records:
        logger.warning("No call records found in results file. Reports will be empty.")

    groups = _group_call_records(call_records)
    logger.info(
        "Found %d call records in %d evaluation groups.",
        len(call_records),
        len(groups),
    )

    # --- Score all groups ---
    detail_rows = _score_all_groups(groups, facts_lookup, config)

    # --- Build detail DataFrame ---
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        # Ensure column ordering matches spec
        present_cols = [c for c in _DETAIL_COLUMNS if c in detail_df.columns]
        extra_cols = [c for c in detail_df.columns if c not in _DETAIL_COLUMNS]
        detail_df = detail_df[present_cols + extra_cols]

    # --- Generate output files ---
    generated_files: Dict[str, str] = {}

    # 1. Per-fact detail
    detail_file = str(out_path / f"per_fact_report.{output_format}")
    if output_format == "json":
        detail_df.to_json(detail_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            detail_df,
            detail_file,
            _detail_csv_header(config, run_id),
        )
    generated_files["per_fact_detail"] = os.path.abspath(detail_file)
    logger.info("Wrote per-fact detail: %s", detail_file)

    # 2. Aggregated by metric type (category)
    by_metric_df = _aggregate_by(detail_df, "category", config)
    by_metric_file = str(out_path / f"by_metric_type.{output_format}")
    if output_format == "json":
        by_metric_df.to_json(by_metric_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            by_metric_df,
            by_metric_file,
            _aggregated_csv_header(
                "Aggregated by Metric Type (Category)",
                "category",
                config,
                run_id,
            ),
        )
    generated_files["by_metric_type"] = os.path.abspath(by_metric_file)
    logger.info("Wrote by-metric-type: %s", by_metric_file)

    # 3. Aggregated by company
    by_company_df = _aggregate_by(detail_df, "company", config)
    by_company_file = str(out_path / f"by_company.{output_format}")
    if output_format == "json":
        by_company_df.to_json(by_company_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            by_company_df,
            by_company_file,
            _aggregated_csv_header(
                "Aggregated by Company",
                "company",
                config,
                run_id,
            ),
        )
    generated_files["by_company"] = os.path.abspath(by_company_file)
    logger.info("Wrote by-company: %s", by_company_file)

    # 4. Aggregated by temperature
    by_temp_df = _aggregate_by(detail_df, "temperature", config)
    by_temp_file = str(out_path / f"by_temperature.{output_format}")
    if output_format == "json":
        by_temp_df.to_json(by_temp_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            by_temp_df,
            by_temp_file,
            _aggregated_csv_header(
                "Aggregated by Temperature",
                "temperature",
                config,
                run_id,
            ),
        )
    generated_files["by_temperature"] = os.path.abspath(by_temp_file)
    logger.info("Wrote by-temperature: %s", by_temp_file)

    # 5. Aggregated by template
    by_tpl_df = _aggregate_by(detail_df, "template", config)
    by_tpl_file = str(out_path / f"by_template.{output_format}")
    if output_format == "json":
        by_tpl_df.to_json(by_tpl_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            by_tpl_df,
            by_tpl_file,
            _aggregated_csv_header(
                "Aggregated by Template",
                "template",
                config,
                run_id,
            ),
        )
    generated_files["by_template"] = os.path.abspath(by_tpl_file)
    logger.info("Wrote by-template: %s", by_tpl_file)

    # 6. Aggregated by difficulty (easy/medium/hard)
    by_diff_df = _aggregate_by(detail_df, "difficulty", config)
    by_diff_file = str(out_path / f"by_difficulty.{output_format}")
    if output_format == "json":
        by_diff_df.to_json(by_diff_file, orient="records", indent=2)
    else:
        _write_csv_with_header(
            by_diff_df,
            by_diff_file,
            _aggregated_csv_header(
                "Aggregated by Difficulty",
                "difficulty",
                config,
                run_id,
            ),
        )
    generated_files["by_difficulty"] = os.path.abspath(by_diff_file)
    logger.info("Wrote by-difficulty: %s", by_diff_file)

    # 7. Summary text (always generated, regardless of output_format)
    summary_text = _generate_summary(detail_df, config, run_id, results_data)
    summary_file = str(out_path / "summary.txt")
    with open(summary_file, "w", encoding="utf-8") as fh:
        fh.write(summary_text)
    generated_files["summary"] = os.path.abspath(summary_file)
    logger.info("Wrote summary: %s", summary_file)

    # 8. Bootstrap CI side-car (non-fatal — skipped if detail_df is empty
    #    or the required columns are missing; a failure here must not
    #    crash report generation). Makes every report publication carry
    #    uncertainty quantification at no extra cost. See docs/STATISTICAL_RIGOR.md.
    try:
        ci_path = _emit_bootstrap_ci_summary(
            detail_df,
            out_path / "ci_summary.csv",
            run_id=run_id,
        )
        if ci_path is not None:
            generated_files["ci_summary"] = os.path.abspath(ci_path)
    except Exception as exc:  # defensive: CI is a side-car, not a requirement
        logger.warning("Bootstrap CI summary failed, continuing: %s", exc)

    logger.info(
        "Report generation complete. %d files written to %s",
        len(generated_files),
        out_path,
    )

    return generated_files
