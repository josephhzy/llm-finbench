"""Model comparison module for the LLM Financial Stability Bench.

Compares evaluation results from two different models (or two runs of the
same model with different configurations) across all key metrics. Produces
a structured comparison report identifying which model performs better on
each dimension and an overall recommendation.

Typical usage::

    from src.comparison import compare_runs, format_comparison_report

    report = compare_runs(
        run_a_summary=summary_a,
        run_b_summary=summary_b,
        label_a="<provider-a/model-a>",
        label_b="<provider-b/model-b>",
    )
    print(format_comparison_report(report))

The comparison operates on summary-level statistics (means across all
evaluation groups) rather than per-fact detail, because the primary use
case is model selection: "which model should I trust for financial data
extraction?" Per-fact drill-downs belong in the reporter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """Summary statistics for one evaluation run.

    These are the aggregate values that appear in summary.txt and the
    aggregated CSVs. They represent means across all (fact, template,
    temperature) evaluation groups in the run.

    Attributes
    ----------
    label : str
        Human-readable identifier for this run (e.g. model name).
    model_name : str
        The exact model identifier from config (e.g. "gpt-5-nano").
    composite_stability : float
        Mean composite stability across all groups. Range [0, 1].
    semantic_consistency : float
        Mean semantic consistency. Range [0, 1].
    factual_consistency : float
        Mean factual consistency. Range [0, 1].
    hallucination_rate : float
        Mean hallucination rate. Range [0, 1]. Lower is better.
    extraction_failure_rate : float
        Mean extraction failure rate. Range [0, 1]. Lower is better.
    n_groups : int
        Total number of evaluation groups scored.
    n_green : int
        Count of groups flagged green (composite >= green_threshold).
    n_yellow : int
        Count of groups flagged yellow.
    n_red : int
        Count of groups flagged red.
    """

    label: str
    model_name: str
    composite_stability: float
    semantic_consistency: float
    factual_consistency: float
    hallucination_rate: float
    extraction_failure_rate: float
    n_groups: int
    n_green: int = 0
    n_yellow: int = 0
    n_red: int = 0


@dataclass
class MetricComparison:
    """Comparison result for a single metric.

    Attributes
    ----------
    metric_name : str
        Human-readable metric name.
    value_a : float
        Value for run A.
    value_b : float
        Value for run B.
    delta : float
        value_b - value_a. Positive means B is higher.
    winner : str
        Label of the better run for this metric, or "tie" if within
        tolerance.
    higher_is_better : bool
        Whether a higher value means better performance for this metric.
    """

    metric_name: str
    value_a: float
    value_b: float
    delta: float
    winner: str
    higher_is_better: bool


@dataclass
class ComparisonReport:
    """Full comparison report between two evaluation runs.

    Attributes
    ----------
    label_a : str
        Label for run A.
    label_b : str
        Label for run B.
    metric_comparisons : List[MetricComparison]
        Per-metric comparison results.
    overall_winner : str
        Label of the overall better run, or "tie".
    wins_a : int
        Number of metrics where A is better.
    wins_b : int
        Number of metrics where B is better.
    ties : int
        Number of metrics that are tied (within tolerance).
    recommendation : str
        Human-readable recommendation text.
    """

    label_a: str
    label_b: str
    metric_comparisons: List[MetricComparison] = field(default_factory=list)
    overall_winner: str = "tie"
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

# Metrics where lower is better (rates).
_LOWER_IS_BETTER = {"hallucination_rate", "extraction_failure_rate"}

# Tie tolerance: differences smaller than this are considered ties.
# 0.5 percentage points (0.005 on a 0-1 scale) avoids declaring a winner
# on noise.
_TIE_TOLERANCE = 0.005


def _compare_metric(
    name: str,
    value_a: float,
    value_b: float,
    label_a: str,
    label_b: str,
    tolerance: float = _TIE_TOLERANCE,
) -> MetricComparison:
    """Compare a single metric between two runs."""
    higher_is_better = name not in _LOWER_IS_BETTER
    delta = value_b - value_a

    if abs(delta) <= tolerance:
        winner = "tie"
    elif higher_is_better:
        winner = label_b if delta > 0 else label_a
    else:
        # Lower is better: negative delta means B is lower (better).
        winner = label_b if delta < 0 else label_a

    return MetricComparison(
        metric_name=name,
        value_a=value_a,
        value_b=value_b,
        delta=delta,
        winner=winner,
        higher_is_better=higher_is_better,
    )


def compare_runs(
    run_a_summary: RunSummary,
    run_b_summary: RunSummary,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
    tolerance: float = _TIE_TOLERANCE,
) -> ComparisonReport:
    """Compare two evaluation runs across all key metrics.

    Parameters
    ----------
    run_a_summary : RunSummary
        Summary statistics for run A.
    run_b_summary : RunSummary
        Summary statistics for run B.
    label_a : str, optional
        Override label for run A. Defaults to run_a_summary.label.
    label_b : str, optional
        Override label for run B. Defaults to run_b_summary.label.
    tolerance : float
        Differences smaller than this are considered ties.

    Returns
    -------
    ComparisonReport
        Full comparison with per-metric results and recommendation.
    """
    la = label_a or run_a_summary.label
    lb = label_b or run_b_summary.label

    metrics_to_compare = [
        (
            "composite_stability",
            run_a_summary.composite_stability,
            run_b_summary.composite_stability,
        ),
        (
            "factual_consistency",
            run_a_summary.factual_consistency,
            run_b_summary.factual_consistency,
        ),
        (
            "semantic_consistency",
            run_a_summary.semantic_consistency,
            run_b_summary.semantic_consistency,
        ),
        (
            "hallucination_rate",
            run_a_summary.hallucination_rate,
            run_b_summary.hallucination_rate,
        ),
        (
            "extraction_failure_rate",
            run_a_summary.extraction_failure_rate,
            run_b_summary.extraction_failure_rate,
        ),
    ]

    comparisons: List[MetricComparison] = []
    wins_a = 0
    wins_b = 0
    ties = 0

    for name, va, vb in metrics_to_compare:
        mc = _compare_metric(name, va, vb, la, lb, tolerance)
        comparisons.append(mc)
        if mc.winner == la:
            wins_a += 1
        elif mc.winner == lb:
            wins_b += 1
        else:
            ties += 1

    # Overall winner based on majority of metric wins.
    if wins_a > wins_b:
        overall_winner = la
    elif wins_b > wins_a:
        overall_winner = lb
    else:
        overall_winner = "tie"

    recommendation = _generate_recommendation(
        la,
        lb,
        comparisons,
        overall_winner,
        run_a_summary,
        run_b_summary,
    )

    return ComparisonReport(
        label_a=la,
        label_b=lb,
        metric_comparisons=comparisons,
        overall_winner=overall_winner,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        recommendation=recommendation,
    )


def _generate_recommendation(
    label_a: str,
    label_b: str,
    comparisons: List[MetricComparison],
    overall_winner: str,
    summary_a: RunSummary,
    summary_b: RunSummary,
) -> str:
    """Generate a human-readable recommendation from comparison results."""
    lines: List[str] = []

    if overall_winner == "tie":
        lines.append(
            f"The two models ({label_a} vs {label_b}) perform comparably "
            f"across the evaluated metrics. Consider cost, latency, or "
            f"other factors for selection."
        )
    else:
        loser = label_b if overall_winner == label_a else label_a
        lines.append(
            f"{overall_winner} outperforms {loser} on the majority of "
            f"evaluated metrics."
        )

    # Highlight the most impactful metric: hallucination rate.
    halluc = next(
        (c for c in comparisons if c.metric_name == "hallucination_rate"),
        None,
    )
    if halluc and halluc.winner != "tie":
        lines.append(
            f"Notably, {halluc.winner} has a lower hallucination rate "
            f"({min(halluc.value_a, halluc.value_b):.1%} vs "
            f"{max(halluc.value_a, halluc.value_b):.1%}), which is "
            f"critical for financial data accuracy."
        )

    # Flag if either model has high hallucination rate overall.
    for summary, label in [(summary_a, label_a), (summary_b, label_b)]:
        if summary.hallucination_rate > 0.5:
            lines.append(
                f"Warning: {label} has a mean hallucination rate of "
                f"{summary.hallucination_rate:.1%}, exceeding 50%. "
                f"Manual verification is strongly recommended."
            )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_comparison_report(report: ComparisonReport) -> str:
    """Format a ComparisonReport as a human-readable text report.

    Parameters
    ----------
    report : ComparisonReport
        The comparison report to format.

    Returns
    -------
    str
        Formatted text report suitable for printing or saving to file.
    """
    lines: List[str] = []
    w = 72

    lines.append("=" * w)
    lines.append("  Model Comparison Report")
    lines.append("=" * w)
    lines.append(f"  Model A: {report.label_a}")
    lines.append(f"  Model B: {report.label_b}")
    lines.append("")

    # Metric comparison table
    lines.append("-" * w)
    lines.append("  METRIC COMPARISON")
    lines.append("-" * w)
    lines.append(f"  {'Metric':<28s} {'A':>8s} {'B':>8s} {'Delta':>8s}  Winner")
    lines.append(f"  {'-' * 66}")

    for mc in report.metric_comparisons:
        delta_str = f"{mc.delta:+.4f}"
        lines.append(
            f"  {mc.metric_name:<28s} {mc.value_a:>8.4f} {mc.value_b:>8.4f} "
            f"{delta_str:>8s}  {mc.winner}"
        )
    lines.append("")

    # Score summary
    lines.append("-" * w)
    lines.append("  SCORECARD")
    lines.append("-" * w)
    lines.append(f"  {report.label_a} wins: {report.wins_a}")
    lines.append(f"  {report.label_b} wins: {report.wins_b}")
    lines.append(f"  Ties:             {report.ties}")
    lines.append(f"  Overall winner:   {report.overall_winner}")
    lines.append("")

    # Recommendation
    lines.append("-" * w)
    lines.append("  RECOMMENDATION")
    lines.append("-" * w)
    # Wrap recommendation at ~70 chars.
    rec_words = report.recommendation.split()
    current_line = "  "
    for word in rec_words:
        if len(current_line) + len(word) + 1 > w:
            lines.append(current_line)
            current_line = "  " + word
        else:
            current_line += (" " if current_line.strip() else "") + word
    if current_line.strip():
        lines.append(current_line)
    lines.append("")

    lines.append("=" * w)

    return "\n".join(lines)
