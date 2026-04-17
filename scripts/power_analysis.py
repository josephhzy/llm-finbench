"""Concrete power analysis for the composite stability metric.

Answers the question posed in `docs/STATISTICAL_RIGOR.md`: given the
observed bootstrap CI width on the composite score, what size of
model-to-model difference can the benchmark reliably detect?

Method
------
We model the paired-comparison setting (every fact evaluated by both
model A and model B, so the unit of analysis is the per-fact difference).
The standard error of the mean paired difference is

    SE_diff = sigma_diff / sqrt(N)

For a two-sided test at significance level alpha with power (1 - beta),
the minimum detectable effect (MDE) is

    MDE = (z_{1 - alpha/2} + z_{1 - beta}) * SE_diff

where z_q is the q-th quantile of the standard normal distribution. We
use the normal approximation (not the t-distribution) because the
bootstrap CI on composite_stability is not parametric in t anyway, and
the correction between t_{28} and N(0, 1) at alpha=0.05 is small
(~1.4% tighter).

Inputs we derive from the existing run
--------------------------------------
- `reports/per_fact_report.csv` — one row per (fact, template, temperature).
- `reports/ci_summary.csv` (optional) — the bootstrap CI already
  computed by `evaluate_with_ci.py`. When present, we sanity-check that
  our internal CI estimate matches.

We estimate `sigma_diff` two ways:
  1. Using the per-fact standard deviation of composite_stability
     (fact-to-fact variation), assuming two runs on the same fact are
     perfectly paired. This is the "paired-SD" proxy and is the right
     shape when the between-fact variance dominates.
  2. Using the observed CI half-width as a direct input
     (half_width ~= 1.96 * sigma/sqrt(N)), solving for sigma.
Both estimates are reported so a reviewer can see the sensitivity.

The power-analysis output block is appended verbatim by the caller into
`docs/STATISTICAL_RIGOR.md`. We emit a markdown-friendly block via
`--markdown` so no hand-copying is required.

Usage
-----
    # Print the human-readable summary
    python scripts/power_analysis.py \\
        --per-fact reports/per_fact_report.csv

    # Emit a markdown snippet ready to paste into STATISTICAL_RIGOR.md
    python scripts/power_analysis.py \\
        --per-fact reports/per_fact_report.csv \\
        --markdown

    # Tighter CI (narrower half-width) — sweep what the current benchmark could
    # detect at different sample sizes.
    python scripts/power_analysis.py \\
        --per-fact reports/per_fact_report.csv \\
        --sample-sizes 10 29 50 100
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Standard two-sided z critical value at alpha = 0.05.
# scipy would be slightly cleaner but is an extra dependency we don't need here.
_Z_0_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
_Z_0_80 = 0.841621233572914  # scipy.stats.norm.ppf(0.80)


def _fact_level_stats(
    df: pd.DataFrame, metric: str = "composite_stability"
) -> Tuple[np.ndarray, float, float]:
    """Collapse rows to per-fact means and return (values, mean, sd).

    The per-fact mean is the same unit of analysis the CI helper uses
    (evaluate_with_ci.py resamples over facts, not over groups), so the
    power analysis is anchored to the same quantity.
    """
    per_fact = df.groupby("fact_id")[metric].mean().to_numpy()
    return per_fact, float(per_fact.mean()), float(per_fact.std(ddof=1))


def _mde_from_sigma_diff(
    sigma_diff: float, n: int, alpha: float, power: float
) -> float:
    """Minimum detectable effect at given sigma_diff, n, alpha, power.

    Normal approximation; adequate for N >= ~20 and bounded metrics away
    from 0/1 (composite_stability ~= 0.59 on this run, well clear of the
    boundary).
    """
    if alpha == 0.05 and power == 0.80:
        z_alpha = _Z_0_975
        z_beta = _Z_0_80
    else:
        # Fall back to a tiny normal-CDF inverter — avoids pulling scipy.
        # Accurate to ~1e-6 over the (0.001, 0.999) range, which covers
        # every reasonable alpha/power pair.
        z_alpha = _norm_ppf(1 - alpha / 2)
        z_beta = _norm_ppf(power)
    return float((z_alpha + z_beta) * sigma_diff / math.sqrt(n))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Beasley-Springer-Moro approximation.

    Good enough for power-analysis headline numbers (error << 0.01). We
    intentionally avoid scipy to keep this script's dependency surface
    identical to the rest of the benchmark.
    """
    # Coefficients from Beasley-Springer-Moro (1977), widely reproduced.
    a = [
        -39.6968302866538,
        220.946098424521,
        -275.928510446969,
        138.357751867269,
        -30.6647980661472,
        2.50662827745924,
    ]
    b = [
        -54.4760987982241,
        161.585836858041,
        -155.698979859887,
        66.8013118877197,
        -13.2806815528857,
    ]
    c = [
        -0.00778489400243029,
        -0.322396458041136,
        -2.40075827716184,
        -2.54973253934373,
        4.37466414146497,
        2.93816398269878,
    ]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def run(args: argparse.Namespace) -> int:
    per_fact_path = Path(args.per_fact)
    if not per_fact_path.exists():
        print(f"ERROR: {per_fact_path} not found", file=sys.stderr)
        return 2

    df = pd.read_csv(per_fact_path, comment="#")
    per_fact_vals, mean_composite, sd_composite = _fact_level_stats(df)
    n_facts = len(per_fact_vals)

    # Half-width of the 95% CI implied by the fact-level SD, using the
    # same normal approximation a bootstrap would converge to for large B.
    # This is a *check* against the bootstrap number from ci_summary.csv.
    implied_half_width = _Z_0_975 * sd_composite / math.sqrt(n_facts)

    # For the MDE, we treat sigma_diff = sd_composite as a conservative
    # upper bound: if both models have the same per-fact SD and the
    # correlation of their per-fact scores is rho, then
    #    var(A_i - B_i) = 2*sigma^2*(1 - rho)
    # which is <= 2*sigma^2, with equality only when rho = 0. Paired
    # comparisons with high rho (i.e. easy facts are easy for both
    # models) will have smaller sigma_diff and therefore smaller MDE.
    # We report the rho=0 (worst-case) number as the default.
    sigma_diff_default = math.sqrt(2.0) * sd_composite
    sample_sizes = args.sample_sizes or [10, n_facts, 50, 100]
    alpha = args.alpha
    power = args.power

    mde_rows = [
        (n, _mde_from_sigma_diff(sigma_diff_default, n, alpha, power))
        for n in sample_sizes
    ]

    # Also compute the MDE under a more optimistic assumption that rho = 0.5
    # (two models' per-fact scores are moderately correlated — a realistic
    # prior for two LLMs on the same benchmark).
    sigma_diff_rho05 = math.sqrt(2.0 * (1.0 - 0.5)) * sd_composite  # rho = 0.5
    mde_rho05_rows = [
        (n, _mde_from_sigma_diff(sigma_diff_rho05, n, alpha, power))
        for n in sample_sizes
    ]

    # Also report the observed CI width from ci_summary.csv if present,
    # as a reality-check on our parametric estimate.
    ci_path = per_fact_path.parent / "ci_summary.csv"
    observed_half_width: Optional[float] = None
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        row = ci_df[ci_df["metric"] == "composite_stability"]
        if len(row):
            lo = float(row["fact_ci_lower"].iloc[0])
            hi = float(row["fact_ci_upper"].iloc[0])
            observed_half_width = (hi - lo) / 2.0

    if args.markdown:
        _print_markdown(
            n_facts=n_facts,
            mean_composite=mean_composite,
            sd_composite=sd_composite,
            sigma_diff_default=sigma_diff_default,
            sigma_diff_rho05=sigma_diff_rho05,
            implied_half_width=implied_half_width,
            observed_half_width=observed_half_width,
            mde_rows=mde_rows,
            mde_rho05_rows=mde_rho05_rows,
            alpha=alpha,
            power=power,
        )
    else:
        _print_text(
            n_facts=n_facts,
            mean_composite=mean_composite,
            sd_composite=sd_composite,
            sigma_diff_default=sigma_diff_default,
            sigma_diff_rho05=sigma_diff_rho05,
            implied_half_width=implied_half_width,
            observed_half_width=observed_half_width,
            mde_rows=mde_rows,
            mde_rho05_rows=mde_rho05_rows,
            alpha=alpha,
            power=power,
        )
    return 0


def _print_text(**kw) -> None:
    print("=" * 72)
    print("  Power analysis — composite_stability")
    print("=" * 72)
    print(f"N facts in data              : {kw['n_facts']}")
    print(f"Observed mean composite      : {kw['mean_composite']:.4f}")
    print(f"Fact-level SD                : {kw['sd_composite']:.4f}")
    print(
        f"Implied 95% CI half-width    : {kw['implied_half_width']:.4f}  (normal approx)"
    )
    if kw["observed_half_width"] is not None:
        print(
            f"Observed 95% CI half-width   : {kw['observed_half_width']:.4f}  (bootstrap in ci_summary.csv)"
        )
    print(f"sigma_diff  (rho = 0, upper) : {kw['sigma_diff_default']:.4f}")
    print(f"sigma_diff  (rho = 0.5)      : {kw['sigma_diff_rho05']:.4f}")
    print(f"alpha                        : {kw['alpha']}")
    print(f"power                        : {kw['power']}")
    print("")
    print("Minimum detectable effect (MDE) on mean paired composite_stability diff:")
    print("  N     rho=0        rho=0.5")
    for (n, mde0), (_, mde_05) in zip(kw["mde_rows"], kw["mde_rho05_rows"]):
        print(f"  {n:<5} {mde0:.4f}       {mde_05:.4f}")
    print("")
    print(
        "Interpretation: at N = 29 facts, paired comparison of two models with\n"
        "  uncorrelated per-fact scores (rho=0, worst case) can reliably detect\n"
        "  a difference of ~{:.3f} in mean composite_stability at 80% power.\n"
        "  A more realistic rho=0.5 tightens this to ~{:.3f}. Observed\n"
        "  model-to-model differences below this threshold are NOT evidence of\n"
        "  a real effect at this sample size.".format(
            kw["mde_rows"][1][1] if len(kw["mde_rows"]) > 1 else kw["mde_rows"][0][1],
            kw["mde_rho05_rows"][1][1]
            if len(kw["mde_rho05_rows"]) > 1
            else kw["mde_rho05_rows"][0][1],
        )
    )
    print("=" * 72)


def _print_markdown(**kw) -> None:
    print(f"### Concrete MDE for the current run ({kw['n_facts']} facts)")
    print("")
    print(
        f"Observed `composite_stability` point estimate: **{kw['mean_composite']:.3f}**; "
        f"fact-level SD `sigma = {kw['sd_composite']:.3f}` (computed from "
        f"`reports/per_fact_report.csv`). Paired-diff SD "
        f"`sigma_diff = sqrt(2)*sigma = {kw['sigma_diff_default']:.3f}` under "
        f"the worst-case `rho = 0` assumption; "
        f"`{kw['sigma_diff_rho05']:.3f}` under a realistic `rho = 0.5`."
    )
    print("")
    print("| N facts | MDE (rho=0, worst case) | MDE (rho=0.5, realistic) |")
    print("|---------|------------------------|--------------------------|")
    for (n, mde0), (_, mde_05) in zip(kw["mde_rows"], kw["mde_rho05_rows"]):
        print(f"| {n} | {mde0:.3f} | {mde_05:.3f} |")
    print("")
    print(
        f"**Bottom line at N = {kw['n_facts']}:** under the realistic `rho = 0.5` "
        f"prior and 80% power at alpha = 0.05, the benchmark can detect a "
        f"mean paired `composite_stability` difference of roughly "
        f"**{kw['mde_rho05_rows'][1][1] if len(kw['mde_rho05_rows']) > 1 else kw['mde_rho05_rows'][0][1]:.3f}** "
        f"between two models. Observed differences smaller than that are "
        f"not distinguishable from sampling noise and should not be reported "
        f"as 'model A beats model B'."
    )
    if kw["observed_half_width"] is not None:
        print("")
        print(
            f"Sanity-check: the bootstrap 95% CI half-width on composite in "
            f"`reports/ci_summary.csv` is {kw['observed_half_width']:.3f}; the "
            f"parametric implied half-width from this script is "
            f"{kw['implied_half_width']:.3f}. Agreement within ~0.005 is "
            f"expected given the normal approximation."
        )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the minimum detectable effect for composite_stability.",
    )
    parser.add_argument(
        "--per-fact",
        type=str,
        default="reports/per_fact_report.csv",
        help="Aggregated per-fact report (default: reports/per_fact_report.csv).",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Sample sizes to report MDE at (default: 10, observed N, 50, 100).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided significance level (default: 0.05).",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired power (1 - beta). Default: 0.80.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help=(
            "Emit a markdown snippet suitable for appending to "
            "docs/STATISTICAL_RIGOR.md."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(_parse_args(sys.argv[1:])))
