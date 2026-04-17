# Statistical Rigour — How to Report and Interpret Results

This document specifies the statistical methodology the benchmark *must* follow to produce defensible conclusions. Point estimates without uncertainty quantification are not acceptable output of this harness.

---

## TL;DR

- **Report `composite = X.XX ± Y.YY (95% CI)`, never a bare point estimate.** Bootstrap the CI across facts.
- **Sample size is N = 29 facts across 5 companies.** Conclusions are bounded by that. A claim like "Model A is better than Model B" requires non-overlapping 95% CIs on the relevant metric, not just `mean_A > mean_B`.
- **Number of repetitions per condition (default: 10) controls the within-condition variance.** Pilot-level runs (N = 3) produce noisy factual-consistency scores — prefer N = 10 for headline numbers, and report the N you used.
- **Stratify by company.** Averaging 7 DBS facts with 6 CapitaLand facts without stratification hides company-level variance. Report `by_company.csv` alongside the headline.

---

## Why bootstrap confidence intervals

The composite stability score is a mean over `n_facts × n_templates × n_temperatures` evaluation groups. Under classical parametric assumptions, the CI on a mean is `mean ± 1.96 × sd / sqrt(N)`. That is misleading here because:

1. **The groups are not i.i.d.** Facts within a company share sources (same annual report) and therefore share model biases. A model that has memorised the DBS FY2024 report performs uniformly well on all 7 DBS facts.
2. **The composite is a non-linear combination** of three sub-scores. Parametric CIs on the composite are not the straightforward propagation of CIs on the sub-scores.
3. **The distribution is non-normal.** The flag distribution `[green, yellow, red]` is bimodal in practice (models tend to be either confident-correct or confident-wrong), violating the normality assumption behind the Student-t CI.

The non-parametric bootstrap sidesteps all three. We resample facts with replacement, recompute the mean composite, and report the 2.5th and 97.5th percentiles of the resampled means.

### The method

```
Input: per_fact_report.csv with one row per (fact × template × temperature) group
B = 10_000        # bootstrap iterations
alpha = 0.05      # 95% CI

bootstrap_means = []
for b in range(B):
    sample = per_fact_report.sample(n=len(per_fact_report), replace=True)
    bootstrap_means.append(sample["composite_stability"].mean())

ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
point    = per_fact_report["composite_stability"].mean()

report = f"composite = {point:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
```

### Fact-level vs group-level resampling

There are two defensible units to resample:
- **Groups** (`fact × template × temperature`): easier to implement, tighter CIs because N is larger (up to 29 × 4 × 5 = 580).
- **Facts**: more honest, because template and temperature are experimental conditions applied within a fact, not independent units of observation.

Our recommendation: **report both**. The group-level CI tells you how much the metric varies across your experimental conditions; the fact-level CI tells you how much it varies across the population of financial facts you sampled. A large gap between the two is itself information — it says "this model's quality depends more on which fact you ask than on how you ask it" or vice versa.

### Why not assume normality and use `1.96 × sd / sqrt(N)`?

1. The composite is bounded in [0, 1]. Normality breaks down near the endpoints.
2. Bimodal distributions at the per-fact level violate the CLT's "smooth enough" assumption.
3. With N = 29, the t-distribution with 28 d.f. has heavier tails than the normal; using 1.96 instead of 2.048 is a small bias toward over-tight CIs.
4. Bootstrap is cheap (10,000 resamples of 29 values takes < 1 s on a laptop) and assumption-free. There is no reason to use a parametric CI here.

---

## Sample-size sufficiency: what does N = 29 buy us?

A rough power-analysis sketch:

Suppose two models' true mean composite scores differ by `δ`, and the per-fact standard deviation of the composite within a model is `σ`. With paired facts (every fact evaluated by both models) and N facts:

```
paired standard error ≈ σ_diff / sqrt(N)
```

where `σ_diff` is the standard deviation of the paired differences. Empirically, `σ_diff ≈ 0.15` on the current dataset (a mix of stable and unstable facts). So:

| N  | Minimum detectable difference at 80% power (α = 0.05) |
|----|-------------------------------------------------------|
| 10 | ~0.13 |
| 29 | ~0.080 |
| 50 | ~0.06 |
| 100 | ~0.042 |

Interpretation: with 29 facts and paired sampling, the benchmark can reliably detect a composite-score difference of roughly **0.080**. Differences smaller than that between two models are not statistically distinguishable at this sample size, even if the point estimates look different.

**Actionable implication for a future expansion to 10 companies (~58 facts):** the minimum detectable difference drops to roughly 0.056. That is a better position for making model-comparison claims, but the headline constraint is clear: *small observed differences are not evidence of a real effect*.

### What a power-analysis script should report

A future `power_analysis.py` should take:
- The observed `σ_diff` from a pilot run, or an analyst's prior on it.
- A target effect size (e.g. "I want to detect a 0.05 difference").
- A desired power (e.g. 0.80) and α (e.g. 0.05).

And output: the minimum N (facts) required. This tells a reader whether a 29-fact study is adequate for the question being asked.

### Concrete MDE for the current run (29 facts)

`scripts/power_analysis.py` runs this calculation against `reports/per_fact_report.csv` and produces the numbers below. Unlike the `σ_diff ≈ 0.15` prior used in the table earlier in this document, these estimates are derived directly from the observed per-fact spread of the current `gpt-5-nano` run, not a hand-picked guess.

Observed `composite_stability` point estimate: **0.594**; fact-level SD `σ = 0.048` (computed from `reports/per_fact_report.csv`). Paired-diff SD `σ_diff = sqrt(2)·σ = 0.068` under the worst-case `ρ = 0` (two models give uncorrelated per-fact scores); `0.048` under a realistic `ρ = 0.5` (two models agree on which facts are easy and which are hard, but differ on the margin).

| N facts | MDE (ρ=0, worst case) | MDE (ρ=0.5, realistic) |
|---------|------------------------|--------------------------|
| 10 | 0.061 | 0.043 |
| 29 | 0.036 | 0.025 |
| 50 | 0.027 | 0.019 |
| 100 | 0.019 | 0.014 |

**Bottom line at N = 29:** under the realistic `ρ = 0.5` prior and 80% power at α = 0.05, the benchmark can detect a mean paired `composite_stability` difference of roughly **0.025** between two models. Observed differences smaller than that are not distinguishable from sampling noise and should not be reported as "model A beats model B." The `0.080` figure earlier in this document was a deliberately conservative prior from a pilot with heavier per-fact noise; the observed data on the `gpt-5-nano` run is tighter, which is good news — but it means a claim like "model B is 0.03 better than gpt-5-nano" would still sit inside the noise envelope at N = 29 and would need a bigger sample to be defensible.

Sanity-check: the bootstrap 95% CI half-width on composite in `reports/ci_summary.csv` is 0.018; the parametric implied half-width from `power_analysis.py` is also 0.018. Agreement within 0.005 is expected given the normal approximation.

---

## What "reporting a CI" means in each artefact

| Artefact | What to add | Example |
|----------|-------------|---------|
| `reports/summary.txt` | Replace `Mean composite stability:   0.XXXX` with `Mean composite stability:   0.XXXX (95% CI: [X.XX, X.XX], N=29 facts)` | `Mean composite stability:   0.594 (95% CI: [0.576, 0.612], N=29 facts)` — these are the actual numbers from the partial `gpt-5-nano` run already in `reports/`. |
| `reports/by_company.csv` | Add `composite_ci_lower`, `composite_ci_upper` columns | — |
| `reports/by_temperature.csv` | Same. Without the CI, a "0.72 at T=0 vs 0.53 at T=1" comparison is not defensible. | — |
| `src/comparison.py` A-vs-B output | Compute the **paired bootstrap** of the difference in composites across facts and report the 95% CI of the difference. If it contains 0, the comparison is not informative. | `composite_diff = +0.04 (95% CI: [-0.01, +0.09]) — inconclusive at N=29.` |

---

## The `evaluate_with_ci.py` helper

`evaluate_with_ci.py` (shipped alongside this doc) wraps an existing `per_fact_report.csv` with bootstrap CIs using numpy. Usage:

```bash
# After a run completes and reports are generated:
python evaluate_with_ci.py reports/per_fact_report.csv --bootstrap 10000 --metric composite_stability
```

It prints:

```
composite_stability
  point estimate:   0.XXX
  95% CI:           [X.XX, X.XX]   (method: percentile bootstrap over facts, B=10000)
  N facts:          29
  N groups:         535
```

And optionally emits a CSV (`reports/ci_summary.csv`) with one row per metric and columns `[metric, point, ci_lower, ci_upper, n_facts, n_groups]`.

The script does NOT re-run the LLM. It only re-aggregates the scores that are already in `per_fact_report.csv`. It is cheap and should be part of every report publication.

---

## Common mistakes this doc is trying to prevent

- **Reporting "`0.75` composite" without any uncertainty.** A single decimal place of precision implies accuracy we have not earned at N = 29.
- **Comparing two models on point estimates only.** `mean_A = 0.75, mean_B = 0.72` with overlapping CIs is a null result, not "A beats B."
- **Treating `by_company.csv` as if facts within a company were independent.** They share a source document; their within-company variance is smaller than their across-company variance.
- **Under-reporting `N` in the summary.** The composite number is meaningless without the sample size.
- **Interpreting `hallucination_rate = 0.65` as "the model hallucinates 65% of the time on financial data."** It is "65% of extractions deviated from the current preliminary ground truth by more than 5% relative tolerance, on this 29-fact subset of 5 SGX names, using 4 prompt templates, at this temperature sweep." The qualifiers matter.
