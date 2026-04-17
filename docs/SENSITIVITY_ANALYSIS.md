# Sensitivity Analysis — Tolerance Sweep Methodology

The hallucination rate is the fraction of extracted values that deviate from ground truth beyond a **relative tolerance**. The default tolerance is 5%. That number is a product of convenience, not truth — for a regulatory disclosure, 5% is ludicrously loose; for a first-pass screening of market summaries, it is fine. Any published hallucination rate therefore needs a tolerance sweep to be interpretable.

This doc specifies the sweep methodology, the interpretation rules, and how to run it against existing results without re-calling the LLM.

---

## The tolerance ladder

Sweep the relative tolerance across `{0.5%, 1%, 2%, 5%, 10%}`. The 0.5% floor is the rounding threshold for a well-prepared financial figure (SGD 22.297 B, not SGD 22.3 B). The 10% ceiling is the upper bound of "roughly right" that would still be considered a hallucination by most reasonable readers. Tolerances outside this range are either too tight to be achievable at fp-arithmetic precision or too loose to carry audit meaning.

For each tolerance level, recompute:

1. **Per-group hallucination rate** (`count_wrong / total` within each `(fact, template, temperature)` group).
2. **Aggregate hallucination rate** (mean across all groups, and/or mean across facts — follow the statistical-rigour policy on fact-level vs group-level aggregation).
3. **Flag distribution** (green / yellow / red counts).
4. **Composite stability** (recomputed — hallucination rate is one of the three weighted terms).

---

## Expected shape of the curve

Hallucination rate is monotonically non-increasing in tolerance: a stricter tolerance cannot produce *fewer* hallucinations than a looser one. The interesting quantity is the *shape* of the decay:

- **Sharp drop** between 0.5% and 5% (e.g. 0.90 → 0.30) means the model is making small numerical errors — rounding, digit-order-of-magnitude confusion. These are often recoverable with better prompting or post-processing.
- **Shallow decay** (0.90 → 0.80 → 0.70) means the model is producing values that are qualitatively wrong (off by 10x, wrong unit, fabricated figure). Loosening the tolerance does not rescue them.
- **Step change near 1%** is characteristic of models that report `X.XX%` where the ground truth is `X.X%` — pure rounding artefact.

Report the curve as a 5-column table and a plot. Do not report a single hallucination rate without the tolerance context.

### Example output table

```
tolerance   hallucination_rate   composite   n_red   n_yellow   n_green
0.005       0.XX                 0.XX        XX      XX         XX
0.010       0.XX                 0.XX        XX      XX         XX
0.020       0.XX                 0.XX        XX      XX         XX
0.050       0.XX                 0.XX        XX      XX         XX
0.100       0.XX                 0.XX        XX      XX         XX
```

Values are intentionally left as `0.XX`; populate from a real run via `rescore_at_tolerance.py`.

---

## Two ways to run the sweep

### Option A — full recomputation from raw results (preferred)

If `results/{run_id}/results.json` exists, the scorer can be re-run end-to-end with a different `hallucination_tolerance`:

```python
from src.scorer import score_fact
# for each group in results.json:
scores = score_fact(
    fact_id=...,
    template=...,
    temperature=...,
    response_texts=group["responses"],
    ground_truth_value=...,
    expected_unit=...,
    embedding_model_name="all-MiniLM-L6-v2",
    hallucination_tolerance=tolerance,   # <-- swept
    composite_weights=config.scoring.composite_weights,
)
```

This is the correct way to produce tolerance-swept hallucination rates because it operates on the **full** extracted-values list per group, not just the modal value. `rescore_at_tolerance.py --results-json results/<run_id>/results.json --out reports/tolerance_sensitivity.csv` is the intended one-line entry point for this path.

### Option B — modal-only approximation from `per_fact_report.csv`

If `results.json` has been lost (or was never persisted in this run, as happens in the current checkpointed state of the repo), the only available signal is the `modal_value` and `ground_truth` per group. A *modal-only* approximation recomputes, for each group, whether the modal value itself passes the tolerance test.

```
modal_hallucination(tolerance) = 1 if |modal_value - gt| / |gt| > tolerance else 0
```

This is a weaker sensitivity curve — it answers "at tolerance t, does the most-common answer count as a hallucination?" rather than the fleet-wide "what fraction of all 10 runs hallucinated?". But it is still directionally useful and, importantly, can be computed offline from the already-saved `per_fact_report.csv`.

`rescore_at_tolerance.py --csv reports/per_fact_report.csv --out reports/tolerance_sensitivity.csv` uses this path and labels its output unambiguously as `modal_hallucination_fraction`, not `hallucination_rate`, to avoid conflation.

### When to use which

| Situation | Path |
|-----------|------|
| Fresh run, raw responses persisted to `results/<run_id>/results.json` | Option A (full recomputation) |
| Archived run, only CSV artefacts survived | Option B (modal-only) |
| Comparing two models for a public claim | Option A — modal-only is not defensible for a headline number |

---

## What "done" looks like

- A committed `reports/tolerance_sensitivity.csv` for the most recent run.
- A short paragraph in the README *Hallucination Rate* section pointing to the curve: "At the default 5% tolerance, hallucination rate is X. At 1% it is Y. At 10% it is Z. The composite score shifts by roughly W across this range."
- The tolerance parameter in `config.yaml` is explicitly justified (e.g. "5% chosen to match typical analyst-report precision; audit-grade use requires 0.5%, see `docs/SENSITIVITY_ANALYSIS.md`").

---

## Common mistakes this doc is trying to prevent

- **Reporting one hallucination rate as the hallucination rate.** It is "the hallucination rate at 5% tolerance." Drop the qualifier and you mislead.
- **Treating tolerance as a free hyperparameter to tune for a favourable number.** Pick the tolerance that matches the downstream use-case, then report the sweep to show robustness, not to cherry-pick.
- **Averaging the curve into a single "tolerance-integrated hallucination rate."** That number has no operational meaning — different applications have different tolerance requirements. Report the curve, not the integral.
- **Computing the sweep only at the aggregate level.** Also do it per-fact, because some facts are rounding-prone (percent metrics reported at one decimal place) while others are order-of-magnitude-prone (monetary values in millions vs billions).
