# Meta-Validation — Does the Composite Score Track Human Judgement?

The composite stability score is a weighted combination of three sub-scores (semantic, factual, hallucination). The weights and the bounds between them were chosen from first-principles reasoning about what "matters" for financial extraction; they were NOT validated against a ground-truth human rating. Until that validation happens, the composite score is a plausible proxy, not a calibrated one.

This document specifies the meta-validation procedure: how to check that the composite tracks what an expert human reviewer thinks is "good" or "bad" output.

---

## The test

Take a sample of LLM outputs that have been scored by the benchmark, hand them to a domain-expert human rater, have them rate quality on a 1–5 Likert scale, and compute the correlation between the human rating and the composite score. The question: does a high composite correspond to output a human would judge as reliable?

**Target:** Pearson's r ≥ 0.6 between human rating and composite score. Below that, the composite's weights or sub-scores need to be revisited.

### Sample size

| Cohort | N |
|--------|---|
| Minimum for directional signal | 20 outputs |
| Adequate for a correlation CI tight enough to decide if r ≥ 0.6 | 50 outputs |
| Preferred | 100 outputs, stratified across the flag distribution |

Stratification: within the 100 outputs, draw roughly equal numbers from each flag bucket (green, yellow, red). This prevents the correlation from being dominated by the bulk of red examples if the model is bad, or green examples if it is good.

### Blinding

The human rater must NOT see the composite score during rating. Otherwise the rating is not independent of the thing being validated. The protocol:

1. Export `reports/meta_validation_sample.csv` with columns `[sample_id, fact_id, template, temperature, response_text, ground_truth, expected_unit]` — crucially NOT the composite or flag.
2. Rater reads each row, assigns a 1–5 rating plus a short free-text justification.
3. Separately, the benchmark's composite score for each row is pulled from `per_fact_report.csv`.
4. Join on `sample_id` and compute Pearson's r, Spearman's ρ, and the Bland-Altman style residual plot (human rating vs composite-implied rating).

### The 1–5 Likert anchors

Fix the rubric before rating starts. Drift between raters (or within a single rater over time) is the main source of noise.

| Score | Anchor |
|-------|--------|
| 5 | Response states the correct value clearly, with appropriate unit and period. No fabrication, no extraneous detail that contradicts the figure. |
| 4 | Response states the correct value but with minor issues (e.g. wrong unit abbreviation, slightly off period, extra commentary that is not factually wrong). |
| 3 | Response is in the right ballpark (within 10% of the truth) but has notable issues: ambiguous phrasing, wrong precision, mixed units. |
| 2 | Response is clearly wrong or unparseable, but the model "tried" — it produced a number in the right order of magnitude and correct semantic frame. |
| 1 | Fabrication: the response confidently states a value that is far from the truth, a number in the wrong unit entirely (e.g. SGD instead of percent), or a refusal dressed up as a number. |

The key boundary is 3/2: "ballpark wrong" is a different kind of failure from "confidently fabricated." A benchmark that collapses those into a single "hallucination" bucket will correlate poorly with human judgement because humans care about the distinction.

### The composite→Likert mapping (what to expect)

A rough prior, based on the definition of the composite:

| Composite range | Expected human rating |
|-----------------|-----------------------|
| 0.9 – 1.0 | 5 |
| 0.75 – 0.89 | 4 |
| 0.5 – 0.74 | 3 |
| 0.25 – 0.49 | 2 |
| 0.0 – 0.24 | 1 |

If the observed correlation is strong (r ≥ 0.7) and the mapping follows this pattern, the composite is well-calibrated. Common failure modes:

- **Low correlation + systematic lift on the human side** (humans rate responses higher than composite would predict): the benchmark is too harsh. Likely the hallucination tolerance is too tight or the factual weight is too high.
- **Low correlation + systematic lift on the composite side** (humans rate lower than composite predicts): the composite is giving credit for stable-but-wrong behaviour. Likely the hallucination weight is too low.
- **High scatter, no systematic bias**: the sub-scores are individually fine but the *weights* are wrong. Re-fit weights by regressing human rating on the three sub-scores (e.g. `rating ~ semantic + factual + (1 - hallucination_rate)`) and compare coefficients to the current `(0.30, 0.40, 0.30)`.

---

## What to do if r < 0.6

1. **Check each sub-score's correlation with human rating individually.** One of the three is probably dragging the composite down.
2. **Inspect the specific disagreements.** Pull the 10 worst-scoring-per-human examples and look for a shared failure mode. Common culprits: the extractor misparsing a valid answer, the semantic encoder giving high similarity to paraphrased fabrications, the tolerance letting through systematic order-of-magnitude errors.
3. **Consider adding a sub-score.** If human ratings depend on something the composite doesn't measure (e.g. presence of a confidence disclaimer, unit correctness at the lexical level), a fourth component may be needed.
4. **Re-fit the weights.** If the sub-scores individually correlate well but the composite doesn't, the weighted sum is suboptimal. Fit weights by maximum likelihood against the human ratings, then report the re-fit composite alongside the default.
5. **Document the new rubric.** Any change to weights or sub-scores MUST be reflected in `README.md` and `config.yaml`, with a dated note that explains the motivation.

---

## Inter-rater reliability

A single human rater is a single point of failure. If the budget allows, have two raters independently score the same 20 outputs and compute Cohen's weighted κ.

- κ ≥ 0.75: solid agreement; either rater can be used going forward.
- 0.5 ≤ κ < 0.75: moderate agreement; the rubric needs clarification before scaling up.
- κ < 0.5: the rubric is broken. Revise before spending more rater time.

Report both the intra-composite-to-human correlation AND the inter-rater κ. A high Pearson's r with low κ means the composite correlates with one rater's idiosyncratic judgement, not with generalisable human quality judgement.

---

## Suggested artefact

```
evaluation/
  META_VALIDATION.md             # this file
  meta_validation_sample.csv     # 50+ outputs, unscored by humans, scored by benchmark (hidden)
  human_ratings.csv              # sample_id, rater_id, rating_1_to_5, justification
  meta_validation_report.md      # Pearson's r, Spearman's ρ, κ, discussion
```

The `meta_validation_report.md` should at minimum state:
- Sample size and stratification.
- Rater(s) and their domain expertise.
- Pearson's r and its 95% CI.
- Spearman's ρ (robust against non-linear composite→rating mappings).
- Cohen's κ if multiple raters.
- A short qualitative section: what kinds of outputs did the composite and human agree on? Where did they disagree, and what does that imply?

---

## Why this is Priority 2, not Priority 0

Meta-validation is more valuable AFTER ground-truth verification (Priority 0) is complete, because right now a human rater cannot easily decide whether an output is "right" — the ground truth itself is preliminary. The order of operations:

1. Verify ground truth (sign-off per `ground_truth/PROVENANCE.md`).
2. Run the benchmark end-to-end.
3. Sample outputs stratified by composite flag.
4. Have a domain expert rate them blind.
5. Compute and report correlations.

Skipping step 1 and going straight to meta-validation produces a circular result: the human rater, unable to verify the benchmark's ground truth, ends up rating "does this look plausible" rather than "is this correct," which is a different and weaker signal.
