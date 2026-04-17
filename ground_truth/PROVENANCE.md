# Ground Truth Provenance

**Status: PRELIMINARY — pending primary-source verification.**

The 29 facts across 5 companies stored in `facts.json` are a preliminary dataset. They were compiled from publicly available FY2024 annual reports of SGX-listed companies, but the dataset has NOT yet been through the second-person verification pass described below. Any benchmark result that cites these facts inherits that caveat — an evaluation is only as reliable as its ground truth.

If you are reviewing this project, read this file before interpreting any metric produced from `facts.json`.

---

## What a verified fact looks like

Every fact should be traceable to an exact page in an exact source document, with a verbatim quote that includes the figure.

### Expected fields (schema contract)

```json
{
  "id": "dbs_fy2024_nim",
  "company": "DBS",
  "metric": "Net Interest Margin",
  "metric_abbreviation": "NIM",
  "period": "FY2024",
  "value": 2.13,
  "unit": "percent",
  "currency": null,
  "source": "DBS Group Holdings FY2024 Annual Report",
  "page": 47,
  "context": "Net interest margin was 2.13% for FY2024.",
  "difficulty": "easy",
  "category": "profitability"
}
```

### Required for "verified" status

| Field | Requirement |
|-------|------------|
| `source` | Exact report title, including the fiscal year and publishing entity. Must match the cover page of the PDF. |
| `page` | Non-null integer page number in the source PDF. If the figure appears in multiple places, cite the primary statement, not the management commentary. |
| `context` | Verbatim quote from the page that contains the figure. Must include the figure itself (not just surrounding prose). |
| `value` | Numeric value as reported, in the unit declared in `unit`. Monetary values are stored at their declared scale (e.g. `22297.0` with `unit: "sgd_millions"` encodes SGD 22.297 billion, NOT `22_297_000_000`). |
| `unit` | One of `percent`, `sgd_millions`, `sgd_billions`, `sgd`, `ratio`, `millions`, `count`. |
| `difficulty` | `easy` (explicitly stated in a headline table), `medium` (inferred from a sub-table or short calculation), `hard` (cross-reference or multi-step). |

### Primary source format expected

- **Annual report PDFs** published directly by the issuing company (investor-relations section of the corporate website). Example: `https://www.dbs.com/investor` → Annual Report 2024 → PDF download.
- **Never** a summary article, wire service, or analyst note. Press-release numbers sometimes differ from audited-financial-statement numbers (rounding, restatements, segment reclassification); the benchmark requires the audited figure.
- **Dated** — the exact FY covered must match the `period` field, e.g. `FY2024` with `period: "FY2024"`.

---

## Verification workflow (required before a fact is promoted from preliminary to verified)

1. **Primary extraction.** Download the official annual-report PDF from the issuing company's investor-relations portal. Hash the PDF (`sha256`) and record the hash and download date in a verification log. (Log file: `ground_truth/verification_log.md`, to be created on first run.)
2. **Locate the figure.** Find the page where the metric appears. Prefer the statement of comprehensive income / balance sheet / pillar 3 disclosure over the management commentary, because these are audited line items.
3. **Quote verbatim.** Copy the sentence or table cell containing the figure into the `context` field. Do not paraphrase. If the figure is in a table, copy the column header and the row label (e.g. `"Net interest margin (FY2024): 2.13%"`).
4. **Record page.** Set the `page` field to the integer page number as rendered in the PDF viewer (not the print-page offset if they differ).
5. **Second-person review.** A reviewer who did not perform the primary extraction opens the same PDF, navigates to the cited page, and confirms that the quoted context and numeric value match. The reviewer signs off with initials and date in `verification_log.md`.
6. **Sign-off to production.** Only facts that have passed steps 1–5 are used in a production evaluation run. Preliminary facts may be used in quick-mode iteration but must be visibly flagged in any report that cites them.

### What counts as a blocking disagreement

- Value mismatch beyond 0.01 in the declared unit → blocker; reconcile before merging.
- Missing page number → blocker; cannot be called "verified."
- Context quote that does not literally contain the figure → blocker; the quote must be a sufficient citation on its own.
- `source` that points to a press release or analyst note instead of the audited report → blocker.

---

## Current status of the dataset (snapshot: 2026-04)

- `facts.json` contains 29 facts across 5 companies (DBS, OCBC, UOB, Singtel, CapitaLand).
- Values are from FY2024 reports as released in 2025.
- `page` is `null` for every current entry → **no fact has a confirmed page citation yet**.
- `context` is populated but is a plausible-sounding phrasing, not a verbatim quote from the PDF → **no fact has been verified against the source PDF**.
- Second-person review has not been performed on any fact.
- `verification_log.md` does not exist yet.

Until the workflow above is run end-to-end on all 29 facts, every published metric carries a "preliminary ground truth" disclaimer in `README.md` and in `reports/summary.txt`.

### Automated validator (`scripts/validate_ground_truth.py`)

A schema-contract validator runs against `facts.json` on every commit. It checks presence and non-null-ness of REQUIRED fields (`id`, `company`, `metric`, `period`, `value`, `unit`, `source`, `difficulty`, `category`), RECOMMENDED fields (`metric_abbreviation`, `context`), and provenance fields (`page`). It also runs a weak heuristic to flag facts whose `context` string does not contain the declared numeric value — a necessary (not sufficient) condition for the context to be a verbatim source quote.

Validator last run 2026-04-16: 29/29 facts have all required fields; 9/29 are missing the optional `metric_abbreviation` (a recommended-but-non-blocking field); 0/29 have PDF page citations; 27/29 have a `context` string containing the declared numeric value. The two with a `context`-value mismatch are `singtel_fy2024_ebitda` (value `3597.0` sgd_millions rounded from context "SGD 3,596.9 million") and `singtel_fy2024_dps` (value `0.15` sgd vs context "15.00 Singapore cents" — same figure in cents, not a data error). Both are expected unit/rounding differences, not broken records, and are left as-is so the validator surfaces them on future runs rather than silently masking the discrepancy.

### Spot-check: `capitaland_fy2024_aum` (modal 676 vs truth 136)

`reports/per_fact_report.csv` shows the `direct_extraction` template returning a modal 676 across all five temperatures for the CapitaLand AUM fact, while the `contextual_extraction` template returns the correct 136 deterministically and `qualitative` / `comparative` return values clustered in the 109–169 range. Ground truth is `136.0 sgd_billions` with context "Assets under management were SGD 136 billion as at 31 December 2024." — the figure reported publicly by CapitaLand Investment for FY2024, confirmed against company investor-relations communications. The 676 output is therefore a model hallucination (roughly 5× the truth, order-of-magnitude consistent with the model conflating CapitaLand group total real-estate AUM with a different scale or period), NOT a unit bug in `facts.json`. The `unit` field remains `sgd_billions`; no change to the dataset. This row is a useful example for the hallucination discussion: the same model, same fact, same tolerance — the failure mode lives entirely in the prompt template.

Validator exit code is 0 iff every fact passes the REQUIRED check, so this can be wired into CI as a blocking gate without adding any runtime dependencies beyond the standard library and the facts file itself.

---

## Known gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No PDF page citations (`page: null` on every fact) | A reviewer cannot audit a fact without re-searching the PDF from scratch. | High — blocks verification sign-off. |
| Context strings are plausible paraphrases, not verbatim quotes | A reviewer reading the context sees a fluent sentence, not the exact text from the source, and cannot grep the PDF for it. | High — blocks verification sign-off. |
| Only 5 companies, all Singapore-listed | Generalisation beyond the SGX market is untested. Any claim about "LLMs on financial data broadly" is an overreach of the evidence. | Medium — expand to 10+ companies to strengthen the sample. |
| Only FY2024 reporting period | Year-over-year drift (metric redefinition, restatement) is not tested. | Medium — add FY2023 comparatives. |
| No facts that require cross-referencing two sections of the same report | The `hard` difficulty bucket is small. Models that can handle easy/medium metrics may still fail on cross-reference reasoning. | Low — enrich the `hard` bucket after the dataset is verified. |
| No inter-rater reliability on `difficulty` or `category` labels | Labels are assigned by a single author; categorical reliability is unknown. | Low — Cohen's kappa with a second labeller once the pool of annotators grows. |

---

## How to update this file when the dataset changes

- When a fact is verified (full workflow completed), update the fact's `page` and `context` in `facts.json` and append an entry to `ground_truth/verification_log.md` with PDF hash, reviewer initials, and date.
- Update the "Current status of the dataset" snapshot in this file to reflect the verification count (e.g. `22 of 29 facts verified`) and the snapshot date.
- Do NOT delete the "Known gaps" table — move gaps to a "Resolved" subsection so the audit trail is preserved.
- When adding new companies, expand the scope note in the snapshot and add a row to the `ground_truth/README.md` companies table.
