"""Re-score semantic consistency with FinBERT and compare against MiniLM.

The README currently caveats that `all-MiniLM-L6-v2` is a general-purpose
encoder, and that a financial-domain model (e.g. FinBERT) may produce
materially different semantic-consistency scores. This script closes that
gap by recomputing the semantic metric on the SAME response texts, using
the FinBERT encoder, and reporting the delta per fact.

What this script does NOT do
----------------------------
- Make any new LLM calls. It operates on response texts that were already
  stored during a benchmark run.
- Change the MiniLM semantic scores. The original `per_fact_report.csv`
  is left untouched. This script writes a sibling CSV
  (`reports/semantic_finbert.csv`) whose rows are the per-fact delta.

Input requirements
------------------
1. A results.json produced by `src/engine.py` that preserves per-group
   response_texts. Path: `results/<run_id>/results.json`.
2. The original MiniLM semantic scores, already in
   `reports/per_fact_report.csv` (the aggregated report).

If results.json is missing — which is the case for the historical runs
in this repo — the script supports a fallback `--from-per-fact-csv` mode
that reconstructs a per-fact paraphrase corpus from the ground-truth
context plus the modal extracted value in per_fact_report.csv. That mode
does NOT produce the same numbers as re-scoring real LLM outputs; it is a
code-path sanity check and is labelled as such in the output CSV. The
preferred path remains re-scoring real responses from a run that
persisted results.json.

Output
------
- `reports/semantic_finbert.csv` — one row per fact, columns:
    fact_id, miniLM_score, finbert_score, delta
- Console summary: mean MiniLM, mean FinBERT, Pearson correlation,
  count of facts where FinBERT disagrees by > 0.1.

Usage
-----
    python scripts/rescore_semantic_finbert.py \\
        --results-json results/20260401_040014_891022/results.json \\
        --per-fact reports/per_fact_report.csv \\
        --out reports/semantic_finbert.csv

    # Alternative FinBERT variant:
    python scripts/rescore_semantic_finbert.py \\
        --results-json ... --per-fact ... --out ... \\
        --finbert-model yiyanghkust/finbert-tone
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make `src/` importable when this file is run directly from the repo root
# (i.e. `python scripts/rescore_semantic_finbert.py ...`). Without this, the
# `from src.scorer import ...` below fails with ModuleNotFoundError.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

# The FinBERT models we support. Defaulting to ProsusAI/finbert because it is
# pre-trained on Reuters TRC2 + Financial PhraseBank, which is closer to
# annual-report prose than finbert-tone (which is fine-tuned for sentiment).
_DEFAULT_FINBERT = "ProsusAI/finbert"
_SUPPORTED_FINBERT = {
    "ProsusAI/finbert",
    "yiyanghkust/finbert-tone",
}


def _synth_response_groups_from_per_fact_csv(
    per_fact_csv: Path,
    facts_json: Path,
) -> List[Dict[str, Any]]:
    """Fallback: build a per-fact paraphrase corpus when results.json is missing.

    This mode is a code-path sanity check. It does NOT reproduce the
    actual LLM responses from the benchmark run; it constructs a
    deterministic synthetic corpus per (fact, template) using:

        1. The fact's ground-truth `context` quote from facts.json
           (used as the 'correct' paraphrase).
        2. The `modal_value` recorded in per_fact_report.csv for each
           (fact, template, temperature) row (used as the 'actually-
           extracted' paraphrase when the model got it wrong).
        3. A fixed set of template-specific lexical variants that mirror
           the 4 prompt templates (direct, contextual, comparative,
           qualitative).

    The output has the same shape as _load_results_json would return, so
    the downstream scoring code is unchanged. A sentinel flag is set on
    each group so the top-level summary can label the output as synthetic.

    Why even offer this: the honest re-score needs raw responses, but we
    still want a single-command way to exercise the FinBERT code path and
    get directional numbers so a reviewer can see the MiniLM-vs-FinBERT
    delta without re-running the benchmark. The synthetic numbers are
    NOT the benchmark's semantic consistency; they are the semantic
    consistency of a controlled paraphrase set that is structured
    similarly to what an LLM would produce.
    """
    df = pd.read_csv(per_fact_csv, comment="#")
    need = {"fact_id", "modal_value", "template", "temperature", "semantic_score"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(
            f"{per_fact_csv} is missing expected columns for synthetic "
            f"reconstruction: {sorted(missing)}."
        )

    with facts_json.open("r", encoding="utf-8") as fh:
        facts_raw = json.load(fh)
    facts_list = (
        facts_raw.get("facts", facts_raw)
        if isinstance(facts_raw, dict)
        else facts_raw
    )
    facts_lookup: Dict[str, Dict[str, Any]] = {f["id"]: f for f in facts_list}

    # Phrase templates, one per prompt-template we ran. Keep them short and
    # financial-report-flavoured so both MiniLM and FinBERT see realistic
    # prose. The %s placeholder is filled with the modal value + unit.
    per_template_phrases: Dict[str, List[str]] = {
        "direct_extraction": [
            "The reported value is %s.",
            "Per the annual report, it is %s.",
            "The figure stands at %s.",
            "Reported value: %s.",
            "%s for the fiscal year.",
        ],
        "contextual_extraction": [
            "Based on the provided context, the value is %s.",
            "From the passage, the reported figure is %s.",
            "The context indicates a value of %s.",
            "According to the excerpt, it is %s.",
            "The passage states the value as %s.",
        ],
        "comparative": [
            "Compared to the prior year, the value is %s.",
            "Year-over-year, this metric now reads %s.",
            "Relative to FY2023, it stands at %s.",
            "It moved to %s over the period.",
            "The year-on-year figure is %s.",
        ],
        "qualitative": [
            "The company has reported a value of %s.",
            "Management commentary suggests a figure of %s.",
            "The metric reflects a level of %s.",
            "Overall, performance is consistent with %s.",
            "Broadly, the figure is %s.",
        ],
    }

    groups: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        fact = facts_lookup.get(row["fact_id"])
        if fact is None:
            continue
        template = row.get("template", "unknown")
        phrases = per_template_phrases.get(
            template, per_template_phrases["direct_extraction"]
        )
        modal = row.get("modal_value")
        unit = fact.get("unit") or ""
        # Render the modal value as a string the encoders can parse.
        if modal is None or (isinstance(modal, float) and np.isnan(modal)):
            rendered = "the reported figure"
        else:
            try:
                modal_f = float(modal)
                # Match the surface form the encoders are likely to see:
                # percents as "x.xx%", monetary millions as "SGD x,xxx million",
                # everything else as the plain number.
                if unit == "percent":
                    rendered = f"{modal_f:.2f}%"
                elif unit.endswith("_millions"):
                    cur = unit.split("_")[0].upper()
                    rendered = f"{cur} {modal_f:,.0f} million"
                elif unit.endswith("_billions"):
                    cur = unit.split("_")[0].upper()
                    rendered = f"{cur} {modal_f:.2f} billion"
                else:
                    rendered = f"{modal_f:g}"
            except (TypeError, ValueError):
                rendered = str(modal)

        # 5 paraphrases — matches the typical n_runs per condition scale.
        response_texts = [p % rendered for p in phrases]
        groups.append(
            {
                "fact_id": row["fact_id"],
                "template": template,
                "temperature": float(row.get("temperature", 0.0)),
                "response_texts": response_texts,
                "_synthetic": True,
            }
        )
    if not groups:
        raise SystemExit(
            "Synthetic reconstruction produced no groups — check that "
            "per_fact_report.csv rows reference fact_ids present in facts.json."
        )
    return groups


def _load_results_json(path: Path) -> List[Dict[str, Any]]:
    """Load the per-group response records from a run's results.json.

    The engine writes a list of group dicts, each with a `fact_id`,
    `template_name`, `temperature`, and a list of `raw_response` strings
    (or a `response_texts` list, depending on engine version).

    Returns a list of group dicts with a normalised key set:
      {"fact_id": str, "template": str, "temperature": float,
       "response_texts": List[str]}
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Engine may write either a bare list or an object with "results" / "groups".
    if isinstance(data, list):
        raw_groups = data
    elif isinstance(data, dict):
        raw_groups = (
            data.get("results")
            or data.get("groups")
            or data.get("call_records")
            or []
        )
    else:
        raise SystemExit(
            f"{path} has unexpected top-level type {type(data).__name__}; "
            "expected a list or an object with 'results'/'groups'/'call_records'."
        )

    if not raw_groups:
        raise SystemExit(
            f"{path} contained no records. Make sure you are pointing at a "
            "results.json produced by a full evaluation run, not just the "
            "config.json that is always written at startup."
        )

    # Case A: already grouped by (fact, template, temperature), with a
    # `response_texts` list per group. Engine writes this when it persists
    # grouped results.
    first = raw_groups[0]
    if "response_texts" in first:
        return [
            {
                "fact_id": g.get("fact_id") or g.get("id"),
                "template": g.get("template")
                or g.get("template_name", "unknown"),
                "temperature": float(g.get("temperature", 0.0)),
                "response_texts": list(g["response_texts"]),
            }
            for g in raw_groups
            if g.get("fact_id") or g.get("id")
        ]

    # Case B: flat list of call records — each record has `raw_response`
    # plus the grouping keys. We re-group on the fly.
    if "raw_response" in first:
        from collections import defaultdict

        buckets: Dict[Tuple[str, str, float], List[str]] = defaultdict(list)
        for r in raw_groups:
            fid = r.get("fact_id") or r.get("id")
            if fid is None:
                continue
            key = (
                fid,
                r.get("template_name", r.get("template", "unknown")),
                float(r.get("temperature", 0.0)),
            )
            buckets[key].append(r.get("raw_response", "") or "")
        return [
            {
                "fact_id": k[0],
                "template": k[1],
                "temperature": k[2],
                "response_texts": v,
            }
            for k, v in sorted(buckets.items())
        ]

    raise SystemExit(
        f"{path} records have neither 'response_texts' nor 'raw_response'. "
        "Cannot locate raw LLM responses to re-score. See the docstring "
        "for the expected schema."
    )


def _load_minilm_per_fact(
    per_fact_csv: Path,
) -> pd.DataFrame:
    """Load the MiniLM semantic score per fact from per_fact_report.csv.

    Collapses (template, temperature) rows to one number per fact by mean,
    mirroring the fact-level aggregation used in evaluate_with_ci.py.
    """
    df = pd.read_csv(per_fact_csv, comment="#")
    if "fact_id" not in df.columns or "semantic_score" not in df.columns:
        raise SystemExit(
            f"{per_fact_csv} is missing 'fact_id' and/or 'semantic_score'. "
            "Point this script at reports/per_fact_report.csv."
        )
    per_fact = df.groupby("fact_id")["semantic_score"].mean().reset_index()
    per_fact = per_fact.rename(columns={"semantic_score": "miniLM_score"})
    return per_fact


def _score_fact_with_model(
    response_groups: List[List[str]],
    model_name: str,
) -> float:
    """Compute the mean semantic consistency across multiple groups for one fact.

    Each group is a list of LLM responses for one (template, temperature)
    condition. We score each group with compute_semantic_consistency and
    return the mean, so that the FinBERT per-fact number is comparable
    to the MiniLM per-fact aggregation used in per_fact_report.csv.
    """
    # Local import so that `python scripts/rescore_semantic_finbert.py --help`
    # does not eagerly load torch/transformers.
    from src.scorer import compute_semantic_consistency  # type: ignore

    group_scores: List[float] = []
    for texts in response_groups:
        if not texts or len(texts) < 2:
            # Fewer than 2 responses — compute_semantic_consistency returns
            # 1.0 in that degenerate case, which would bias the mean. Skip.
            continue
        group_scores.append(compute_semantic_consistency(texts, model_name))
    if not group_scores:
        return float("nan")
    return float(np.mean(group_scores))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient; returns NaN when either side is constant."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run(args: argparse.Namespace) -> int:
    per_fact_path = Path(args.per_fact)
    out_path = Path(args.out)
    results_path = Path(args.results_json) if args.results_json else None

    if not per_fact_path.exists():
        print(
            f"ERROR: per-fact CSV not found at {per_fact_path}",
            file=sys.stderr,
        )
        return 2

    is_synthetic = False
    if args.from_per_fact_csv:
        if not Path(args.ground_truth).exists():
            print(
                f"ERROR: --from-per-fact-csv mode requires a valid "
                f"--ground-truth path; got {args.ground_truth}",
                file=sys.stderr,
            )
            return 2
        print(
            "NOTE: Running in --from-per-fact-csv fallback mode. The output "
            "CSV contains scores on a synthetic paraphrase corpus, NOT on "
            "real LLM responses. See the script header for rationale.",
            file=sys.stderr,
        )
        is_synthetic = True
    elif results_path is None or not results_path.exists():
        print(
            f"ERROR: results.json not found at {results_path}",
            file=sys.stderr,
        )
        print(
            "This repository's historical runs (in results/*/) did not "
            "preserve raw response texts — only config.json was saved. "
            "FinBERT re-scoring requires response_texts or raw_response "
            "records. Run a new benchmark evaluation (the current engine "
            "writes results.json) and re-run this script, or pass "
            "--from-per-fact-csv to use the synthetic-corpus fallback.",
            file=sys.stderr,
        )
        return 2

    if args.finbert_model not in _SUPPORTED_FINBERT:
        print(
            f"WARN: {args.finbert_model} is not in the vetted FinBERT list "
            f"({sorted(_SUPPORTED_FINBERT)}). Proceeding anyway — the "
            "scorer will route any HuggingFace model through transformers "
            "mean-pooling as long as it has the AutoModel API.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Step 1: load MiniLM per-fact scores (already computed, in the CSV)
    # ------------------------------------------------------------------
    minilm = _load_minilm_per_fact(per_fact_path)

    # ------------------------------------------------------------------
    # Step 2: load the response groups — either real (results.json)
    # or synthetic (per_fact_report.csv + facts.json paraphrases).
    # ------------------------------------------------------------------
    if is_synthetic:
        groups = _synth_response_groups_from_per_fact_csv(
            per_fact_path, Path(args.ground_truth)
        )
    else:
        assert results_path is not None
        groups = _load_results_json(results_path)
    # Group by fact_id so we can aggregate per-fact
    from collections import defaultdict
    per_fact_groups: Dict[str, List[List[str]]] = defaultdict(list)
    for g in groups:
        per_fact_groups[g["fact_id"]].append(g["response_texts"])

    # ------------------------------------------------------------------
    # Step 3: score each fact with FinBERT
    # ------------------------------------------------------------------
    try:
        # Probe load — catches network / model-download failures BEFORE we
        # iterate through all 29 facts and hit the same error 29 times.
        from src.scorer import _get_embedding_model  # type: ignore

        _get_embedding_model(args.finbert_model)
    except Exception as exc:
        print(
            f"ERROR: Could not load FinBERT model {args.finbert_model!r}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Typical causes: no network access to HuggingFace Hub, "
            "insufficient disk space for the ~400 MB model, or "
            "transformers/torch version mismatch. The code path is "
            "committed; rerun on a machine with network + disk.",
            file=sys.stderr,
        )
        return 3

    rows: List[Dict[str, Any]] = []
    print(
        f"Scoring {len(per_fact_groups)} facts with FinBERT "
        f"({args.finbert_model}) …",
        file=sys.stderr,
    )
    for fact_id, groups_for_fact in sorted(per_fact_groups.items()):
        finbert_score = _score_fact_with_model(groups_for_fact, args.finbert_model)
        minilm_row = minilm[minilm["fact_id"] == fact_id]
        minilm_score = (
            float(minilm_row["miniLM_score"].iloc[0])
            if len(minilm_row)
            else float("nan")
        )
        rows.append(
            {
                "fact_id": fact_id,
                "miniLM_score": round(minilm_score, 6),
                "finbert_score": round(finbert_score, 6),
                "delta": round(finbert_score - minilm_score, 6),
            }
        )

    # ------------------------------------------------------------------
    # Step 4: write CSV
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_label = (
        "synthetic paraphrase corpus from per_fact_report.csv "
        "(code-path validation — NOT real LLM responses)"
        if is_synthetic
        else str(results_path)
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# Semantic consistency — MiniLM vs FinBERT comparison\n"
            f"# FinBERT model: {args.finbert_model}\n"
            f"# Source:        {source_label}\n"
            "#\n"
            "# Columns:\n"
            "#   fact_id       - Ground-truth fact identifier\n"
            "#   miniLM_score  - Per-fact mean of semantic_score from per_fact_report.csv\n"
            "#   finbert_score - Same metric, recomputed with FinBERT mean-pooled embeddings\n"
            "#   delta         - finbert_score - miniLM_score\n"
            "#\n"
        )
        writer = csv.DictWriter(
            fh, fieldnames=["fact_id", "miniLM_score", "finbert_score", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Step 5: console summary
    # ------------------------------------------------------------------
    df_out = pd.DataFrame(rows)
    minilm_vals = df_out["miniLM_score"].to_numpy(dtype=float)
    finbert_vals = df_out["finbert_score"].to_numpy(dtype=float)

    # Drop any NaNs from the summary stats
    mask = ~np.isnan(minilm_vals) & ~np.isnan(finbert_vals)
    minilm_clean = minilm_vals[mask]
    finbert_clean = finbert_vals[mask]

    mean_minilm = float(minilm_clean.mean()) if len(minilm_clean) else float("nan")
    mean_finbert = (
        float(finbert_clean.mean()) if len(finbert_clean) else float("nan")
    )
    corr = _pearson(minilm_clean, finbert_clean)
    disagree_over_0_1 = int(
        ((finbert_clean - minilm_clean).__abs__() > 0.1).sum()
    )

    print("")
    print("=" * 72)
    print("  MiniLM vs FinBERT semantic consistency")
    print("=" * 72)
    if is_synthetic:
        print(
            "MODE                : --from-per-fact-csv fallback "
            "(synthetic corpus — code-path validation only)"
        )
    else:
        print("MODE                : real LLM responses from results.json")
    print(f"FinBERT model       : {args.finbert_model}")
    print(f"Facts scored        : {len(df_out)}  ({len(minilm_clean)} with both scores)")
    print(f"Mean MiniLM score   : {mean_minilm:.4f}")
    print(f"Mean FinBERT score  : {mean_finbert:.4f}")
    print(f"Mean delta          : {mean_finbert - mean_minilm:+.4f}")
    print(
        f"Pearson correlation : {corr:.4f}"
        if not np.isnan(corr)
        else "Pearson correlation : nan (insufficient variance)"
    )
    print(f"Facts with |delta| > 0.1 : {disagree_over_0_1}")
    print(f"Output CSV          : {out_path}")
    print("=" * 72)
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score semantic consistency with FinBERT on the same response "
            "texts used by the original MiniLM run, and compare."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to results/<run_id>/results.json produced by src/engine.py.",
    )
    parser.add_argument(
        "--from-per-fact-csv",
        action="store_true",
        help=(
            "Fallback mode: build a synthetic per-fact paraphrase corpus "
            "from per_fact_report.csv + ground_truth/facts.json. Use only "
            "when results.json from a real run is not available — the "
            "output is labelled synthetic and is NOT a substitute for "
            "re-scoring actual LLM responses."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="ground_truth/facts.json",
        help="Path to facts.json (only used by --from-per-fact-csv).",
    )
    parser.add_argument(
        "--per-fact",
        type=str,
        default="reports/per_fact_report.csv",
        help="Existing MiniLM aggregated report (default: reports/per_fact_report.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/semantic_finbert.csv",
        help="Output CSV with per-fact MiniLM-vs-FinBERT comparison.",
    )
    parser.add_argument(
        "--finbert-model",
        type=str,
        default=_DEFAULT_FINBERT,
        help=(
            "FinBERT model identifier. Supported: "
            f"{sorted(_SUPPORTED_FINBERT)}. Default: {_DEFAULT_FINBERT}."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(_parse_args(sys.argv[1:])))
