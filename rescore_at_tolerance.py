"""
Sweep the hallucination tolerance and report how the rates shift.

See docs/SENSITIVITY_ANALYSIS.md for the motivation and the difference
between the two modes this script supports.

Modes
-----
1. --csv mode (modal-only approximation)
     Reads reports/per_fact_report.csv, which contains modal_value and
     ground_truth per group. For each tolerance level, flags each group
     as hallucinated-at-tolerance if |modal - gt| / |gt| > tolerance.
     Output column is `modal_hallucination_fraction` — NOT the full
     hallucination_rate, because the per-run extracted values are not
     available in the aggregated CSV.

2. --results-json mode (full recomputation, preferred)
     Reads results/<run_id>/results.json and re-runs the hallucination
     computation on the full extracted-values list per group. Output
     column is `hallucination_rate` — directly comparable to the number
     originally published. Produces the same semantics as re-running
     `report.py` with a different tolerance in config.yaml, but does
     so without making any LLM calls.

Usage
-----
    # Fast, offline, modal-only approximation:
    python rescore_at_tolerance.py \\
        --csv reports/per_fact_report.csv \\
        --out reports/tolerance_sensitivity.csv

    # Full recomputation (requires the raw results JSON from the run):
    python rescore_at_tolerance.py \\
        --results-json results/20260401_040014_891022/results.json \\
        --ground-truth ground_truth/facts.json \\
        --out reports/tolerance_sensitivity.csv

    # Custom tolerance ladder:
    python rescore_at_tolerance.py \\
        --csv reports/per_fact_report.csv \\
        --tolerances 0.001 0.005 0.01 0.02 0.05 0.1 0.2 \\
        --out reports/tolerance_sensitivity.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_TOLERANCES: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1)


def _is_hallucination(
    value: Optional[float], ground_truth: float, tolerance: float
) -> bool:
    """Mirrors src.scorer.compute_hallucination_rate's per-value predicate.

    None -> True (extraction failure counts as hallucination).
    ground_truth == 0 -> fall back to absolute tolerance 0.001.
    else -> |value - gt| / |gt| > tolerance.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    if ground_truth == 0.0:
        return abs(value) > 0.001
    return abs(value - ground_truth) / abs(ground_truth) > tolerance


def _flag_from_composite(
    composite: float,
    green_threshold: float,
    yellow_threshold: float,
) -> str:
    if composite >= green_threshold:
        return "green"
    if composite >= yellow_threshold:
        return "yellow"
    return "red"


# ---------------------------------------------------------------------------
# CSV (modal-only) mode
# ---------------------------------------------------------------------------


def rescore_from_csv(
    csv_path: Path,
    tolerances: Iterable[float],
    composite_weights: Dict[str, float],
    green_threshold: float,
    yellow_threshold: float,
) -> pd.DataFrame:
    """Modal-only hallucination sensitivity from per_fact_report.csv.

    For each tolerance, for each group row, decide whether the MODAL
    extracted value alone would be considered a hallucination at that
    tolerance. Aggregate across groups.

    This is a weaker signal than the full hallucination_rate because it
    ignores the per-run extracted values, but it is computable offline
    from the aggregated CSV.
    """
    df = pd.read_csv(csv_path, comment="#")

    required_cols = {
        "fact_id",
        "modal_value",
        "ground_truth",
        "semantic_score",
        "factual_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(
            f"{csv_path} is missing expected columns: {sorted(missing)}. "
            "Ensure you are pointing at reports/per_fact_report.csv."
        )

    rows = []
    for tol in tolerances:
        halluc = df.apply(
            lambda r: _is_hallucination(r["modal_value"], r["ground_truth"], tol),
            axis=1,
        )
        halluc_rate = float(halluc.mean())

        # Recompose the composite with the modal-only hallucination signal
        # as a proxy. Note: this is an APPROXIMATION. The original composite
        # uses the full per-run hallucination_rate; here we replace that
        # term with the modal-hallucination indicator. Direction of effect
        # is right, magnitude is approximate.
        composite_approx = (
            df["semantic_score"] * composite_weights["semantic_consistency"]
            + df["factual_score"] * composite_weights["factual_consistency"]
            + (1.0 - halluc.astype(float)) * composite_weights["hallucination_rate"]
        )
        mean_composite = float(composite_approx.mean())

        flags = composite_approx.apply(
            lambda c: _flag_from_composite(c, green_threshold, yellow_threshold)
        )
        n_green = int((flags == "green").sum())
        n_yellow = int((flags == "yellow").sum())
        n_red = int((flags == "red").sum())

        rows.append(
            {
                "tolerance": tol,
                "modal_hallucination_fraction": round(halluc_rate, 6),
                "composite_stability_approx": round(mean_composite, 6),
                "n_green": n_green,
                "n_yellow": n_yellow,
                "n_red": n_red,
                "n_groups": len(df),
                "mode": "csv_modal_only",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# results.json (full recomputation) mode
# ---------------------------------------------------------------------------


def rescore_from_results_json(
    results_path: Path,
    ground_truth_path: Path,
    tolerances: Iterable[float],
    composite_weights: Dict[str, float],
    green_threshold: float,
    yellow_threshold: float,
) -> pd.DataFrame:
    """Full hallucination_rate sweep from a raw results.json.

    This is the preferred mode: it uses the per-run extracted values, so
    the resulting hallucination_rate is directly comparable to the number
    originally published by report.py. The composite is also recomputed
    from the actual weights rather than via the modal-only approximation.

    Schema assumption: results.json is a JSON object with a top-level
    "results" (list) or equivalent structure produced by src/engine.py.
    Each entry is expected to have either:
      - an 'extracted_values' list of floats / nulls, or
      - a 'response_texts' list, in which case extraction is re-run.
    The function is forgiving about the outer container: if it finds a
    list of group dicts at the top level, it uses that; otherwise it
    expects 'results' or 'groups' as the key.
    """
    import sys

    # src/ may not be on PYTHONPATH when this script is invoked directly
    # from the repo root, so add it explicitly.
    repo_root = results_path.resolve().parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from src.extractor import extract_numeric  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"Could not import src.extractor. Run from the repo root "
            f"or ensure PYTHONPATH is configured. Original error: {exc}"
        )

    with results_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        groups = data
    elif isinstance(data, dict):
        groups = data.get("results") or data.get("groups") or []
    else:
        raise SystemExit(
            f"{results_path} has unexpected top-level type {type(data).__name__}"
        )
    if not groups:
        raise SystemExit(f"{results_path} contains no evaluation groups.")

    with ground_truth_path.open("r", encoding="utf-8") as fh:
        gt_raw = json.load(fh)
    facts_list = gt_raw.get("facts", gt_raw) if isinstance(gt_raw, dict) else gt_raw
    gt_by_id: Dict[str, Dict[str, Any]] = {f["id"]: f for f in facts_list}

    # Per-group extracted values (pre-compute once, reuse per tolerance).
    per_group = []
    for g in groups:
        fact_id = g.get("fact_id") or g.get("id")
        if fact_id not in gt_by_id:
            # Skip groups whose ground truth is no longer in the facts file.
            continue
        gt = gt_by_id[fact_id]
        ground_truth_value = float(gt["value"])
        expected_unit = gt.get("unit")

        if "extracted_values" in g and g["extracted_values"] is not None:
            extracted = [None if v is None else float(v) for v in g["extracted_values"]]
        elif "response_texts" in g:
            extracted = []
            for text in g["response_texts"]:
                res = extract_numeric(text, expected_unit=expected_unit)
                extracted.append(res.value)
        else:
            # Nothing we can do — skip this group.
            continue

        semantic = g.get("semantic_score", g.get("semantic_consistency"))
        factual = g.get("factual_score", g.get("factual_consistency"))
        per_group.append(
            {
                "fact_id": fact_id,
                "extracted": extracted,
                "ground_truth": ground_truth_value,
                "semantic": semantic,
                "factual": factual,
            }
        )

    if not per_group:
        raise SystemExit(
            "No usable groups in results.json (no extracted_values and no "
            "response_texts found, or fact_ids no longer in facts.json)."
        )

    rows = []
    for tol in tolerances:
        all_rates = []
        all_composites = []
        flags_counter = {"green": 0, "yellow": 0, "red": 0}
        for g in per_group:
            n = len(g["extracted"])
            if n == 0:
                continue
            n_halluc = sum(
                _is_hallucination(v, g["ground_truth"], tol) for v in g["extracted"]
            )
            rate = n_halluc / n
            all_rates.append(rate)

            semantic = g["semantic"] if g["semantic"] is not None else 0.0
            factual = g["factual"] if g["factual"] is not None else 0.0
            composite = (
                semantic * composite_weights["semantic_consistency"]
                + factual * composite_weights["factual_consistency"]
                + (1.0 - rate) * composite_weights["hallucination_rate"]
            )
            all_composites.append(composite)
            flags_counter[
                _flag_from_composite(composite, green_threshold, yellow_threshold)
            ] += 1

        rows.append(
            {
                "tolerance": tol,
                "hallucination_rate": round(float(np.mean(all_rates)), 6),
                "composite_stability": round(float(np.mean(all_composites)), 6),
                "n_green": flags_counter["green"],
                "n_yellow": flags_counter["yellow"],
                "n_red": flags_counter["red"],
                "n_groups": len(all_rates),
                "mode": "results_json_full",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(
            "# Tolerance sensitivity sweep — see docs/SENSITIVITY_ANALYSIS.md\n"
            "# Column semantics:\n"
            "#   tolerance                       Relative tolerance threshold\n"
            "#   hallucination_rate OR           Full mean rate over all (fact,template,temperature,run) units\n"
            "#   modal_hallucination_fraction    Fraction of groups whose MODAL value alone fails the tolerance test\n"
            "#   composite_stability[_approx]    Recomputed composite (approx in csv mode)\n"
            "#   n_green/n_yellow/n_red          Flag distribution recomputed at this tolerance\n"
            "#   n_groups                        Number of evaluation groups used\n"
            "#   mode                            csv_modal_only | results_json_full\n"
            "#\n"
        )
        df.to_csv(fh, index=False)


def run(args: argparse.Namespace) -> int:
    tolerances = list(args.tolerances) if args.tolerances else list(DEFAULT_TOLERANCES)
    composite_weights = {
        "semantic_consistency": args.weight_semantic,
        "factual_consistency": args.weight_factual,
        "hallucination_rate": args.weight_hallucination,
    }

    if args.results_json:
        if not args.ground_truth:
            raise SystemExit(
                "--results-json requires --ground-truth (path to facts.json) "
                "so ground-truth values can be re-matched."
            )
        df = rescore_from_results_json(
            Path(args.results_json),
            Path(args.ground_truth),
            tolerances,
            composite_weights,
            args.green_threshold,
            args.yellow_threshold,
        )
    elif args.csv:
        df = rescore_from_csv(
            Path(args.csv),
            tolerances,
            composite_weights,
            args.green_threshold,
            args.yellow_threshold,
        )
    else:
        raise SystemExit(
            "Supply either --csv reports/per_fact_report.csv OR "
            "--results-json results/<run_id>/results.json --ground-truth ground_truth/facts.json"
        )

    # Console summary.
    print("=" * 72)
    print("  Tolerance sensitivity sweep")
    print("=" * 72)
    print(df.to_string(index=False))
    print("")

    if args.out:
        _write_csv(df, Path(args.out))
        print(f"Wrote: {args.out}")
    print("=" * 72)
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep hallucination tolerance across configured levels.",
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to reports/per_fact_report.csv (modal-only approximation).",
    )
    src.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to results/<run_id>/results.json (full recomputation).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="ground_truth/facts.json",
        help="Path to facts.json (only needed in --results-json mode).",
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="*",
        default=None,
        help=f"Tolerance ladder (default: {DEFAULT_TOLERANCES}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path (e.g. reports/tolerance_sensitivity.csv).",
    )
    parser.add_argument(
        "--weight-semantic",
        type=float,
        default=0.30,
        help="Composite weight for semantic_consistency (default: 0.30).",
    )
    parser.add_argument(
        "--weight-factual",
        type=float,
        default=0.40,
        help="Composite weight for factual_consistency (default: 0.40).",
    )
    parser.add_argument(
        "--weight-hallucination",
        type=float,
        default=0.30,
        help="Composite weight for (1 - hallucination_rate) (default: 0.30).",
    )
    parser.add_argument(
        "--green-threshold",
        type=float,
        default=0.75,
        help="Composite threshold for the green flag (default: 0.75).",
    )
    parser.add_argument(
        "--yellow-threshold",
        type=float,
        default=0.50,
        help="Composite threshold for the yellow flag (default: 0.50).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(_parse_args(sys.argv[1:])))
