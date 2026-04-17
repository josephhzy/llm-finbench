"""Validate ground_truth/facts.json against the schema contract.

Checks every fact record for:

  * presence and non-null-ness of REQUIRED fields (schema contract from
    ground_truth/PROVENANCE.md);
  * presence of RECOMMENDED fields (nice-to-have, not blocking);
  * presence of PROVENANCE fields that distinguish a "preliminary" fact
    from a "verified" one — specifically `page` (integer) and whether
    the `context` quote contains the declared `value` (a weak heuristic
    for verbatim extraction).

This script has no hard dependencies beyond the standard library + json.
Run it from the repo root:

    python scripts/validate_ground_truth.py \\
        --facts ground_truth/facts.json

Exit code 0 means: every fact has every REQUIRED field. A non-zero exit
status means there is at least one missing REQUIRED field (a hard gap).
Missing RECOMMENDED or PROVENANCE fields are warnings, not errors.

The primary output is a short text summary printed to stdout. When
invoked with `--markdown`, it also emits a markdown block that the
`PROVENANCE.md` "Current status" section is expected to include.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# The schema contract — must match ground_truth/PROVENANCE.md
# ---------------------------------------------------------------------------

# Fields every fact MUST have (not None, not empty string). Missing any
# of these blocks production use of the fact.
REQUIRED_FIELDS: Tuple[str, ...] = (
    "id",
    "company",
    "metric",
    "period",
    "value",
    "unit",
    "source",
    "difficulty",
    "category",
)

# Fields we expect to see populated but do not hard-block when missing.
# These are the "nice-to-have" fields whose presence improves downstream
# analysis but whose absence does not invalidate a fact.
RECOMMENDED_FIELDS: Tuple[str, ...] = (
    "metric_abbreviation",
    "context",
)

# Fields that distinguish a "verified" fact from a "preliminary" one.
# Their absence does NOT block use of the fact, but their presence is the
# signal that a human reviewer has audited the source document.
PROVENANCE_FIELDS: Tuple[str, ...] = (
    "page",  # integer page number in the source PDF
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """Treat None and empty string as 'missing'.

    The ground-truth file uses `null` for unknown page numbers and
    `""` is an equivalent "not yet populated" marker that slipped in
    occasionally. Treat both the same so the validator output is
    robust to either.
    """
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _value_appears_in_context(value: Any, context: str) -> bool:
    """Weak heuristic: does the `context` quote contain the declared value?

    Exact substring match against the numeric value in a couple of common
    surface forms. This does NOT prove the context is a verbatim quote
    from the PDF (that requires a human reviewer), but it does catch
    the obvious gap where the context text mentions no figure at all.
    """
    if _is_missing(context) or _is_missing(value):
        return False
    text = context.lower()
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False

    # Surface forms to probe — match how the ground truth tends to render.
    candidates: List[str] = []
    # Integer-looking values
    if v == int(v):
        candidates.append(f"{int(v)}")
        candidates.append(f"{int(v):,}")
    # Decimal
    candidates.append(f"{v:.2f}")
    candidates.append(f"{v:.1f}")
    # Plain
    candidates.append(f"{v}")

    return any(c.lower() in text for c in candidates)


def validate_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a per-fact validation report.

    Returns a dict with keys:
      fact_id          — fact identifier, or "<unknown>" if even that is missing
      missing_required — list of required fields that are null/empty
      missing_recommended — list of recommended fields that are null/empty
      missing_provenance  — list of provenance fields that are null/empty
      context_contains_value — bool, heuristic
    """
    fact_id = fact.get("id", "<unknown>")

    missing_required = [f for f in REQUIRED_FIELDS if _is_missing(fact.get(f))]
    missing_recommended = [f for f in RECOMMENDED_FIELDS if _is_missing(fact.get(f))]
    missing_provenance = [f for f in PROVENANCE_FIELDS if _is_missing(fact.get(f))]

    context = fact.get("context", "")
    value = fact.get("value")
    context_contains_value = _value_appears_in_context(value, context)

    return {
        "fact_id": fact_id,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "missing_provenance": missing_provenance,
        "context_contains_value": context_contains_value,
    }


def validate_all(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load facts.json, validate every fact, and return per-fact + aggregate reports."""
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    facts_list = raw.get("facts", raw) if isinstance(raw, dict) else raw
    if not isinstance(facts_list, list):
        raise SystemExit(
            f"{path} did not parse to a list of facts. "
            f"Top-level type: {type(facts_list).__name__}."
        )

    per_fact = [validate_fact(f) for f in facts_list]

    total = len(per_fact)
    n_required_ok = sum(1 for r in per_fact if not r["missing_required"])
    n_recommended_ok = sum(1 for r in per_fact if not r["missing_recommended"])
    n_with_page = sum(1 for r in per_fact if "page" not in r["missing_provenance"])
    n_context_has_value = sum(1 for r in per_fact if r["context_contains_value"])

    aggregate = {
        "total": total,
        "n_required_ok": n_required_ok,
        "n_required_missing": total - n_required_ok,
        "n_recommended_ok": n_recommended_ok,
        "n_with_page_citation": n_with_page,
        "n_context_contains_value": n_context_has_value,
    }

    return per_fact, aggregate


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_text(per_fact: List[Dict[str, Any]], aggregate: Dict[str, Any]) -> None:
    total = aggregate["total"]
    print("=" * 72)
    print("  ground_truth/facts.json validator")
    print("=" * 72)
    print(f"Total facts                    : {total}")
    print(f"With ALL required fields       : {aggregate['n_required_ok']}/{total}")
    print(f"With ALL recommended fields    : {aggregate['n_recommended_ok']}/{total}")
    print(
        f"With a non-null `page`         : {aggregate['n_with_page_citation']}/{total}"
    )
    print(
        f"Context appears to quote value : {aggregate['n_context_contains_value']}/{total}"
    )
    print("")

    # List any fact that is missing REQUIRED fields (these are real blockers)
    bad = [r for r in per_fact if r["missing_required"]]
    if bad:
        print("Facts missing one or more REQUIRED fields:")
        for r in bad:
            print(f"  {r['fact_id']}: missing {r['missing_required']}")
    else:
        print("All facts pass the REQUIRED-field check.")
    print("")

    # List any fact that is missing RECOMMENDED fields (non-blocking)
    soft = [r for r in per_fact if r["missing_recommended"]]
    if soft:
        print("Facts missing one or more RECOMMENDED fields (non-blocking):")
        for r in soft:
            print(f"  {r['fact_id']}: missing {r['missing_recommended']}")
    print("=" * 72)


def _print_markdown(
    per_fact: List[Dict[str, Any]], aggregate: Dict[str, Any], run_date: str
) -> None:
    total = aggregate["total"]
    print(
        f"Validator last run {run_date}: "
        f"{aggregate['n_required_ok']}/{total} facts have all required fields; "
        f"{aggregate['n_with_page_citation']}/{total} have PDF page citations; "
        f"{aggregate['n_context_contains_value']}/{total} have a `context` "
        f"string that contains the declared numeric value (weak heuristic for "
        f"verbatim extraction — NOT a substitute for human review)."
    )


def run(args: argparse.Namespace) -> int:
    facts_path = Path(args.facts)
    if not facts_path.exists():
        print(f"ERROR: facts file not found at {facts_path}", file=sys.stderr)
        return 2

    per_fact, aggregate = validate_all(facts_path)

    if args.markdown:
        _print_markdown(per_fact, aggregate, args.run_date or "<today>")
    else:
        _print_text(per_fact, aggregate)

    # Non-zero exit iff any fact is missing a REQUIRED field.
    return 0 if aggregate["n_required_missing"] == 0 else 1


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate ground_truth/facts.json against the schema contract.",
    )
    parser.add_argument(
        "--facts",
        type=str,
        default="ground_truth/facts.json",
        help="Path to the facts JSON file (default: ground_truth/facts.json).",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help=(
            "Emit the one-line markdown snippet expected by "
            "PROVENANCE.md's 'Current status' section, rather than the "
            "full text summary."
        ),
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Override the run date in markdown output (default: <today>).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(_parse_args(sys.argv[1:])))
