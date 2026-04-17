"""Numeric value extractor for LLM Financial Stability Bench.

Extracts numeric values from free-form LLM response text using only regex
and string manipulation — never an LLM. This is critical to evaluation
integrity: we cannot use the model under test to parse its own outputs.

Design decisions:
- Returns ExtractedValue with value=None when extraction fails, rather than
  guessing. Failed extractions are data — they reveal prompt design issues
  and should be tracked separately.
- Uses modal value (most common) downstream, not mean, because mean hides
  bimodal distributions. This extractor's job is to produce clean floats
  for the scorer to aggregate.
- Handles Singapore financial reporting conventions: S$/SGD currency,
  MAS-standard metrics (NIM, NPL, CET1), and the specific numeric formats
  found in DBS/OCBC/UOB/Singtel/CapitaLand annual reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class ExtractedValue:
    """A single extracted numeric value with contextual metadata."""

    value: Optional[float]  # The extracted number, or None if failed
    raw_match: Optional[str]  # The matched text substring
    unit: Optional[str]  # "percent", "millions", "billions", "ratio", "sgd", etc.
    currency: Optional[str]  # "SGD", "USD", "HKD", etc.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Master regex: matches numeric literals including comma-separated thousands,
# European-style space-separated thousands, and decimal points.
# Captures an optional leading minus or parenthetical negative.
# Group 1: optional leading minus
# Group 2: optional opening parenthesis (signals negative)
# Group 3: the numeric body  (e.g. "1,234.56" or "5.2" or "1 234.56")
# Group 4: optional closing parenthesis
_NUM_PATTERN = re.compile(
    r"""
    (?:^|(?<=\s)|(?<=\$))        # boundary: start, whitespace, or $ symbol
    (-)?                          # (1) optional leading minus
    (\()?                         # (2) optional opening paren (negative indicator)
    (                             # (3) numeric body
        (?:\d{1,3}(?:,\d{3})+)   #     comma-grouped: 1,234 or 1,234,567
        (?:\.\d+)?                #     optional decimal after comma-grouped
      |                           #   OR
        (?:\d{1,3}(?:\ \d{3})+)  #     space-grouped (European): 1 234 or 1 234 567
        (?:\.\d+)?                #     optional decimal after space-grouped
      |                           #   OR
        \d+\.\d+                  #     plain decimal: 123.45
      |                           #   OR
        \d+                       #     plain integer: 123
    )
    (\))?                         # (4) optional closing paren
    """,
    re.VERBOSE,
)

# Scale words that follow a number
_SCALE_MAP = {
    "trillion": 1_000_000_000_000,
    "trn": 1_000_000_000_000,
    "t": 1_000_000_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "bil": 1_000_000_000,
    "b": 1_000_000_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "mil": 1_000_000,
    "m": 1_000_000,
    "thousand": 1_000,
    "k": 1_000,
}

# Regex to detect scale words immediately after the number
_SCALE_PATTERN = re.compile(
    r"\s*(trillion|trn|billion|bil|bn|million|mil|mn|thousand|[tbmk])\b",
    re.IGNORECASE,
)

# Percentage indicators
# Note: '%' is non-word so \b doesn't work after it — handle separately.
_PCT_PATTERN = re.compile(
    r"\s*(?:(%)|(?:(per\s*cent(?:age)?(?:\s*point)?|percent(?:age)?(?:\s*point)?|bps|basis\s*points?)\b))",
    re.IGNORECASE,
)

# Basis points indicator (subset of _PCT_PATTERN).
# Detected separately so the value can be divided by 100:
# 25 bps = 0.25%, not 25%.
_BPS_PATTERN = re.compile(
    r"\s*(bps|basis\s*points?)\b",
    re.IGNORECASE,
)

# Ratio indicators (e.g. "0.85x", "0.85 times")
_RATIO_PATTERN = re.compile(
    r"\s*(x|times)\b",
    re.IGNORECASE,
)

# Cents indicator (e.g. "15.0 cents")
_CENTS_PATTERN = re.compile(
    r"\s*cents?\b",
    re.IGNORECASE,
)

# Currency prefixes — order matters (longer/more-specific first to avoid
# partial matches). E.g. "US$" must be checked before "S$", and "S$" must
# not match when preceded by "U".
_CURRENCY_PREFIXES = [
    (re.compile(r"US\$\s*$"), "USD"),
    (re.compile(r"USD\s*$", re.IGNORECASE), "USD"),
    (re.compile(r"HK\$\s*$"), "HKD"),
    (re.compile(r"HKD\s*$", re.IGNORECASE), "HKD"),
    (re.compile(r"(?<![A-Za-z])S\$\s*$"), "SGD"),  # S$ but not US$
    (re.compile(r"SGD\s*$", re.IGNORECASE), "SGD"),
    (re.compile(r"A\$\s*$"), "AUD"),
    (re.compile(r"AUD\s*$", re.IGNORECASE), "AUD"),
    (re.compile(r"RM\s*$"), "MYR"),
    (re.compile(r"MYR\s*$", re.IGNORECASE), "MYR"),
    (re.compile(r"EUR\s*$", re.IGNORECASE), "EUR"),
    (re.compile(r"€\s*$"), "EUR"),
    (re.compile(r"£\s*$"), "GBP"),
    (re.compile(r"GBP\s*$", re.IGNORECASE), "GBP"),
    (re.compile(r"\$\s*$"), "USD"),  # bare $ defaults to USD
]

# Negative-indicator words that precede the number (within ~30 chars).
# Only includes words that describe the VALUE as negative, not movement words
# ("declined by", "fell by") which describe direction of change but leave the
# metric value itself positive. Using movement words here would cause
# "NIM declined by 2.14%" to extract -2.14 instead of 2.14, creating false
# hallucination flags against a positive ground truth.
_NEG_WORDS = re.compile(
    r"\b(loss|losses|negative|deficit|shortfall|decline|decrease)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_number(s: str) -> Optional[float]:
    """Parse a numeric string into a float.

    Handles:
    - Comma thousands: "1,234.56" -> 1234.56
    - Space thousands (European-style): "1 234.56" -> 1234.56
    - Plain: "1234.56" -> 1234.56
    - Integer: "1234" -> 1234.0

    Returns None if parsing fails.
    """
    if s is None:
        return None
    # Strip commas
    cleaned = s.replace(",", "")
    # Strip internal spaces (European thousands separator)
    # But only spaces that are between digits (already matched by regex)
    cleaned = re.sub(r"(\d) (\d)", r"\1\2", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _detect_sign(text: str, match_start: int) -> int:
    """Detect whether the number should be negative from surrounding context.

    Checks a window of text preceding the match for negative-indicator words
    like "loss of", "decline of", "decrease of", etc.

    Returns -1 if negative indicators found, +1 otherwise.
    """
    # Look back up to 40 characters for negative context words
    lookback = max(0, match_start - 40)
    preceding = text[lookback:match_start]
    if _NEG_WORDS.search(preceding):
        return -1
    return 1


def _detect_scale(text: str, match_end: int) -> float:
    """Detect scale multiplier (billion, million, etc.) after the number.

    Returns the multiplier (e.g. 1_000_000_000 for "billion") or 1.0 if
    no scale word found.
    """
    after = text[match_end : match_end + 20]
    m = _SCALE_PATTERN.match(after)
    if m:
        word = m.group(1).lower()
        return _SCALE_MAP.get(word, 1.0)
    return 1.0


def _detect_percentage(text: str, match_end: int) -> bool:
    """Check if the number is followed by a percentage indicator."""
    after = text[match_end : match_end + 25]
    return bool(_PCT_PATTERN.match(after))


def _detect_basis_points(text: str, match_end: int) -> bool:
    """Check if the number is followed by a basis points indicator (bps / basis points).

    Used in addition to _detect_percentage so the value can be divided by 100:
    25 bps -> 0.25 (percent), not 25.  One basis point = 0.01%.
    """
    after = text[match_end : match_end + 25]
    return bool(_BPS_PATTERN.match(after))


def _detect_ratio(text: str, match_end: int) -> bool:
    """Check if the number is followed by a ratio indicator (x, times)."""
    after = text[match_end : match_end + 10]
    return bool(_RATIO_PATTERN.match(after))


def _detect_cents(text: str, match_end: int) -> bool:
    """Check if the number is followed by 'cents'."""
    after = text[match_end : match_end + 10]
    return bool(_CENTS_PATTERN.match(after))


def _detect_currency(text: str, match_start: int) -> Optional[str]:
    """Detect currency from prefix characters before the number.

    Scans the text immediately preceding the number for currency symbols
    like S$, SGD, USD, $, etc.

    Returns the ISO currency code or None.
    """
    # Look back up to 10 characters
    lookback = max(0, match_start - 10)
    preceding = text[lookback:match_start]

    for pattern, currency in _CURRENCY_PREFIXES:
        if pattern.search(preceding):
            return currency
    return None


def _determine_unit(
    is_pct: bool,
    is_ratio: bool,
    is_cents: bool,
    scale: float,
    currency: Optional[str],
) -> Optional[str]:
    """Determine the unit label from detected indicators."""
    if is_pct:
        return "percent"
    if is_ratio:
        return "ratio"
    if is_cents:
        return "cents"
    if scale >= 1_000_000_000_000:
        return "trillions"
    if scale >= 1_000_000_000:
        return "billions"
    if scale >= 1_000_000:
        return "millions"
    if scale >= 1_000:
        return "thousands"
    if currency:
        return currency.lower()
    return None


def _compute_raw_match(text: str, match_start: int, match_end: int) -> str:
    """Extract a meaningful raw_match substring including surrounding context.

    Extends the match boundaries to include currency prefix and unit suffix
    so the raw_match is human-readable (e.g. "S$5.2 billion" not just "5.2").
    """
    # Extend left to capture currency prefix
    left = match_start
    lookback = max(0, match_start - 10)
    prefix_text = text[lookback:match_start]
    # Check for currency symbols
    for pattern, _ in _CURRENCY_PREFIXES:
        m = pattern.search(prefix_text)
        if m:
            left = lookback + m.start()
            break

    # Check for negative sign further back
    if left > 0 and text[left - 1] == "-":
        left -= 1

    # Extend right to capture unit suffix
    right = match_end
    after = text[match_end : match_end + 25]
    for suffix_pat in [_SCALE_PATTERN, _PCT_PATTERN, _RATIO_PATTERN, _CENTS_PATTERN]:
        m = suffix_pat.match(after)
        if m:
            right = match_end + m.end()
            break

    return text[left:right].strip()


# ---------------------------------------------------------------------------
# Candidate scoring (for ranking when multiple numeric values found)
# ---------------------------------------------------------------------------


def _score_candidate(
    ev: ExtractedValue,
    expected_unit: Optional[str],
    position: int,
    total_candidates: int,
) -> float:
    """Score a candidate for relevance.  Higher = more likely the answer.

    Scoring rationale:
    - Matching the expected_unit is the strongest signal (the caller knows
      what kind of value they're looking for).
    - Having a currency prefix is a moderate signal (financial context).
    - Earlier position is a weak tiebreaker (LLMs tend to state the answer
      early).
    """
    score = 0.0

    # Strong signal: unit match
    if expected_unit and ev.unit:
        eu = expected_unit.lower().strip()
        u = ev.unit.lower().strip()
        if eu == u:
            score += 100
        # Fuzzy matches for percent aliases
        elif eu in ("pct", "%", "percentage") and u == "percent":
            score += 100
        elif eu == "percent" and u in ("pct", "%", "percentage"):
            score += 100

    # Medium signal: has currency
    if ev.currency:
        score += 10

    # Medium signal: has unit
    if ev.unit:
        score += 5

    # Weak signal: position (earlier is better)
    if total_candidates > 1:
        score += (total_candidates - position) / total_candidates

    return score


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def _extract_one(text: str, match: re.Match) -> ExtractedValue:
    """Build an ExtractedValue from a single regex match within text."""
    leading_minus = match.group(1)
    open_paren = match.group(2)
    num_body = match.group(3)
    close_paren = match.group(4)

    match_start = match.start()
    match_end = match.end()

    # Parse the raw number
    parsed = _parse_number(num_body)
    if parsed is None:
        return ExtractedValue(value=None, raw_match=num_body, unit=None, currency=None)

    # Determine sign
    # Explicit minus or parenthetical negative take precedence
    is_negative = False
    if leading_minus == "-":
        is_negative = True
    elif open_paren == "(" and close_paren == ")":
        # Both parens captured directly (e.g. "(2.3)")
        is_negative = True
    elif open_paren == "(":
        # Opening paren captured but closing wasn't (e.g. "(2.3%)" where
        # % sits between the number and ")"). Scan forward for a closing
        # paren within a reasonable window.
        after_full = text[match_end : match_end + 30]
        if ")" in after_full:
            is_negative = True
    else:
        # Check contextual negative words
        if _detect_sign(text, match_start) == -1:
            is_negative = True

    # Detect surrounding context
    is_pct = _detect_percentage(text, match_end)
    is_bps = _detect_basis_points(text, match_end)
    is_ratio = _detect_ratio(text, match_end)
    is_cents = _detect_cents(text, match_end)
    scale = _detect_scale(text, match_end)
    currency = _detect_currency(text, match_start)

    # Compute value
    value = parsed * scale
    if is_negative:
        value = -abs(value)

    # Basis points conversion: 25 bps -> 0.25 (percent).
    # Must happen after sign handling so negative bps are also converted.
    if is_bps:
        value = value / 100.0

    # Cents conversion: "15.0 cents" -> 0.15 dollars
    # Only applied when is_cents is True; the caller can also use
    # expected_unit to decide downstream.
    # We store the raw cents value here; conversion to dollars happens
    # only if expected_unit hints at it (handled in extract_numeric).

    # Determine unit
    unit = _determine_unit(is_pct, is_ratio, is_cents, scale, currency)

    # Raw match substring
    raw = _compute_raw_match(text, match_start, match_end)

    return ExtractedValue(value=value, raw_match=raw, unit=unit, currency=currency)


def extract_all_candidates(text: str) -> List[ExtractedValue]:
    """Extract ALL numeric candidates from the text.

    Useful for debugging — lets you see every number the extractor found
    and how it interpreted each one.
    """
    candidates: List[ExtractedValue] = []
    for match in _NUM_PATTERN.finditer(text):
        ev = _extract_one(text, match)
        if ev.value is not None:
            candidates.append(ev)
    return candidates


def _normalize_expected_unit(expected_unit: Optional[str]) -> Optional[str]:
    """Normalise compound unit strings from facts.json to the base tokens the
    extractor understands.

    facts.json stores units like "sgd_millions" or "usd_billions", but the
    extractor's regex candidates carry plain tokens like "millions" or
    "billions".  Without this step, _score_candidate can never match the
    expected_unit and always falls back to position-based ranking.
    """
    if not expected_unit:
        return expected_unit
    mapping = {
        "sgd_millions": "millions",
        "usd_millions": "millions",
        "hkd_millions": "millions",
        "myr_millions": "millions",
        "sgd_billions": "billions",
        "usd_billions": "billions",
        "hkd_billions": "billions",
    }
    return mapping.get(expected_unit.lower().strip(), expected_unit)


def extract_numeric(text: str, expected_unit: Optional[str] = None) -> ExtractedValue:
    """Extract a single numeric value from LLM response text.

    This is the main entry point. It finds all numeric candidates, scores
    them for relevance (using expected_unit as the primary signal), and
    returns the best candidate.

    Args:
        text: The LLM response text to parse.
        expected_unit: Optional hint about what kind of value we expect.
            Common values: "percent", "billions", "millions", "ratio",
            "sgd", "usd", "cents". Helps disambiguate when the response
            contains multiple numbers.

    Returns:
        ExtractedValue with value=None if no numeric candidate found.
        Never guesses — a None value means extraction failed and should
        be recorded as such.
    """
    if not text or not text.strip():
        return ExtractedValue(value=None, raw_match=None, unit=None, currency=None)

    expected_unit = _normalize_expected_unit(expected_unit)
    candidates = extract_all_candidates(text)

    if not candidates:
        return ExtractedValue(value=None, raw_match=None, unit=None, currency=None)

    # If only one candidate, return it (possibly with cents conversion)
    if len(candidates) == 1:
        result = candidates[0]
        return _maybe_convert_cents(result, expected_unit)

    # Multiple candidates: score and rank
    scored = []
    for i, ev in enumerate(candidates):
        s = _score_candidate(ev, expected_unit, i, len(candidates))
        scored.append((s, i, ev))

    # Sort by score descending, then by original position ascending
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0][2]
    return _maybe_convert_cents(best, expected_unit)


def _maybe_convert_cents(
    ev: ExtractedValue, expected_unit: Optional[str]
) -> ExtractedValue:
    """Convert cents to dollars if the expected_unit hints at a dollar amount.

    "15.0 cents" -> 0.15 when expected_unit is "sgd", "usd", or similar
    dollar-denominated unit. Otherwise, leave as-is (the caller may want
    the raw cents value).
    """
    if ev.unit != "cents" or ev.value is None:
        return ev

    dollar_units = {"sgd", "usd", "hkd", "myr", "aud", "eur", "gbp", "dollars"}
    if expected_unit and expected_unit.lower().strip() in dollar_units:
        return ExtractedValue(
            value=ev.value / 100.0,
            raw_match=ev.raw_match,
            unit=expected_unit.lower().strip(),
            currency=ev.currency,
        )
    # If no expected_unit hint, keep as cents
    return ev
