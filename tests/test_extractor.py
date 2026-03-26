"""Tests for the numeric extractor module (30+ test cases).

Covers every format: percentages, currencies with scale,
comma formatting, negatives, ratios, cents, full-sentence extraction,
multi-number disambiguation, extraction failure, and edge cases.
"""

from __future__ import annotations

import pytest

from src.extractor import ExtractedValue, extract_all_candidates, extract_numeric


# ===========================================================================
# 1. Basic percentages
# ===========================================================================


class TestBasicPercentages:
    """Test percentage extraction in various notations."""

    def test_percent_symbol(self):
        result = extract_numeric("2.14%")
        assert result.value == pytest.approx(2.14)
        assert result.unit == "percent"

    def test_percent_word(self):
        result = extract_numeric("2.14 percent")
        assert result.value == pytest.approx(2.14)
        assert result.unit == "percent"

    def test_per_cent_two_words(self):
        result = extract_numeric("2.14 per cent")
        assert result.value == pytest.approx(2.14)
        assert result.unit == "percent"

    def test_percentage_word(self):
        result = extract_numeric("2.14 percentage")
        assert result.value == pytest.approx(2.14)
        assert result.unit == "percent"

    def test_zero_percent(self):
        result = extract_numeric("0%")
        assert result.value == pytest.approx(0.0)
        assert result.unit == "percent"

    def test_zero_point_zero_percent(self):
        result = extract_numeric("0.0%")
        assert result.value == pytest.approx(0.0)
        assert result.unit == "percent"

    def test_whole_number_percent(self):
        result = extract_numeric("18%")
        assert result.value == pytest.approx(18.0)
        assert result.unit == "percent"


# ===========================================================================
# 2. Currency with scale
# ===========================================================================


class TestCurrencyWithScale:
    """Test currency prefixes combined with scale words."""

    def test_dollar_billion(self):
        result = extract_numeric("$5.2 billion")
        assert result.value == pytest.approx(5_200_000_000)
        assert result.unit == "billions"

    def test_dollar_comma_million(self):
        result = extract_numeric("$5,200 million")
        assert result.value == pytest.approx(5_200_000_000)
        assert result.unit == "millions"

    def test_sgd_prefix_billion(self):
        result = extract_numeric("S$5.2B")
        assert result.value == pytest.approx(5_200_000_000)
        assert result.currency == "SGD"

    def test_sgd_word_lowercase_b(self):
        result = extract_numeric("SGD 5.2b")
        assert result.value == pytest.approx(5_200_000_000)
        assert result.currency == "SGD"

    def test_sgd_millions(self):
        result = extract_numeric("SGD 22,269 million")
        assert result.value == pytest.approx(22_269_000_000)
        assert result.currency == "SGD"

    def test_usd_prefix(self):
        result = extract_numeric("US$1.5 billion")
        assert result.value == pytest.approx(1_500_000_000)
        assert result.currency == "USD"

    def test_bare_dollar_defaults_usd(self):
        result = extract_numeric("$100 million")
        assert result.value == pytest.approx(100_000_000)
        assert result.currency == "USD"

    def test_sgd_thousands(self):
        result = extract_numeric("SGD 500 thousand")
        assert result.value == pytest.approx(500_000)
        assert result.currency == "SGD"


# ===========================================================================
# 3. Comma formatting
# ===========================================================================


class TestCommaFormatting:
    """Test comma-separated thousands."""

    def test_comma_thousands_with_decimal(self):
        result = extract_numeric("1,234.56")
        assert result.value == pytest.approx(1234.56)

    def test_comma_thousands_integer(self):
        result = extract_numeric("22,269")
        assert result.value == pytest.approx(22269.0)

    def test_comma_millions(self):
        result = extract_numeric("1,234,567")
        assert result.value == pytest.approx(1234567.0)

    def test_plain_decimal(self):
        result = extract_numeric("1234.56")
        assert result.value == pytest.approx(1234.56)


# ===========================================================================
# 4. Negative values
# ===========================================================================


class TestNegativeValues:
    """Test negative number extraction from various notations."""

    def test_minus_sign_percent(self):
        result = extract_numeric("-2.3%")
        assert result.value == pytest.approx(-2.3)
        assert result.unit == "percent"

    def test_parenthetical_negative(self):
        result = extract_numeric("(2.3%)")
        assert result.value == pytest.approx(-2.3)
        assert result.unit == "percent"

    def test_loss_of_keyword(self):
        result = extract_numeric("loss of 2.3%")
        assert result.value == pytest.approx(-2.3)
        assert result.unit == "percent"

    def test_decline_of_keyword(self):
        result = extract_numeric("decline of 2.3%")
        assert result.value == pytest.approx(-2.3)
        assert result.unit == "percent"

    def test_decrease_keyword(self):
        result = extract_numeric("a decrease of 500 million")
        assert result.value is not None
        assert result.value < 0

    def test_parenthetical_currency(self):
        result = extract_numeric("$(1.5 billion)")
        assert result.value is not None
        assert result.value < 0


# ===========================================================================
# 5. Ratio format
# ===========================================================================


class TestRatioFormat:
    """Test ratio extraction (x, times)."""

    def test_x_suffix(self):
        result = extract_numeric("0.85x")
        assert result.value == pytest.approx(0.85)
        assert result.unit == "ratio"

    def test_times_word(self):
        result = extract_numeric("0.85 times")
        assert result.value == pytest.approx(0.85)
        assert result.unit == "ratio"

    def test_ratio_with_expected_unit(self):
        result = extract_numeric("The ratio was 0.42x.", expected_unit="ratio")
        assert result.value == pytest.approx(0.42)
        assert result.unit == "ratio"


# ===========================================================================
# 6. Cents conversion
# ===========================================================================


class TestCentsConversion:
    """Test cents detection and conversion to dollars."""

    def test_cents_raw(self):
        """Without expected_unit hint, cents value is preserved as-is."""
        result = extract_numeric("15.0 cents")
        assert result.value == pytest.approx(15.0)
        assert result.unit == "cents"

    def test_cents_with_sgd_expected_unit(self):
        """With expected_unit='sgd', 15.0 cents -> 0.15 dollars."""
        result = extract_numeric("15.0 cents", expected_unit="sgd")
        assert result.value == pytest.approx(0.15)
        assert result.unit == "sgd"

    def test_cents_with_usd_expected_unit(self):
        """With expected_unit='usd', cents are converted."""
        result = extract_numeric("50 cents", expected_unit="usd")
        assert result.value == pytest.approx(0.50)
        assert result.unit == "usd"


# ===========================================================================
# 7. Full sentence extraction
# ===========================================================================


class TestFullSentenceExtraction:
    """Test extraction from natural-language financial sentences."""

    def test_nim_sentence(self):
        text = "DBS's net interest margin for FY2024 was 2.14%."
        result = extract_numeric(text, expected_unit="percent")
        assert result.value == pytest.approx(2.14)

    def test_total_income_sentence(self):
        text = "Total income rose 10% to SGD 22,269 million in 2024."
        result = extract_numeric(text, expected_unit="millions")
        assert result.value == pytest.approx(22_269_000_000)

    def test_cet1_sentence(self):
        text = "CET1 ratio was 14.8%, well above the regulatory minimum."
        result = extract_numeric(text, expected_unit="percent")
        assert result.value == pytest.approx(14.8)

    def test_revenue_sentence(self):
        text = "Group revenue was SGD 14,520 million for the financial year ended 31 March 2024."
        result = extract_numeric(text)
        assert result.value is not None
        # The number 14520 (possibly with scale) should be the primary candidate
        assert result.value >= 14000


# ===========================================================================
# 8. Multiple numbers — disambiguation
# ===========================================================================


class TestMultipleNumbers:
    """Test that the extractor picks the right number when multiple are present."""

    def test_picks_percent_with_unit_hint(self):
        text = "NIM improved from 2.16% in 2023 to 2.14% in 2024."
        result = extract_numeric(text, expected_unit="percent")
        assert result.value is not None
        assert result.unit == "percent"

    def test_picks_millions_with_unit_hint(self):
        text = "Total income rose 10% to SGD 22,269 million in 2024."
        result = extract_numeric(text, expected_unit="millions")
        assert result.value == pytest.approx(22_269_000_000)

    def test_extract_all_candidates_returns_multiple(self):
        text = "Revenue was $5.2 billion and profit was 2.14%."
        candidates = extract_all_candidates(text)
        assert len(candidates) >= 2


# ===========================================================================
# 9. Extraction failure
# ===========================================================================


class TestExtractionFailure:
    """Test that extraction returns value=None for unparseable text."""

    def test_no_data_available(self):
        result = extract_numeric("No data available")
        assert result.value is None

    def test_empty_string(self):
        result = extract_numeric("")
        assert result.value is None

    def test_none_like_whitespace(self):
        result = extract_numeric("   ")
        assert result.value is None

    def test_purely_alphabetic(self):
        result = extract_numeric("The net interest margin increased slightly.")
        assert result.value is None

    def test_raw_match_is_none_on_failure(self):
        result = extract_numeric("No numeric data here at all.")
        assert result.raw_match is None


# ===========================================================================
# 10. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases: very large numbers, zero, plain integers."""

    def test_very_large_number(self):
        result = extract_numeric("$1.5 trillion")
        assert result.value == pytest.approx(1_500_000_000_000)
        assert result.unit == "trillions"

    def test_plain_integer(self):
        result = extract_numeric("42")
        assert result.value == pytest.approx(42.0)

    def test_plain_decimal(self):
        result = extract_numeric("3.14159")
        assert result.value == pytest.approx(3.14159)

    def test_extracted_value_dataclass_fields(self):
        """Verify the ExtractedValue dataclass has expected attributes."""
        ev = ExtractedValue(value=1.0, raw_match="1.0", unit="percent", currency=None)
        assert ev.value == 1.0
        assert ev.raw_match == "1.0"
        assert ev.unit == "percent"
        assert ev.currency is None

    def test_extract_all_candidates_empty_on_no_numbers(self):
        candidates = extract_all_candidates("No numbers here")
        assert candidates == []

    def test_european_space_separated(self):
        """Test European-style space-separated thousands: '1 234.56'."""
        result = extract_numeric("1 234.56")
        assert result.value == pytest.approx(1234.56)

    def test_basis_points(self):
        # 25 basis points = 0.25% (1 bps = 0.01%)
        result = extract_numeric("25 basis points")
        assert result.value == pytest.approx(0.25)
        assert result.unit == "percent"

    def test_bps_abbreviation(self):
        # 25 bps = 0.25%
        result = extract_numeric("25 bps")
        assert result.value == pytest.approx(0.25)
        assert result.unit == "percent"
