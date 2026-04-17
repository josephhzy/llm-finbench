"""Tests for the prompt template module.

Covers template rendering, missing-field handling, unknown-template errors,
generate_all_prompts skipping behaviour, and list_templates.
"""

from __future__ import annotations

import pytest

from src.prompts import (
    TEMPLATES,
    PromptTemplate,
    generate_all_prompts,
    get_template,
    list_templates,
    render_prompt,
)


# ===========================================================================
# Template rendering with a real fact
# ===========================================================================


class TestTemplateRendering:
    """Test that each template renders correctly given a well-formed fact."""

    @pytest.fixture
    def full_fact(self):
        """A fact dict with all possible fields populated."""
        return {
            "id": "dbs_fy2024_nim",
            "company": "DBS",
            "metric": "Net Interest Margin",
            "metric_abbreviation": "NIM",
            "period": "FY2024",
            "value": 2.14,
            "unit": "percent",
            "context": "Group NIM for the full year 2024 was 2.14%.",
        }

    def test_direct_extraction_renders(self, full_fact):
        rendered = render_prompt("direct_extraction", full_fact)
        assert "DBS" in rendered
        assert "Net Interest Margin" in rendered
        assert "FY2024" in rendered
        assert "numeric value" in rendered.lower()

    def test_contextual_extraction_renders(self, full_fact):
        rendered = render_prompt("contextual_extraction", full_fact)
        assert "DBS" in rendered
        assert "Group NIM for the full year 2024 was 2.14%." in rendered
        assert "FY2024" in rendered

    def test_comparative_renders(self, full_fact):
        rendered = render_prompt("comparative", full_fact)
        assert "DBS" in rendered
        assert "compare" in rendered.lower()
        assert "FY2024" in rendered

    def test_qualitative_renders(self, full_fact):
        rendered = render_prompt("qualitative", full_fact)
        assert "DBS" in rendered
        assert "NIM" in rendered
        assert "FY2024" in rendered


# ===========================================================================
# Missing required fields
# ===========================================================================


class TestMissingFields:
    """Test that missing required fields raise KeyError."""

    def test_direct_extraction_missing_company(self):
        fact = {"metric": "NIM", "period": "FY2024"}
        with pytest.raises(KeyError, match="company"):
            render_prompt("direct_extraction", fact)

    def test_direct_extraction_missing_metric(self):
        fact = {"company": "DBS", "period": "FY2024"}
        with pytest.raises(KeyError, match="metric"):
            render_prompt("direct_extraction", fact)

    def test_contextual_missing_context(self):
        fact = {"company": "DBS", "metric": "NIM", "period": "FY2024"}
        with pytest.raises(KeyError, match="context"):
            render_prompt("contextual_extraction", fact)

    def test_qualitative_missing_metric_abbreviation(self):
        fact = {"company": "DBS", "period": "FY2024"}
        with pytest.raises(KeyError, match="metric_abbreviation"):
            render_prompt("qualitative", fact)


# ===========================================================================
# Unknown template
# ===========================================================================


class TestUnknownTemplate:
    """Test that requesting a non-existent template raises ValueError."""

    def test_get_template_unknown(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent_template")

    def test_render_prompt_unknown(self):
        with pytest.raises(ValueError, match="Unknown template"):
            render_prompt("does_not_exist", {"company": "DBS"})


# ===========================================================================
# generate_all_prompts
# ===========================================================================


class TestGenerateAllPrompts:
    """Test batch prompt generation, including skip behaviour."""

    def test_generates_valid_combinations(self, sample_fact):
        """A fact with all standard fields works for direct_extraction."""
        results = generate_all_prompts([sample_fact], ["direct_extraction"])
        assert len(results) == 1
        assert results[0]["fact_id"] == sample_fact["id"]
        assert results[0]["template_name"] == "direct_extraction"
        assert isinstance(results[0]["rendered_prompt"], str)
        assert results[0]["fact"] is sample_fact

    def test_skips_fact_missing_required_field(self):
        """A fact missing 'context' should be skipped for contextual_extraction."""
        fact_no_context = {
            "id": "test_fact",
            "company": "DBS",
            "metric": "NIM",
            "period": "FY2024",
            # "context" is intentionally missing
        }
        results = generate_all_prompts([fact_no_context], ["contextual_extraction"])
        assert len(results) == 0

    def test_skips_fact_missing_metric_abbreviation(self):
        """A fact with metric_abbreviation=None should be skipped for qualitative."""
        fact_no_abbrev = {
            "id": "test_fact",
            "company": "DBS",
            "metric": "Total Income",
            "period": "FY2024",
            # metric_abbreviation not present
        }
        results = generate_all_prompts([fact_no_abbrev], ["qualitative"])
        assert len(results) == 0

    def test_multiple_facts_multiple_templates(self, sample_facts):
        """Test generating prompts across multiple facts and templates."""
        # Use first 3 facts and 2 templates
        facts_subset = sample_facts[:3]
        templates = ["direct_extraction", "comparative"]
        results = generate_all_prompts(facts_subset, templates)
        # Every fact should match both templates (they require company, metric, period)
        assert len(results) == 6

    def test_unknown_template_in_list_is_skipped(self, sample_fact):
        """An unknown template name in the list is skipped (not raised)."""
        results = generate_all_prompts(
            [sample_fact], ["direct_extraction", "totally_fake_template"]
        )
        # Only direct_extraction should produce a result
        assert len(results) == 1
        assert results[0]["template_name"] == "direct_extraction"

    def test_empty_facts_list(self):
        results = generate_all_prompts([], ["direct_extraction"])
        assert results == []

    def test_empty_templates_list(self, sample_fact):
        results = generate_all_prompts([sample_fact], [])
        assert results == []


# ===========================================================================
# list_templates
# ===========================================================================


class TestListTemplates:
    """Test that list_templates returns all registered templates."""

    def test_returns_all_four(self):
        names = list_templates()
        assert len(names) == 4
        assert "direct_extraction" in names
        assert "contextual_extraction" in names
        assert "comparative" in names
        assert "qualitative" in names

    def test_matches_templates_dict_keys(self):
        names = list_templates()
        assert set(names) == set(TEMPLATES.keys())


# ===========================================================================
# PromptTemplate dataclass
# ===========================================================================


class TestPromptTemplateDataclass:
    """Test the PromptTemplate dataclass directly."""

    def test_render_with_extra_keys(self):
        """Extra kwargs beyond required_fields should be silently ignored."""
        tpl = get_template("direct_extraction")
        rendered = tpl.render(
            company="DBS",
            metric="NIM",
            period="FY2024",
            extra_field="should_be_ignored",
        )
        assert "DBS" in rendered

    def test_template_has_description(self):
        for name, tpl in TEMPLATES.items():
            assert tpl.description, f"Template '{name}' has empty description"

    def test_template_has_required_fields(self):
        for name, tpl in TEMPLATES.items():
            assert len(tpl.required_fields) > 0, (
                f"Template '{name}' has no required_fields"
            )
