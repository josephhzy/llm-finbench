"""Prompt template module for the LLM Financial Stability Bench.

Defines structured prompt templates for financial data extraction tasks.
Each template tests a different aspect of LLM behaviour: direct recall,
context-grounded extraction, comparative reasoning, and qualitative prose.

Why multiple templates?
Different phrasings surface different failure modes. A model might nail
a direct extraction question but hallucinate when asked to compare
year-over-year, or drop precision when embedding a figure in prose.
Measuring across templates lets us disentangle *what the model knows*
from *how the question is framed*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    name: str
    template: str
    description: str
    required_fields: List[str]

    def render(self, **kwargs: str) -> str:
        """Render the template with the provided keyword arguments.

        Raises KeyError if a required field is missing from kwargs.
        """
        missing = [f for f in self.required_fields if f not in kwargs]
        if missing:
            raise KeyError(
                f"Template '{self.name}' requires fields {missing} "
                f"which were not provided. Got: {list(kwargs.keys())}"
            )
        return self.template.format(**kwargs)


# ---------------------------------------------------------------------------
# Module-level template registry
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, PromptTemplate] = {
    "direct_extraction": PromptTemplate(
        name="direct_extraction",
        template=(
            "What was {company}'s {metric} for {period}? "
            "Provide only the numeric value with its unit."
        ),
        description="Direct factual question, minimal context",
        required_fields=["company", "metric", "period"],
    ),
    "contextual_extraction": PromptTemplate(
        name="contextual_extraction",
        template=(
            "In {company}'s {period} annual report, the following was stated: "
            '"{context}" Based on this, what was the exact {metric}? '
            "Respond with just the number and unit."
        ),
        description="Provides source context, tests extraction accuracy",
        required_fields=["company", "metric", "period", "context"],
    ),
    "comparative": PromptTemplate(
        name="comparative",
        template=(
            "How did {company}'s {metric} in {period} compare to the previous "
            "year? Provide the specific {period} figure."
        ),
        description="Asks for comparison, tests if model hallucinates prior-year data",
        required_fields=["company", "metric", "period"],
    ),
    "qualitative": PromptTemplate(
        name="qualitative",
        template=(
            "Describe {company}'s {metric_abbreviation} performance in {period}, "
            "including the specific reported figure."
        ),
        description="Open-ended, tests if model includes accurate figures in prose",
        required_fields=["company", "metric_abbreviation", "period"],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_template(name: str) -> PromptTemplate:
    """Get a template by name.

    Raises ValueError if the template name is not found in the registry.
    """
    if name not in TEMPLATES:
        raise ValueError(
            f"Unknown template '{name}'. Available templates: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[name]


def list_templates() -> List[str]:
    """Return a list of all available template names."""
    return list(TEMPLATES.keys())


def render_prompt(template_name: str, fact: dict) -> str:
    """Render a prompt template using fields from a fact dict.

    The fact dict is unpacked as keyword arguments into the template's
    format string. Only the fields declared in the template's
    ``required_fields`` need to be present, but extra keys are harmless
    (they are simply ignored by ``str.format``).

    Raises:
        ValueError: If *template_name* does not match a registered template.
        KeyError: If *fact* is missing one or more required fields.
    """
    template = get_template(template_name)  # raises ValueError
    return template.render(**fact)  # raises KeyError


def generate_all_prompts(
    facts: List[dict],
    template_names: List[str],
) -> List[dict]:
    """Generate every valid (fact, template) combination.

    For each fact and each requested template, attempts to render the
    prompt.  If a fact lacks a field required by the template (e.g. the
    ``qualitative`` template needs ``metric_abbreviation`` but the fact
    does not have it), the combination is skipped and a warning is logged.

    Returns:
        A list of dicts, each containing:
            - ``fact_id``:        the fact's ``id`` field
            - ``template_name``:  which template was used
            - ``rendered_prompt``: the final prompt string
            - ``fact``:           the original fact dict (for downstream use)
    """
    results: List[dict] = []

    for fact in facts:
        fact_id = fact.get("id", "unknown")

        for tname in template_names:
            # Validate template name early so typos surface immediately.
            try:
                template = get_template(tname)
            except ValueError:
                logger.error("Skipping unknown template '%s'.", tname)
                continue

            # Check required fields *before* rendering so we can give a
            # clear skip-warning rather than a cryptic KeyError.
            missing = [
                f
                for f in template.required_fields
                if f not in fact or fact.get(f) is None
            ]
            if missing:
                logger.warning(
                    "Skipping fact '%s' x template '%s': "
                    "fact is missing required fields %s.",
                    fact_id,
                    tname,
                    missing,
                )
                continue

            rendered = template.render(**fact)
            results.append(
                {
                    "fact_id": fact_id,
                    "template_name": tname,
                    "rendered_prompt": rendered,
                    "fact": fact,
                }
            )

    return results
