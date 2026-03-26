"""CLI entry point for running LLM Financial Stability Bench evaluations.

Parses command-line arguments, loads configuration and ground-truth facts,
estimates API cost, and delegates to the EvaluationEngine. Supports quick
mode for development iteration, dry-run for cost previews, and checkpoint
resume for crash recovery.

Usage examples::

    # Full evaluation (will prompt if cost exceeds threshold)
    python evaluate.py

    # Quick mode: single temperature, 3 runs, subset of facts
    python evaluate.py --quick

    # Dry run: print prompts and cost, make zero API calls
    python evaluate.py --dry-run

    # Filter to one company, override run count
    python evaluate.py --company DBS --n-runs 5

    # Resume a crashed run
    python evaluate.py --resume 20260321_143022_123456
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import os

from dotenv import load_dotenv

from src.config import EvaluationConfig, load_config
from src.engine import EvaluationEngine

# Load .env file if present so users can set API keys there.
# Has no effect if .env doesn't exist — os.environ values always take precedence.
load_dotenv()

# Map of provider name -> required environment variable.
_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _check_api_key(provider: str) -> None:
    """Verify the API key for the active provider is set before running.

    Only checks the key for the configured provider — an empty or missing
    key for any other provider is silently ignored.

    Raises SystemExit with a clear message if the key is missing.
    """
    env_var = _PROVIDER_KEY_MAP.get(provider)
    if env_var is None:
        # Unknown provider — the adapter registry will give a better error later.
        return
    if not os.environ.get(env_var):
        print(
            f"Error: {env_var} is not set.\n"
            f"Add it to your .env file:\n\n"
            f"  {env_var}=your-api-key-here\n",
            file=sys.stderr,
        )
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="LLM Financial Stability Bench — Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py --quick              # Fast dev run\n"
            "  python evaluate.py --dry-run            # Cost estimate only\n"
            "  python evaluate.py --company DBS        # Single company\n"
            "  python evaluate.py --resume <run_id>    # Resume crashed run\n"
        ),
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: single temperature (0.0), 3 runs, subset of facts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and cost estimate, make zero API calls",
    )
    parser.add_argument(
        "--company",
        default=None,
        help='Filter to one company (e.g., "DBS")',
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override runs_per_combination from config",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_ID",
        help="Resume a previous run by its run_id",
    )

    return parser.parse_args()


def _load_facts(path: str) -> list[dict]:
    """Load ground-truth facts from a JSON file.

    Parameters
    ----------
    path : str
        Path to facts.json.

    Returns
    -------
    list[dict]
        List of fact dictionaries, each containing at minimum an "id" field.

    Raises
    ------
    FileNotFoundError
        If the facts file does not exist.
    json.JSONDecodeError
        If the file contains invalid JSON.
    ValueError
        If the file does not contain a list of facts.
    """
    facts_path = Path(path)
    if not facts_path.exists():
        raise FileNotFoundError(f"Facts file not found: {facts_path.resolve()}")

    with open(facts_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both raw array and wrapper object with "facts" key
    if isinstance(data, dict) and "facts" in data:
        data = data["facts"]

    if not isinstance(data, list):
        raise ValueError(
            f"facts.json must contain a JSON array or object with 'facts' key, "
            f"got {type(data).__name__}"
        )

    return data


def main() -> None:
    """Main entry point for the evaluation CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    # --- 1. Load configuration ---
    try:
        config = load_config(path=args.config, quick=args.quick)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Verify API key for the active provider (skip for dry-run) ---
    if not args.dry_run:
        _check_api_key(config.model.provider)

    # --- 3. Override runs_per_combination if --n-runs provided ---
    if args.n_runs is not None:
        if args.n_runs < 1:
            print("Error: --n-runs must be >= 1", file=sys.stderr)
            sys.exit(1)
        # EvaluationConfig is frozen, so reconstruct with the override
        config = replace(
            config,
            evaluation=EvaluationConfig(
                temperatures=config.evaluation.temperatures,
                runs_per_combination=args.n_runs,
                templates=config.evaluation.templates,
            ),
        )

    # --- 4. Load ground-truth facts ---
    facts_path = "ground_truth/facts.json"
    try:
        facts = _load_facts(facts_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"Facts loading error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not facts:
        print("Error: facts.json is empty — nothing to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Apply company filter FIRST, before max_facts.
    # Order matters: --quick --company DBS should mean "DBS facts, capped at 6",
    # not "first 6 facts from the file, then filtered to DBS" (which can silently
    # return 0 facts if DBS facts happen to be positioned after the first 6).
    if args.company:
        filter_lower = args.company.lower()
        all_facts = facts  # Keep unfiltered list for the error message
        facts = [f for f in facts if f.get("company", "").lower() == filter_lower]
        if not facts:
            available = sorted(set(f.get("company", "?") for f in all_facts))
            print(
                f"Error: no facts match company '{args.company}'. "
                f"Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Company filter '{args.company}': {len(facts)}/{len(all_facts)} facts selected")

    # Apply max_facts limit in quick mode (after company filter).
    if args.quick:
        max_facts = config.quick_mode.max_facts
        original_count = len(facts)
        if original_count > max_facts:
            facts = facts[:max_facts]
            print(f"Quick mode: capped to {max_facts} of {original_count} facts")

    print(f"Loaded {len(facts)} facts from {facts_path}")

    # --- 4. Create engine ---
    try:
        engine = EvaluationEngine(config, facts)
    except Exception as exc:
        print(f"Engine initialisation error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 5. Run the evaluation ---
    # engine.run() prints a full evaluation plan (calls, cost, model, etc.)
    # before making any API calls — no need to duplicate it here.
    try:
        run_id = engine.run(
            dry_run=args.dry_run,
            company_filter=None,  # Already applied above, before max_facts slice
            resume_run_id=args.resume,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except ValueError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 6. Print results location and next steps ---
    print(f"Results saved to results/{run_id}/")

    if not args.dry_run:
        print(f"Run: python report.py --run-id {run_id} to generate reports")


if __name__ == "__main__":
    main()
