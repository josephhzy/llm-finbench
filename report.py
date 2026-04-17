"""CLI entry point for generating reports from saved evaluation results.

Takes a completed run_id, loads raw results from the results directory,
scores each evaluation group, and produces tabular reports (CSV or JSON)
plus a human-readable summary.

Usage examples::

    # Generate CSV reports for a specific run
    python report.py --run-id 20260321_143022_123456

    # Generate JSON reports to a custom directory
    python report.py --run-id 20260321_143022_123456 --format json --output my_reports

    # Use a non-default config and results directory
    python report.py --run-id 20260321_143022_123456 --config prod_config.yaml --input prod_results
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config
from src.reporter import generate_report


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the report generator."""
    parser = argparse.ArgumentParser(
        description="LLM Financial Stability Bench — Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python report.py --run-id 20260321_143022_123456\n"
            "  python report.py --run-id 20260321_143022_123456 --format json\n"
            "  python report.py --run-id 20260321_143022_123456 --output my_reports\n"
        ),
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Which run to generate reports for (required)",
    )
    parser.add_argument(
        "--input",
        default="results",
        help="Results directory (default: results)",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory (default: reports)",
    )
    parser.add_argument(
        "--format",
        default="csv",
        choices=["csv", "json"],
        help="Output format: csv or json (default: csv)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the report generation CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    # --- 1. Load configuration ---
    try:
        config = load_config(path=args.config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Generate reports ---
    print(f"Generating reports for run: {args.run_id}")
    print(f"  Results dir: {args.input}")
    print(f"  Output dir:  {args.output}")
    print(f"  Format:      {args.format}")
    print()

    try:
        generated_files = generate_report(
            run_id=args.run_id,
            results_dir=args.input,
            config=config,
            output_dir=args.output,
            output_format=args.format,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- 3. Print paths to generated files ---
    print("Reports generated successfully:")
    for report_name, file_path in generated_files.items():
        print(f"  {report_name:20s} -> {file_path}")


if __name__ == "__main__":
    main()
