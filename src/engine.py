"""Execution engine for the LLM Financial Stability Bench.

Orchestrates the full evaluation sweep: generates all (fact x template x
temperature x run) combinations, calls the LLM adapter for each, extracts
numeric values from responses, and checkpoints results to disk.

Why raw storage only?
The engine stores CallRecord objects as-is — no aggregation, no scoring, no
flagging. Aggregation happens in the reporter, where it can be re-sliced in
different ways without re-running API calls. This separation means a $15
evaluation run produces data that can be analysed from multiple angles.

Checkpointing rationale:
A full evaluation run may exceed 3000 API calls over several hours. If the
process crashes at call 2500, we must not lose the first 2500 results. The
engine checkpoints every `save_interval` calls (default 50) and on resume
skips any (fact_id, template, temperature, run_index) combination that is
already recorded. This means a resumed run produces byte-identical results
to one that never crashed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.adapters import get_adapter
from src.config import AppConfig, snapshot_config
from src.extractor import extract_numeric
from src.prompts import generate_all_prompts

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """A single API call result with all metadata needed for downstream scoring.

    Every field is stored as-is from the adapter response and extractor output.
    No aggregation or interpretation happens here — that is the reporter's job.

    Attributes:
        fact_id: Identifier of the ground-truth fact being tested.
        template_name: Which prompt template was used.
        temperature: Sampling temperature for this call.
        run_index: Which repetition within (fact, template, temperature).
        raw_response: The full LLM response text.
        extracted_value: Numeric value parsed from the response, or None if
            extraction failed. None is data — it means the LLM responded in
            an unparseable format and should be tracked separately.
        extracted_unit: Unit string from the extractor (e.g. "percent",
            "billions"), or None.
        latency_ms: Wall-clock time for the API call in milliseconds.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        finish_reason: "end_turn", "max_tokens", or "error".
        timestamp: ISO-8601 timestamp of when the call was made.
    """

    fact_id: str
    template_name: str
    temperature: float
    run_index: int
    raw_response: str
    extracted_value: Optional[float]
    extracted_unit: Optional[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str
    timestamp: str  # ISO format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completed_key(
    fact_id: str,
    template_name: str,
    temperature: float,
    run_index: int,
) -> str:
    """Build a deterministic string key for checkpoint deduplication.

    We use a string rather than a tuple because JSON serialisation of sets
    is awkward (JSON has no set type). A pipe-delimited string is simple,
    unambiguous, and round-trips cleanly through JSON.
    """
    return f"{fact_id}|{template_name}|{temperature}|{run_index}"


def _serialize_records(records: List[CallRecord]) -> List[dict]:
    """Convert CallRecords to plain dicts for JSON serialisation."""
    return [asdict(r) for r in records]


def _deserialize_records(raw_list: List[dict]) -> List[CallRecord]:
    """Reconstruct CallRecords from dicts loaded from JSON."""
    results: List[CallRecord] = []
    for d in raw_list:
        results.append(CallRecord(
            fact_id=d["fact_id"],
            template_name=d["template_name"],
            temperature=d["temperature"],
            run_index=d["run_index"],
            raw_response=d["raw_response"],
            extracted_value=d.get("extracted_value"),
            extracted_unit=d.get("extracted_unit"),
            latency_ms=d["latency_ms"],
            input_tokens=d["input_tokens"],
            output_tokens=d["output_tokens"],
            finish_reason=d["finish_reason"],
            timestamp=d["timestamp"],
        ))
    return results


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EvaluationEngine:
    """Orchestrates the full LLM evaluation sweep with checkpointing.

    The engine is stateless between runs — all persistent state lives on disk
    in the results directory. This makes it safe to construct a new engine
    instance and resume a prior run just by passing the same run_id.

    Typical usage::

        config = load_config("config.yaml")
        facts = load_facts("ground_truth/facts.json")
        engine = EvaluationEngine(config, facts)

        # Preview cost before committing
        run_id = engine.run(dry_run=True)

        # Execute for real
        run_id = engine.run()
    """

    def __init__(self, config: AppConfig, facts: List[dict]) -> None:
        self._config = config
        self._facts = facts
        self._adapter = None  # Lazy-initialized on first API call
        self._results_dir = config.checkpoint.directory
        self._save_interval = config.checkpoint.save_interval

    def _get_adapter(self):
        """Lazy-initialize the LLM adapter on first use.

        Deferred so that dry-run and cost estimation work even when the
        provider SDK (e.g. anthropic) is not installed.
        """
        if self._adapter is None:
            adapter_cls = get_adapter(self._config.model.provider)

            api_config: Dict[str, Any] = {
                "timeout_seconds": self._config.api.timeout_seconds,
                "max_retries": self._config.api.max_retries,
                "base_backoff_seconds": self._config.api.base_backoff_seconds,
                "max_backoff_seconds": self._config.api.max_backoff_seconds,
                "rate_limit_rpm": self._config.api.rate_limit_rpm,
                "avg_input_tokens": self._config.cost.avg_input_tokens,
                "avg_output_tokens": self._config.cost.avg_output_tokens,
                "price_per_1k_input": self._config.cost.price_per_1k_input,
                "price_per_1k_output": self._config.cost.price_per_1k_output,
            }

            self._adapter = adapter_cls(
                model_name=self._config.model.name,
                max_tokens=self._config.model.max_tokens,
                api_config=api_config,
            )
        return self._adapter

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_total_calls(self, facts: Optional[List[dict]] = None) -> int:
        """Count valid (fact x template x temperature x run) combinations.

        Not every fact has every field required by every template (e.g. the
        qualitative template needs metric_abbreviation). We delegate to
        generate_all_prompts to compute the actual valid set, then multiply
        by temperatures and runs.
        """
        target_facts = facts if facts is not None else self._facts
        prompts = generate_all_prompts(
            target_facts,
            list(self._config.evaluation.templates),
        )
        n_prompts = len(prompts)
        n_temps = len(self._config.evaluation.temperatures)
        n_runs = self._config.evaluation.runs_per_combination
        return n_prompts * n_temps * n_runs

    def estimate_cost(self, facts: Optional[List[dict]] = None) -> float:
        """Estimate total USD cost for the evaluation run.

        Uses cost parameters from config directly so this works even
        without the provider SDK installed (needed for --dry-run).
        """
        total_calls = self.estimate_total_calls(facts)
        cost = self._config.cost
        input_cost = (cost.avg_input_tokens / 1000) * cost.price_per_1k_input
        output_cost = (cost.avg_output_tokens / 1000) * cost.price_per_1k_output
        return total_calls * (input_cost + output_cost)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        """Return the directory for a specific run's artifacts."""
        return Path(self._results_dir) / run_id

    def _checkpoint_path(self, run_id: str) -> str:
        """Return the path to the checkpoint file for a run."""
        return str(self._run_dir(run_id) / "checkpoint.json")

    def _save_checkpoint(
        self,
        run_id: str,
        records: List[CallRecord],
        completed_keys: Set[str],
    ) -> None:
        """Persist current progress to disk.

        Writes atomically by first writing to a temp file then renaming,
        so a crash mid-write cannot corrupt the checkpoint.
        """
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "run_id": run_id,
            "n_completed": len(records),
            "completed_keys": sorted(completed_keys),
            "records": _serialize_records(records),
        }

        checkpoint_file = run_dir / "checkpoint.json"
        tmp_file = run_dir / "checkpoint.json.tmp"

        with open(tmp_file, "w", encoding="utf-8") as fh:
            json.dump(checkpoint_data, fh, indent=2, ensure_ascii=False)

        # Atomic rename (on POSIX; on Windows this is best-effort but still
        # far safer than writing directly to the target file).
        tmp_file.replace(checkpoint_file)

        logger.debug(
            "Checkpoint saved: %d records to %s",
            len(records),
            checkpoint_file,
        )

    def _load_checkpoint(self, run_id: str) -> Tuple[List[CallRecord], Set[str]]:
        """Load a previous checkpoint, if one exists.

        Returns (records, completed_keys). If no checkpoint file is found,
        returns ([], set()) so the caller can proceed from scratch.
        """
        checkpoint_file = Path(self._checkpoint_path(run_id))

        if not checkpoint_file.exists():
            return [], set()

        try:
            with open(checkpoint_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            records = _deserialize_records(data.get("records", []))
            completed_keys = set(data.get("completed_keys", []))

            logger.info(
                "Resumed from checkpoint: %d completed calls loaded from %s",
                len(records),
                checkpoint_file,
            )
            return records, completed_keys

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Corrupt checkpoint at %s (%s). Starting fresh.",
                checkpoint_file,
                exc,
            )
            return [], set()

    def _save_final_results(
        self,
        run_id: str,
        records: List[CallRecord],
        evaluated_facts: List[dict],
    ) -> None:
        """Write the completed results file.

        This is the authoritative artifact — the checkpoint file is only
        for crash recovery. The results file is what the reporter reads.

        Parameters
        ----------
        evaluated_facts : List[dict]
            The actual subset of facts evaluated in this run (after company
            filter and quick-mode max_facts are applied). Storing all facts
            when only a subset was evaluated would make the artifact misleading.
        """
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        results_data = {
            "run_id": run_id,
            "config": snapshot_config(self._config),
            "facts": evaluated_facts,
            "n_records": len(records),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "call_records": _serialize_records(records),
        }

        results_file = run_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as fh:
            json.dump(results_data, fh, indent=2, ensure_ascii=False)

        logger.info("Final results saved: %d records to %s", len(records), results_file)

    def _save_config_snapshot(self, run_id: str) -> None:
        """Save the full config to the run directory for reproducibility.

        Anyone looking at results/{run_id}/ should be able to see exactly
        which parameters produced those results, even if config.yaml has
        since been edited.
        """
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        config_file = run_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as fh:
            json.dump(snapshot_config(self._config), fh, indent=2, ensure_ascii=False)

        logger.info("Config snapshot saved to %s", config_file)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        dry_run: bool = False,
        resume_run_id: Optional[str] = None,
    ) -> str:
        """Execute the full evaluation sweep.

        Parameters
        ----------
        dry_run : bool
            If True, prints estimated calls, cost, and sample prompts, but
            makes zero API calls. Use this to verify the pipeline before
            committing spend.
        resume_run_id : str, optional
            If provided, resume a previous run instead of starting fresh.
            The engine loads the checkpoint, skips completed calls, and
            continues from where it left off.

        Returns
        -------
        str
            The run_id, which identifies the results directory:
            ``results/{run_id}/``.

        Note
        ----
        Company filtering is handled by the caller (evaluate.py) before
        engine construction. The engine evaluates all facts it was given.
        """
        facts = self._facts

        # ----- 1. Generate all prompts -----
        prompts = generate_all_prompts(
            facts,
            list(self._config.evaluation.templates),
        )

        if not prompts:
            logger.error("No valid (fact, template) combinations generated.")
            raise ValueError(
                "No valid prompts generated. Check that facts have the "
                "fields required by the configured templates."
            )

        temperatures = self._config.evaluation.temperatures
        n_runs = self._config.evaluation.runs_per_combination
        total_calls = len(prompts) * len(temperatures) * n_runs
        estimated_cost = self.estimate_cost(facts=facts)

        # ----- 2. Print cost estimate -----
        print(f"\n{'=' * 60}")
        print(f"  LLM Financial Stability Bench — Evaluation Plan")
        print(f"{'=' * 60}")
        print(f"  Model:         {self._config.model.provider}/{self._config.model.name}")
        print(f"  Facts:         {len(facts)}")
        print(f"  Templates:     {len(self._config.evaluation.templates)}")
        print(f"  Temperatures:  {temperatures}")
        print(f"  Runs/combo:    {n_runs}")
        print(f"  Valid prompts: {len(prompts)} (fact x template combinations)")
        print(f"  Total calls:   {total_calls}")
        print(f"  Est. cost:     ${estimated_cost:.4f}")
        print(f"  Save interval: every {self._save_interval} calls")
        print(f"{'=' * 60}\n")

        # ----- 3. Dry run: show samples and exit -----
        if dry_run:
            n_samples = min(3, len(prompts))
            print(f"DRY RUN — showing {n_samples} sample prompt(s):\n")
            for i, p in enumerate(prompts[:n_samples]):
                print(f"  [{i + 1}] fact_id={p['fact_id']}, "
                      f"template={p['template_name']}")
                print(f"      {p['rendered_prompt'][:120]}...")
                print()
            print("No API calls made. Exiting dry run.\n")

            # Still return a run_id so callers have a consistent interface.
            run_id = resume_run_id or self._generate_run_id()
            return run_id

        # ----- 4. Cost confirmation gate -----
        threshold = self._config.cost.confirmation_threshold
        if estimated_cost > threshold:
            print(
                f"Estimated cost (${estimated_cost:.4f}) exceeds threshold "
                f"(${threshold:.2f})."
            )
            try:
                answer = input("Proceed? [y/N]: ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                print("Aborted by user.")
                logger.info("Run aborted: user declined cost confirmation.")
                raise KeyboardInterrupt("User declined cost confirmation.")

        # ----- 5. Generate or reuse run_id -----
        run_id = resume_run_id or self._generate_run_id()

        # ----- 6. Save config snapshot -----
        self._save_config_snapshot(run_id)

        # ----- 7. Load checkpoint if resuming -----
        records, completed_keys = self._load_checkpoint(run_id)
        calls_since_checkpoint = 0
        completed_count = len(records)

        # ----- 8. Execute the sweep -----
        logger.info("Starting evaluation run %s (%d total calls).", run_id, total_calls)
        start_wall = time.monotonic()
        errors_count = 0

        # Progress bar: only shown in non-dry-run mode and when tqdm is installed.
        progress_bar = None
        if tqdm is not None and not dry_run:
            progress_bar = tqdm(
                total=total_calls,
                initial=completed_count,
                desc="Evaluating",
                unit="call",
            )

        for prompt_entry in prompts:
            fact_id = prompt_entry["fact_id"]
            template_name = prompt_entry["template_name"]
            rendered_prompt = prompt_entry["rendered_prompt"]
            fact = prompt_entry["fact"]

            # Determine expected unit from the fact metadata, if available.
            # This helps the extractor disambiguate multi-number responses.
            expected_unit = fact.get("unit")

            for temperature in temperatures:
                for run_index in range(n_runs):
                    key = _make_completed_key(
                        fact_id, template_name, temperature, run_index,
                    )

                    # Skip if already completed (checkpoint resume).
                    if key in completed_keys:
                        continue

                    # --- Make the API call ---
                    response = self._get_adapter().generate(rendered_prompt, temperature)

                    # --- Extract numeric value ---
                    # Skip extraction for error responses: the text is an error
                    # message (e.g. "[ERROR] RateLimitError: 429"), not an LLM
                    # answer. Numbers in error strings (HTTP codes, timeouts)
                    # would corrupt the checkpoint and downstream scoring.
                    if response.finish_reason == "error":
                        extracted_value: Optional[float] = None
                        extracted_unit: Optional[str] = None
                    else:
                        extraction = extract_numeric(response.text, expected_unit)
                        extracted_value = extraction.value
                        extracted_unit = extraction.unit

                    # --- Build record ---
                    record = CallRecord(
                        fact_id=fact_id,
                        template_name=template_name,
                        temperature=temperature,
                        run_index=run_index,
                        raw_response=response.text,
                        extracted_value=extracted_value,
                        extracted_unit=extracted_unit,
                        latency_ms=response.latency_ms,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        finish_reason=response.finish_reason,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    records.append(record)
                    completed_keys.add(key)
                    completed_count += 1
                    calls_since_checkpoint += 1

                    if response.finish_reason == "error":
                        errors_count += 1

                    # --- Progress logging ---
                    pct = (completed_count / total_calls) * 100.0
                    fact_label = fact.get("company", fact_id)
                    metric_label = fact.get("metric", "")
                    if response.finish_reason == "error":
                        call_suffix = " [API ERROR]"
                    elif extracted_value is None:
                        call_suffix = " [EXTRACTION FAILED]"
                    else:
                        call_suffix = ""
                    logger.info(
                        "Call %d/%d (%.1f%%) -- %s %s @ temp %.1f, run %d%s",
                        completed_count,
                        total_calls,
                        pct,
                        fact_label,
                        metric_label,
                        temperature,
                        run_index + 1,
                        call_suffix,
                    )

                    # --- Update progress bar ---
                    if progress_bar is not None:
                        progress_bar.update(1)

                    # --- Checkpoint ---
                    if calls_since_checkpoint >= self._save_interval:
                        self._save_checkpoint(run_id, records, completed_keys)
                        calls_since_checkpoint = 0

        # Close progress bar
        if progress_bar is not None:
            progress_bar.close()

        # ----- 9. Final save -----
        elapsed_s = time.monotonic() - start_wall
        self._save_checkpoint(run_id, records, completed_keys)
        self._save_final_results(run_id, records, evaluated_facts=facts)

        # ----- 10. Summary -----
        extraction_failures = sum(
            1 for r in records
            if r.extracted_value is None and r.finish_reason != "error"
        )
        total_tokens_in = sum(r.input_tokens for r in records)
        total_tokens_out = sum(r.output_tokens for r in records)

        print(f"\n{'=' * 60}")
        print(f"  Evaluation Complete — Run {run_id}")
        print(f"{'=' * 60}")
        print(f"  Total calls:          {len(records)}")
        print(f"  Errors (adapter):     {errors_count}")
        print(f"  Extraction failures:  {extraction_failures}")
        print(f"  Total input tokens:   {total_tokens_in:,}")
        print(f"  Total output tokens:  {total_tokens_out:,}")
        print(f"  Wall-clock time:      {elapsed_s:.1f}s")
        print(f"  Results directory:     {self._run_dir(run_id)}")
        print(f"{'=' * 60}\n")

        logger.info(
            "Run %s complete: %d calls, %d errors, %d extraction failures, %.1fs",
            run_id,
            len(records),
            errors_count,
            extraction_failures,
            elapsed_s,
        )

        return run_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a timestamp-based run ID.

        Format: YYYYMMDD_HHMMSS_ffffff — human-readable, sortable, and
        unique to the microsecond (sufficient for sequential runs).
        """
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
