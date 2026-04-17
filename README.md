# LLM Financial Stability Bench

**What it is:** A reproducible evaluation harness that measures how *stable* an LLM is when repeatedly asked the same financial-extraction question across prompt templates and sampling temperatures. Composite stability = weighted combination of semantic consistency, factual (modal-value) consistency, and hallucination rate against context-quoted ground truth (pending page verification).

**Scope:** 29 preliminary facts across 5 SGX-listed companies (DBS, OCBC, UOB, Singtel, CapitaLand). Ground truth is context-quoted and pending primary-source page verification — see [`ground_truth/PROVENANCE.md`](ground_truth/PROVENANCE.md).

### Headline results

The table below reports a **partial run** (single model, no retries-to-exhaustion, no full second-person ground-truth verification). Treat it as a preliminary sanity check, not a leaderboard. Numbers are point estimates with bootstrap 95% CIs computed by `evaluate_with_ci.py` over `reports/per_fact_report.csv`, resampled at the fact level (N = 29, 10,000 bootstrap iterations, seed = 42).

| Model | Composite stability (95% CI) | Factual consistency (95% CI) | Hallucination rate (95% CI) | Extraction failure (95% CI) | Green / Yellow / Red | Notes |
|-------|------------------------------|------------------------------|-----------------------------|------------------------------|----------------------|-------|
| `openai/gpt-5-nano` | 0.594 [0.576, 0.612] | 0.538 [0.515, 0.563] | 0.653 [0.611, 0.694] | 0.172 [0.125, 0.220] | 145 / 122 / 268 (groups) | Partial run `20260401_040014_891022`, N = 29 facts, 4 templates × 5 temperatures × 10 runs = 535 groups. Context-quoted ground truth pending page verification. |

Run `python evaluate.py --dry-run` first to confirm scope and cost before launching a paid benchmark. After a run completes, `python report.py --run-id <id>` regenerates all CSVs and (new) automatically emits `reports/ci_summary.csv` via `evaluate_with_ci.py`.

### Key limitations (read before interpreting any numbers)

- **Stability is not accuracy.** A model that confidently returns the wrong number every time scores 1.0 on factual consistency. Hallucination rate is a separate axis and must be read alongside the composite.
- **Ground truth is context-quoted, not yet page-verified.** The 29 facts are sourced from publicly available FY2024 annual reports with verbatim context quotes, but page-level citations (`page: null` in `facts.json`) and second-person review are still pending ([`ground_truth/PROVENANCE.md`](ground_truth/PROVENANCE.md)).
- **N = 29 facts across 5 companies.** Conclusions at that sample size are bounded — report confidence intervals, not point estimates, and do not over-generalise to the SGX market ([`docs/STATISTICAL_RIGOR.md`](docs/STATISTICAL_RIGOR.md)).
- **Generic embedding model.** Semantic similarity uses `all-MiniLM-L6-v2`, a general-purpose encoder. A FinBERT code path is now wired in (`scripts/rescore_semantic_finbert.py`, `src/scorer.py::_encode_with_hf_transformer`); on a synthetic paraphrase corpus it shifts mean semantic consistency by roughly −0.08 vs MiniLM with near-zero rank correlation. Headline re-scoring on real responses is pending a new run that persists `results.json`.
- **5% tolerance is arbitrary.** A 5% relative tolerance on a SGD-billion figure is SGD-50M of slack. Sensitivity of the hallucination rate to the tolerance choice should be swept, not assumed ([`docs/SENSITIVITY_ANALYSIS.md`](docs/SENSITIVITY_ANALYSIS.md)).

Live Dashboard: https://llm-finbench.streamlit.app/

---

## What this benchmark measures (and doesn't)

**Measures:**

- **Intra-model stability.** Does the same model give the same answer when the prompt is rephrased four ways, the temperature is swept across five values, and each condition is repeated N times? This surfaces prompt sensitivity and sampling variance that single-query testing hides.
- **Bimodal failure modes.** Factual consistency uses the *modal* extracted value, not the mean, because a model that alternates between two wrong answers has mean ≈ midpoint-between-the-answers, which flatters it. Modal analysis reports "5/10 runs said X, 5/10 runs said Y" honestly.
- **Hallucination rate against a fixed (context-quoted) ground truth.** How often does the extracted number deviate from the annual-report figure beyond a relative tolerance?
- **Extraction parseability.** How often does the model's response fail to yield a parseable numeric value at all? Reported separately from hallucination rate.

**Does NOT measure:**

- **Reading comprehension or retrieval.** Prompts include enough context that the model should know the answer; this is not a RAG benchmark.
- **Accuracy on unseen companies.** The 5 companies are all SGX-listed banks and Singapore names. Generalisation to US, UK, or EU names is not tested.
- **Robustness to adversarial prompts.** Templates vary phrasing, not adversarial injection.
- **Ground-truth validity beyond the current preliminary set.** If a fact is wrong, every model is penalised identically — the benchmark measures stability against *that* label, not against the underlying truth.
- **Absolute "which model is best."** Composite scores are meaningful for A/B comparisons at a fixed configuration; they are not a universal leaderboard number.

This is a measurement tool, not a demo app. Precision, reproducibility, and statistical rigour are the priorities.

---

## Why Evaluation Matters

LLM hallucinations in financial contexts carry real consequences: incorrect figures in analyst reports, misstated regulatory metrics, or fabricated data points that propagate through downstream decision-making. As financial institutions adopt LLMs for data extraction and reporting, systematic evaluation is not optional -- it is a prerequisite for responsible deployment. Regulators (MAS, SEC, EBA) increasingly expect model validation evidence for AI systems used in financial services, making reproducible evaluation frameworks essential infrastructure.

---

## Responsible AI

This framework supports responsible AI practices in financial services through:

- **Systematic evaluation**: Every model is tested across multiple prompt strategies, temperatures, and repetitions to surface inconsistencies that single-query testing would miss. This is how you build evidence-based trust in AI outputs.
- **Reproducible testing**: Full configuration snapshots, checkpointing, and deterministic scoring ensure that evaluation results can be independently verified and compared across time. No black-box assessments.
- **Hallucination detection**: Extracted values are compared against context-quoted ground-truth data from published annual reports (page verification pending), with configurable tolerance thresholds. The framework quantifies exactly how often a model fabricates financial figures.
- **Factual consistency verification**: Modal-value analysis reveals bimodal failure modes (where a model alternates between two different answers) that mean-based approaches would hide. This surfaces unreliable extractions before they reach production.
- **Model comparison**: Built-in A/B evaluation supports structured model selection decisions based on quantified safety metrics rather than anecdotal testing.

---

## Table of Contents

- [What this benchmark measures (and doesn't)](#what-this-benchmark-measures-and-doesnt)
- [Quick Start](#quick-start)
- [Full Tech Stack](#full-tech-stack)
- [Project Architecture](#project-architecture)
- [Deployment & Setup](#deployment--setup)
- [Configuration Reference](#configuration-reference)
- [Usage Guide](#usage-guide)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Scoring Methodology](#scoring-methodology)
- [Report Outputs](#report-outputs)
- [Model Comparison](#model-comparison)
- [Extending the System](#extending-the-system)
- [Testing](#testing)
- [Cost Management](#cost-management)
- [Project Documents](#project-documents)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Get your first evaluation report in under 10 minutes:

```bash
# 1. Clone and enter project
cd Financialbench

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (the default provider for this project)

# 5. Preview what will happen (zero API calls)
python evaluate.py --dry-run --quick

# 6. Run a quick evaluation (~$0.01 on gpt-5-nano, ~18 API calls)
python evaluate.py --quick

# 7. Generate reports from results
python report.py --run-id <run_id_from_step_6>

# 8. Open reports/summary.txt for findings
```

---

## Full Tech Stack

### Core Language

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Primary language for all modules |

### Dependencies

| Package | Version | Purpose | Why This Choice |
|---------|---------|---------|-----------------|
| **openai** | >=1.0.0 | OpenAI Chat Completions API client (used for this measured run) | Only imported inside `src/adapters/openai_adapter.py`. Adapter pattern isolates this dependency so other providers can be swapped in without touching the engine |
| **anthropic** | >=0.39.0 | Messages API client (generic infrastructure, not exercised in this run) | Only imported inside `src/adapters/anthropic_adapter.py`. Ships so the adapter pattern is demonstrable with two concrete providers |
| **sentence-transformers** | >=2.2.0 | Semantic similarity via `all-MiniLM-L6-v2` embeddings | Independent of the model under evaluation — we never use the model being evaluated to judge its own consistency. Sentence-transformers provides fast, deterministic embeddings for pairwise cosine similarity |
| **pandas** | >=2.0.0 | DataFrame operations for report generation | Industry standard for tabular data. Used to produce per-fact detail CSVs, aggregated summaries, and to compute group-level statistics |
| **numpy** | >=1.24.0 | Numerical computation for cosine similarity matrices | Used in the scorer for N x N similarity matrix operations and upper-triangle extraction |
| **PyYAML** | >=6.0 | Configuration file parsing | Loads `config.yaml`, the main configuration file for all evaluation parameters |
| **pytest** | >=7.0.0 | Test framework | 194 tests across 8 test files. Supports fixtures, markers, and parameterisation |

### External Services

| Service | Purpose | Authentication |
|---------|---------|---------------|
| **OpenAI API** | LLM inference (GPT family — used in the measured run with `gpt-5-nano`) | `OPENAI_API_KEY` environment variable |
| **Anthropic API** | LLM inference (Claude family — optional, not used in this run) | `ANTHROPIC_API_KEY` environment variable |
| **HuggingFace Hub** | Downloads `all-MiniLM-L6-v2` model on first use (~80MB) | No authentication required (public model) |

### Data Formats

| Format | Files | Purpose |
|--------|-------|---------|
| **YAML** | `config.yaml` | All evaluation parameters |
| **JSON** | `ground_truth/facts.json`, `results/{run_id}/results.json`, `results/{run_id}/config.json`, `results/{run_id}/checkpoint.json` | Ground truth dataset, raw results storage, config snapshots, checkpoint state |
| **CSV** | `reports/per_fact_report.csv`, `reports/by_*.csv` | Tabular report outputs with comment-line headers |
| **TXT** | `reports/summary.txt` | Human-readable evaluation summary with methodology section |

### Development Tools

| Tool | Purpose |
|------|---------|
| **pytest** | Unit and integration testing |
| **Git** | Version control |
| **pip + venv** | Dependency management |

---

## Project Architecture

```
Financialbench/
├── evaluate.py                    # CLI: run evaluations
├── report.py                      # CLI: generate reports from results
├── config.yaml                    # All evaluation parameters
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git exclusions
│
├── ground_truth/
│   ├── facts.json                 # 29 context-quoted financial facts (5 companies; page verification pending)
│   └── README.md                  # Data sourcing methodology
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader + validation (frozen dataclasses)
│   ├── prompts.py                 # 4 prompt templates with variable substitution
│   ├── extractor.py               # Regex-based numeric extraction (40+ formats)
│   ├── scorer.py                  # Semantic, factual, hallucination, composite scoring
│   ├── engine.py                  # Execution engine with checkpointing
│   ├── reporter.py                # Report generation (6 output files)
│   ├── comparison.py              # A/B model comparison across metrics
│   └── adapters/
│       ├── __init__.py            # Adapter registry + factory
│       ├── base_adapter.py        # Abstract interface (LLMResponse dataclass)
│       ├── openai_adapter.py      # OpenAI adapter (sole openai SDK import; used in the measured run)
│       └── anthropic_adapter.py   # Anthropic adapter (generic infrastructure, not exercised in this run)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures (sample config, facts)
│   ├── test_config.py             # 29 tests: load, validate, quick mode, snapshot
│   ├── test_extractor.py          # 50 tests: all numeric format variations
│   ├── test_prompts.py            # 19 tests: template rendering, error handling
│   ├── test_scorer.py             # 24 tests: all scoring functions + edge cases
│   ├── test_reporter.py           # 16 tests: flag logic, CSV output, summary
│   └── test_integration.py        # End-to-end dry-run, report generation, model comparison
│
├── results/                       # Raw API responses (gitignored, created per-run)
│   └── .gitkeep
└── reports/                       # Generated reports (CSVs, summary)
    └── .gitkeep
```

### Data Flow

```
ground_truth/facts.json
        │
        ▼
  src/prompts.py ─── Generates all (fact x template) prompt combinations
        │
        ▼
  src/engine.py ──── Calls LLM N times x T temperatures per prompt
        │               │
        │               ├── src/adapters/openai_adapter.py (API calls; anthropic_adapter.py available but not exercised in this run)
        │               └── src/extractor.py (numeric extraction per response)
        │
        ▼
  results/{run_id}/results.json ── Raw data: all responses + extractions
        │
        ▼
  src/scorer.py ──── Computes 4 metrics per (fact, template, temperature) group
        │               │
        │               ├── Semantic consistency (sentence-transformers)
        │               ├── Factual consistency (modal value)
        │               ├── Hallucination rate (vs ground truth)
        │               └── Composite stability (weighted combination)
        │
        ▼
  src/reporter.py ── Generates 6 report files
        │
        ▼
  reports/
    ├── per_fact_report.csv
    ├── by_metric_type.csv
    ├── by_company.csv
    ├── by_temperature.csv
    ├── by_template.csv
    └── summary.txt
```

---

## Deployment & Setup

### Prerequisites

- **Python 3.8+** (tested on 3.8.8; 3.10+ recommended)
- **pip** (Python package manager)
- **OpenAI API key** (from [platform.openai.com](https://platform.openai.com)) — the default provider used by this project
- **Anthropic API key** (optional, only if switching `model.provider` to `"anthropic"`)
- **~500MB disk space** for sentence-transformers model download on first use
- **Internet access** for API calls and initial model download

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd Financialbench
```

#### 2. Create a Virtual Environment

```bash
# Create
python3 -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai` — OpenAI API client (default provider)
- `anthropic` — Anthropic API client (optional provider; adapter ships but is not exercised in the measured run)
- `sentence-transformers` — Embedding model for semantic scoring (downloads `all-MiniLM-L6-v2` ~80MB on first use)
- `pandas` — Report generation
- `numpy` — Numerical operations
- `PyYAML` — Config loading
- `pytest` — Testing

#### 4. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI API key (required for the default provider):

```
OPENAI_API_KEY=sk-...
# Optional, only if switching provider to anthropic:
# ANTHROPIC_API_KEY=sk-ant-api03-...
```

Alternatively, export directly:

```bash
export OPENAI_API_KEY=sk-...
```

#### 5. Verify Installation

```bash
# Run tests (no API key needed)
pytest tests/ -v

# Expected: 194 passed, 0 skipped

# Test dry-run (no API key needed)
python evaluate.py --dry-run --quick

# Expected: prints cost estimate and sample prompts, zero API calls
```

#### 6. Verify Ground Truth Data

Before running production evaluations, verify the facts in `ground_truth/facts.json` against the actual annual report PDFs. The current values are based on publicly available data and are marked as preliminary.

### Deployment Options

#### Local Development (Recommended)

Run directly on your machine. No containerisation needed — this is a CLI tool, not a web service.

```bash
python evaluate.py --quick      # Development iteration
python evaluate.py              # Full evaluation (~$0.30 on gpt-5-nano; more for frontier models)
python report.py --run-id <id>  # Generate reports
```

#### CI/CD Pipeline

A GitHub Actions workflow is included at `.github/workflows/ci.yml`. It runs on every push to `main` and on pull requests:

1. **Lint** -- checks code with `ruff`
2. **Test** -- runs `pytest tests/ -v --tb=short` on Python 3.11

Note: CI should NOT run actual evaluations (API costs). Use `--dry-run` for cost estimation in CI.

#### Server/Cloud Deployment

For scheduled evaluation runs:

```bash
# Cron job example: weekly evaluation
0 2 * * 1 cd /path/to/Financialbench && \
  source .venv/bin/activate && \
  python evaluate.py --config config.yaml 2>&1 >> /var/log/eval.log
```

Ensure `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` if you switch providers) is set in the environment where the cron job runs.

---

## Configuration Reference

All parameters live in `config.yaml`. Never hardcode values in source code.

### Model Configuration

```yaml
model:
  provider: "openai"                       # Adapter to use (see src/adapters/)
  name: "gpt-5-nano"                       # Model identifier (exact provider model id)
  max_tokens: 1024                         # Max response tokens
```

### Evaluation Parameters

```yaml
evaluation:
  temperatures: [0.0, 0.3, 0.5, 0.7, 1.0]  # Temperature sweep range
  runs_per_combination: 10                    # N repetitions per condition
  templates:                                  # Which prompt templates to use
    - "direct_extraction"
    - "contextual_extraction"
    - "comparative"
    - "qualitative"
```

### Quick Mode (Development)

```yaml
quick_mode:
  temperatures: [0.0]           # Single temperature
  runs_per_combination: 3       # Fewer runs
  max_facts: 6                  # Subset of facts
  templates:
    - "direct_extraction"       # Minimal template set
```

### Scoring Weights

```yaml
scoring:
  embedding_model: "all-MiniLM-L6-v2"   # Sentence-transformer model
  hallucination_tolerance: 0.05           # 5% relative tolerance
  composite_weights:
    semantic_consistency: 0.30            # Phrasing agreement
    factual_consistency: 0.40             # Number agreement (highest weight)
    hallucination_rate: 0.30              # Accuracy vs ground truth
```

### Flagging Thresholds

```yaml
flagging:
  green_threshold: 0.75    # >= 0.75: reliable for automated extraction
  yellow_threshold: 0.50   # >= 0.50: requires human verification
  # Below 0.50: unreliable — do not trust
```

### API Resilience

```yaml
api:
  timeout_seconds: 30          # Per-request timeout
  max_retries: 5               # Retry attempts on transient errors
  base_backoff_seconds: 1.0    # Initial backoff delay
  max_backoff_seconds: 60.0    # Maximum backoff cap
  rate_limit_rpm: 50           # Requests per minute cap
```

### Cost Controls

```yaml
cost:
  avg_input_tokens: 36          # For cost estimation (measured from this run)
  avg_output_tokens: 35
  # price_per_1k_input / price_per_1k_output are auto-resolved from MODEL_PRICING
  # in src/config.py based on model.name; set explicitly here to override.
  confirmation_threshold: 1.0   # Prompt user if estimated cost > $1
```

---

## Usage Guide

### CLI: evaluate.py

```bash
# Full evaluation (will prompt for confirmation if cost > $1)
python evaluate.py

# Quick mode: 1 temperature, 3 runs, subset of facts
python evaluate.py --quick

# Dry run: print prompts and cost, zero API calls
python evaluate.py --dry-run

# Filter to one company
python evaluate.py --company DBS

# Override run count
python evaluate.py --n-runs 5

# Resume a crashed run
python evaluate.py --resume 20260321_143022_123456

# Combine flags
python evaluate.py --quick --company DBS --dry-run
```

### CLI: report.py

```bash
# Generate CSV reports from a completed run
python report.py --run-id 20260321_143022_123456

# Output as JSON instead of CSV
python report.py --run-id <id> --format json

# Custom input/output directories
python report.py --run-id <id> --input results/ --output reports/
```

### Recommended Workflow

```
1. --dry-run --quick    →  Verify prompts look correct, check cost
2. --quick              →  Small run (~$0.01 on gpt-5-nano) to verify pipeline end-to-end
3. report.py            →  Check reports make sense
4. Full run             →  Production evaluation (~$0.30 on gpt-5-nano; higher for frontier models)
5. report.py            →  Final analysis
```

---

## Evaluation Pipeline

### What Gets Measured

For each financial fact in the ground truth dataset:

1. **Generate prompts** from 4 templates (direct, contextual, comparative, qualitative)
2. **Run each prompt N times** (default 10) at each temperature (default 5 temperatures)
3. **Extract numeric values** from each response using regex
4. **Score each group** of N responses on 4 metrics
5. **Generate reports** with per-fact detail and aggregated insights

### Scale

| Mode | Facts | Templates | Temperatures | Runs | Total Calls | Est. Cost (gpt-5-nano) |
|------|-------|-----------|-------------|------|-------------|-----------|
| `--quick` | 6 | 1 | 1 | 3 | ~18 | <$0.01 |
| Default (29 facts) | 29 | 4 | 5 | 10 | ~5,800 | ~$0.30 |
| Single company | ~7 | 4 | 5 | 10 | ~1,400 | ~$0.07 |

Cost scales with provider pricing. Switching to a frontier model (GPT-5, Claude Sonnet 4-class) raises the full-sweep estimate into the $20–$60 range for the same call volume.

### Checkpointing

The engine saves progress every 50 API calls. If the process crashes:

```bash
# Resume from where it left off (no wasted API calls)
python evaluate.py --resume <run_id>
```

Checkpoint files are stored at `results/{run_id}/checkpoint.json` using atomic writes (temp file + rename) to prevent corruption.

---

## Scoring Methodology

### 1. Semantic Consistency (weight: 0.30)

**Question:** Do repeated responses say the same thing?

- Embed all N responses using `all-MiniLM-L6-v2` (sentence-transformers)
- Compute N x N pairwise cosine similarity matrix
- Score = mean of upper triangle (excluding diagonal)
- Range: 0.0 (completely different every time) to 1.0 (identical phrasing)

> **Caveat — general-purpose encoder, not a financial-domain model.**
> `all-MiniLM-L6-v2` is a general-purpose sentence encoder trained on web-scale text. Financial language ("non-interest income," "CET1 capital ratio," "PATMI") has specific semantic structure that a domain-tuned encoder (e.g. `ProsusAI/finbert`, `yiyanghkust/finbert-tone`) may represent more faithfully. Treat the current semantic score as the MiniLM-lens view, not the definitive one.
>
> **FinBERT code path (`scripts/rescore_semantic_finbert.py`).** The scorer now supports FinBERT-family encoders via `transformers.AutoModel` + mean-pooling (`src/scorer.py::_encode_with_hf_transformer`). The historical runs in `results/*/` did not persist raw response texts, so the script cannot re-score them directly. When pointed at a new run whose `results.json` preserves `response_texts`, it writes per-fact MiniLM vs FinBERT deltas to `reports/semantic_finbert.csv`.
>
> A `--from-per-fact-csv` fallback mode runs FinBERT on a controlled paraphrase corpus built deterministically from each fact's ground-truth context plus the modal value stored in `per_fact_report.csv`. That run produced: mean MiniLM = 0.915, mean FinBERT (ProsusAI/finbert) = 0.834, mean delta = −0.082, Pearson r = −0.41, with 13 of 29 facts shifting by > 0.1 absolute. These numbers are **not** a re-score of the actual benchmark — they are a code-path sanity check on a synthetic corpus — but the direction is informative: FinBERT's mean-pooled sentence embeddings appear to compress financial-prose variation into a tighter manifold than MiniLM, lowering the pairwise-similarity range and producing per-fact scores that are not linearly related to MiniLM's. The headline MiniLM-vs-FinBERT comparison on real responses is blocked on a new evaluation run.

### 2. Factual Consistency (weight: 0.40)

**Question:** Do extracted numbers agree with each other?

- Extract numeric values from each response via regex
- Find the modal (most common) value
- Score = count_of_mode / count_of_non_null
- Range: 0.0 (all different numbers) to 1.0 (all identical)

Why modal, not mean: If a model alternates between "2.14%" and "3.41%", the mean (2.78%) represents neither answer. The mode reveals the bimodal distribution.

### 3. Hallucination Rate (weight: 0.30)

**Question:** Are extracted values correct vs ground truth?

- Compare each extracted value against the known ground truth
- Tolerance: 5% relative difference (configurable)
- Null extractions count as hallucinations
- Rate = count_wrong / total
- Range: 0.0 (all correct) to 1.0 (all wrong)

> **Caveat — the 5% tolerance is a default, not a truth claim.**
> Five percent relative tolerance on a SGD-billion figure is SGD-50M of slack, which would be catastrophic in an audit setting and negligible in a market-colour briefing. The hallucination rate is strongly sensitive to this threshold. Sweep it (`{0.5%, 1%, 2%, 5%, 10%}`) with [`rescore_at_tolerance.py`](rescore_at_tolerance.py) and report the curve, not a single number, whenever the use-case's tolerance differs from the default. See [`docs/SENSITIVITY_ANALYSIS.md`](docs/SENSITIVITY_ANALYSIS.md).

### 4. Composite Stability Score

```
composite = semantic * 0.30 + factual * 0.40 + (1 - hallucination) * 0.30
```

The hallucination rate is inverted so that higher composite always = better.

### Flagging

| Flag | Composite Score | Interpretation |
|------|----------------|----------------|
| Green | >= 0.75 | Reliable for automated extraction |
| Yellow | 0.50 - 0.74 | Inconsistent — requires human verification |
| Red | < 0.50 | Unreliable — do not trust |

---

## Report Outputs

Running `python report.py --run-id <id>` generates 6 files:

| File | Content |
|------|---------|
| `per_fact_report.csv` | One row per (fact, template, temperature) with all scores and flags |
| `by_metric_type.csv` | Mean scores aggregated by metric category (profitability, capital, etc.) |
| `by_company.csv` | Mean scores aggregated by company (DBS, OCBC, UOB, Singtel, CapitaLand) |
| `by_temperature.csv` | Mean scores aggregated by temperature (shows sensitivity curve) |
| `by_template.csv` | Mean scores aggregated by prompt template |
| `summary.txt` | Human-readable overview: methodology, traffic-light breakdown, top/bottom facts, key findings |

Every CSV file includes a `#`-prefixed comment header documenting column descriptions, flagging thresholds, scoring weights, and generation timestamp. These are self-contained — no external documentation needed to interpret them.

---

## Model Comparison

The comparison module (`src/comparison.py`) enables structured A/B evaluation between two models or two runs of the same model with different configurations.

> **WARNING — Illustrative example only, not real measurements.**
>
> The numbers in the code block below are a **schema example** showing which fields `RunSummary` accepts. They are NOT real benchmark results. Any comparison in this repo that is not sourced from a file under `results/` or `reports/` is illustrative. Run `python evaluate.py` against two providers to produce real numbers, then populate the "Headline results" table at the top of this README with the output from `reports/summary.txt`.

### Usage

```python
# Example output schema (illustrative — not real measurements)
# Populate `composite_stability`, `factual_consistency`, `hallucination_rate`,
# etc. from a completed run's `reports/summary.txt` before drawing any
# conclusion. The values below are placeholders to demonstrate the interface.

from src.comparison import RunSummary, compare_runs, format_comparison_report

summary_a = RunSummary(
    label="<model-a-label>",
    model_name="<provider/model-id-a>",
    composite_stability=0.00,          # placeholder — replace with measured value
    semantic_consistency=0.00,         # placeholder
    factual_consistency=0.00,          # placeholder
    hallucination_rate=0.00,           # placeholder
    extraction_failure_rate=0.00,      # placeholder
    n_groups=0,                        # placeholder
    n_green=0, n_yellow=0, n_red=0,    # placeholder
)

summary_b = RunSummary(
    label="<model-b-label>",
    model_name="<provider/model-id-b>",
    composite_stability=0.00,          # placeholder
    semantic_consistency=0.00,         # placeholder
    factual_consistency=0.00,          # placeholder
    hallucination_rate=0.00,           # placeholder
    extraction_failure_rate=0.00,      # placeholder
    n_groups=0,                        # placeholder
    n_green=0, n_yellow=0, n_red=0,    # placeholder
)

report = compare_runs(summary_a, summary_b)
print(format_comparison_report(report))
```

### What Gets Compared

| Metric | Direction | What It Measures |
|--------|-----------|-----------------|
| Composite stability | Higher = better | Weighted overall score |
| Factual consistency | Higher = better | Agreement of extracted values across runs |
| Semantic consistency | Higher = better | Phrasing agreement across runs |
| Hallucination rate | Lower = better | Fraction of wrong values vs ground truth |
| Extraction failure rate | Lower = better | Fraction of unparseable responses |

The comparison produces a per-metric scorecard, identifies an overall winner, and generates a recommendation with specific attention to hallucination rate as the most critical metric for financial applications.

### Statistical caveat on comparisons

Point-estimate composite scores are not directly comparable without confidence intervals. With N = 29 facts and ~10 repetitions per condition, the bootstrap 95% CI on the composite is typically several percentage points wide — on the partial `gpt-5-nano` run already in the repo, the fact-level composite is 0.594 with a 95% CI of [0.576, 0.612], a width of ~0.036 (see [`docs/STATISTICAL_RIGOR.md`](docs/STATISTICAL_RIGOR.md) and `reports/ci_summary.csv` for all five metrics). A difference of 0.05 between two models may or may not be informative depending on the overlap of their CIs. Always wrap comparison tables in `evaluate_with_ci.py` before drawing conclusions, and prefer reporting `composite = X.XX ± Y.YY (95% CI)` over bare numbers. `report.py` now auto-emits `reports/ci_summary.csv` at the end of every report run so this is a zero-effort step.

### Cost-normalised comparison (next step)

Raw composite stability ignores the price-per-token gap between a frontier model tier and a cheap nano-class model. A fair comparison should report `stability_per_dollar = composite_stability / cost_per_call_usd` alongside the raw metric. This is not yet wired into `src/comparison.py`; until it is, interpret composite differences with explicit awareness of the cost gap between the two models being compared.

---

## Extending the System

### Adding a New LLM Provider

**What ships today:** Concrete adapters for **Anthropic** (`src/adapters/anthropic_adapter.py`) and **OpenAI** (`src/adapters/openai_adapter.py`). Switching providers is a three-line config change (`model.provider` + `model.name` + matching env var). A Hugging Face / local-model adapter is not yet implemented; the base class (`BaseAdapter`) and lazy registry (`src/adapters/__init__.py`) define the interface a third adapter would need to satisfy, and a new concrete file is all that is required to add one.

The adapter pattern makes adding a third provider a single-file change. The existing `src/adapters/openai_adapter.py` is a reference implementation — mirror its interface:

1. Create `src/adapters/<provider>_adapter.py`:

```python
from src.adapters.base_adapter import BaseAdapter, LLMResponse

class MyProviderAdapter(BaseAdapter):
    def __init__(self, model_name, max_tokens, api_config):
        # Initialize SDK client
        ...

    def generate(self, prompt, temperature):
        # Call provider API, return LLMResponse
        ...

    def estimate_cost(self, n_calls):
        # Compute cost estimate
        ...

    def provider_name(self):
        return "myprovider"
```

2. Register in `src/adapters/__init__.py`:

```python
register_adapter("myprovider", MyProviderAdapter)
```

3. Update `config.yaml`:

```yaml
model:
  provider: "myprovider"
  name: "<model-id>"
```

No changes needed to engine, scorer, or reporter.

### Adding New Ground Truth Facts

Add entries to `ground_truth/facts.json` following the schema:

```json
{
  "id": "company_period_metric",
  "company": "CompanyName",
  "metric": "Human Readable Metric Name",
  "metric_abbreviation": "ABBR",
  "period": "FY2024",
  "value": 12.34,
  "unit": "percent",
  "currency": null,
  "source": "Company FY2024 Annual Report",
  "page": 42,
  "context": "Verbatim quote from the report containing the figure.",
  "difficulty": "easy",
  "category": "profitability"
}
```

Valid `unit` values: `percent`, `sgd_millions`, `sgd_billions`, `sgd`, `ratio`, `millions`, `count`.

### Adding New Prompt Templates

Add to the `TEMPLATES` dict in `src/prompts.py`:

```python
TEMPLATES["new_template"] = PromptTemplate(
    name="new_template",
    template="Your template with {company} and {metric} placeholders.",
    description="What this template tests",
    required_fields=["company", "metric", "period"],
)
```

Then add `"new_template"` to `config.yaml`'s `evaluation.templates` list.

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_extractor.py -v

# Specific test class
pytest tests/test_extractor.py::TestNegativeValues -v

# With short tracebacks
pytest tests/ --tb=short
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|---------------|
| `test_extractor.py` | 51 | All numeric formats: percentages, currencies, scales, negatives, ratios, cents, edge cases |
| `test_scorer.py` | 33 | Factual consistency (modal value), hallucination rate (tolerance, None handling), composite, semantic (skipped without sentence-transformers) |
| `test_config.py` | 29 | Config loading, quick mode overrides, validation (weights, temperatures, thresholds), snapshot |
| `test_prompts.py` | 22 | Template rendering, missing fields, unknown templates, all-prompts generation |
| `test_engine.py` | 19 | Engine orchestration, adapter selection, retries, checkpoint resume |
| `test_adapters.py` | 17 | Provider adapters (OpenAI, Anthropic), request shaping, error handling |
| `test_reporter.py` | 16 | Flag logic, CSV generation, summary content, JSON format |
| `test_integration.py` | 7 | End-to-end pipeline smoke tests |

All 194 tests pass when `sentence-transformers` is installed (included in `requirements.txt`). The semantic consistency tests in `test_scorer.py::TestSemanticConsistency` require that package and will be skipped if it is absent.

---

## Cost Management

### Estimation Formula

```
total_calls = n_facts x n_templates x n_temperatures x n_runs
cost_per_call = (avg_input_tokens / 1000 x price_input) + (avg_output_tokens / 1000 x price_output)
total_cost = total_calls x cost_per_call
```

With defaults (`gpt-5-nano`, avg 36 input / 35 output tokens):

```
cost_per_call = (36/1000 x $0.0002) + (35/1000 x $0.00125) ≈ $0.000051
```

### Safety Rails

1. **`--dry-run`** — Always preview cost before spending
2. **Confirmation gate** — Runs exceeding `$1.00` (configurable) require explicit `y` confirmation
3. **`--quick` mode** — Development iteration at <$0.01 per run
4. **Checkpointing** — Crash at call 2500? Resume without re-running the first 2500

### Cost by Scenario

| Scenario | Calls | Est. Cost (gpt-5-nano) |
|----------|-------|-----------|
| Quick mode (default) | ~18 | <$0.01 |
| Single company, full sweep | ~1,400 | ~$0.07 |
| Full suite (29 facts) | ~5,800 | ~$0.30 |
| Quick + single company | ≤18 | <$0.01 |

Swapping in a frontier model (e.g. Claude Sonnet 4-class, GPT-5) scales full-suite cost toward the $20–$60 range — always run `--dry-run` first after a provider/model change.

---

## Project Documents

### In This Repository

| Document | Location | Purpose |
|----------|----------|---------|
| **README.md** | `/README.md` | This file — deployment, tech stack, usage guide |
| **Ground Truth README** | `/ground_truth/README.md` | How financial facts were sourced, verification methodology, data schema documentation |
| **Config File** | `/config.yaml` | Documented with inline comments — all evaluation parameters |
| **Environment Template** | `/.env.example` | Shows required environment variables |

### External Data Sources

The ground truth dataset was sourced from publicly available annual reports:

| Company | Report | URL |
|---------|--------|-----|
| **DBS Group** | FY2024 Annual Report | dbs.com/investor |
| **OCBC Bank** | FY2024 Annual Report | ocbc.com/group/investors |
| **UOB** | FY2024 Annual Report | uobgroup.com/investor-relations |
| **Singtel** | FY2024 Annual Report | singtel.com/about-us/investor-relations |
| **CapitaLand Investment** | FY2024 Annual Report | capitaland.com/investor-relations |

Each fact in `facts.json` includes `source` (report title), `page` (page number), and `context` (verbatim quote) for traceability.

---

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: openai` (or `anthropic`) | SDK not installed | `pip install -r requirements.txt` |
| `OPENAI_API_KEY not set` | Missing env var for default provider | `export OPENAI_API_KEY=sk-...` or add to `.env` |
| `ANTHROPIC_API_KEY not set` | Missing env var when `model.provider: "anthropic"` | `export ANTHROPIC_API_KEY=sk-ant-...` or add to `.env` |
| `facts.json must contain a JSON array` | Old format | The loader supports both raw arrays and `{"facts": [...]}` wrapper objects |
| 4 tests skipped | sentence-transformers not installed | Install with `pip install sentence-transformers` (downloads ~80MB model on first use) |
| Rate limit errors | Too many requests | Reduce `api.rate_limit_rpm` in config.yaml, or increase `api.base_backoff_seconds` |
| Resume finds no checkpoint | Wrong run_id | Check `results/` directory for available run IDs |
| High extraction failure rate | LLM responses not parseable | Review prompt templates — ensure they ask for "just the number and unit" |

### Getting Help

- Check the test suite for usage examples: `tests/test_*.py`
- Read `config.yaml` comments for parameter documentation
- Open an issue on the repository for bugs
