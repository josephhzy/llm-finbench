# LLM Financial Stability Bench

A research-grade evaluation harness that measures LLM consistency and hallucination rates on financial data extraction tasks, focused on SGX-listed companies (DBS, OCBC, UOB, Singtel, CapitaLand).

This is a measurement tool, not a demo app. Precision, reproducibility, and statistical rigour are the priorities.

Live Dashboard: https://llm-finbench.streamlit.app/

---

## Table of Contents

- [Quick Start](#quick-start)
- [Full Tech Stack](#full-tech-stack)
- [Project Architecture](#project-architecture)
- [Deployment & Setup](#deployment--setup)
- [Configuration Reference](#configuration-reference)
- [Usage Guide](#usage-guide)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Scoring Methodology](#scoring-methodology)
- [Report Outputs](#report-outputs)
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
# Edit .env and add your Anthropic API key

# 5. Preview what will happen (zero API calls)
python evaluate.py --dry-run --quick

# 6. Run a quick evaluation (~$0.09, ~18 API calls)
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
| **anthropic** | >=0.39.0 | Anthropic Messages API client for Claude models | Only imported inside `src/adapters/anthropic_adapter.py`. Adapter pattern isolates this dependency so other providers can be added without touching the engine |
| **sentence-transformers** | >=2.2.0 | Semantic similarity via `all-MiniLM-L6-v2` embeddings | Independent of Claude — we never use the model being evaluated to judge its own consistency. Sentence-transformers provides fast, deterministic embeddings for pairwise cosine similarity |
| **pandas** | >=2.0.0 | DataFrame operations for report generation | Industry standard for tabular data. Used to produce per-fact detail CSVs, aggregated summaries, and to compute group-level statistics |
| **numpy** | >=1.24.0 | Numerical computation for cosine similarity matrices | Used in the scorer for N x N similarity matrix operations and upper-triangle extraction |
| **PyYAML** | >=6.0 | Configuration file parsing | Loads `config.yaml`, the the main configuration file for all evaluation parameters |
| **pytest** | >=7.0.0 | Test framework | 148 tests across 6 test files. Supports fixtures, markers, and parameterisation |

### External Services

| Service | Purpose | Authentication |
|---------|---------|---------------|
| **Anthropic API** | LLM inference (Claude models) | `ANTHROPIC_API_KEY` environment variable |
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
│   ├── facts.json                 # 33 verified financial facts (5 companies)
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
│   └── adapters/
│       ├── __init__.py            # Adapter registry + factory
│       ├── base_adapter.py        # Abstract interface (LLMResponse dataclass)
│       └── anthropic_adapter.py   # Claude adapter (sole anthropic SDK import)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures (sample config, facts)
│   ├── test_config.py             # 29 tests: load, validate, quick mode, snapshot
│   ├── test_extractor.py          # 50 tests: all numeric format variations
│   ├── test_prompts.py            # 19 tests: template rendering, error handling
│   ├── test_scorer.py             # 24 tests: all scoring functions + edge cases
│   └── test_reporter.py           # 16 tests: flag logic, CSV output, summary
│
├── results/                       # Raw API responses (gitignored, created per-run)
│   └── .gitkeep
├── reports/                       # Generated reports (CSVs, summary)
│   └── .gitkeep
└── tasks/
    ├── todo.md                    # Build progress checklist
    └── lessons.md                 # Patterns and corrections log
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
        │               ├── src/adapters/anthropic_adapter.py (API calls)
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
- **Anthropic API key** (from [console.anthropic.com](https://console.anthropic.com))
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
- `anthropic` — Claude API client
- `sentence-transformers` — Embedding model for semantic scoring (downloads `all-MiniLM-L6-v2` ~80MB on first use)
- `pandas` — Report generation
- `numpy` — Numerical operations
- `PyYAML` — Config loading
- `pytest` — Testing

#### 4. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and set your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Alternatively, export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

#### 5. Verify Installation

```bash
# Run tests (no API key needed)
pytest tests/ -v

# Expected: 148 passed, 0 skipped

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
python evaluate.py              # Full evaluation (~$34)
python report.py --run-id <id>  # Generate reports
```

#### CI/CD Pipeline

Add to your CI to run the test suite:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ -v
```

Note: CI should NOT run actual evaluations (API costs). Use `--dry-run` for cost estimation in CI.

#### Server/Cloud Deployment

For scheduled evaluation runs:

```bash
# Cron job example: weekly evaluation
0 2 * * 1 cd /path/to/Financialbench && \
  source .venv/bin/activate && \
  python evaluate.py --config config.yaml 2>&1 >> /var/log/eval.log
```

Ensure `ANTHROPIC_API_KEY` is set in the environment where the cron job runs.

---

## Configuration Reference

All parameters live in `config.yaml`. Never hardcode values in source code.

### Model Configuration

```yaml
model:
  provider: "anthropic"                    # Adapter to use (see src/adapters/)
  name: "claude-sonnet-4-20250514"   # Model identifier
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
  avg_input_tokens: 200         # For cost estimation
  avg_output_tokens: 300
  price_per_1k_input: 0.003     # USD per 1K input tokens
  price_per_1k_output: 0.015    # USD per 1K output tokens
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
2. --quick              →  Small run (~$0.09) to verify pipeline end-to-end
3. report.py            →  Check reports make sense
4. Full run             →  Production evaluation (~$34)
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

| Mode | Facts | Templates | Temperatures | Runs | Total Calls | Est. Cost |
|------|-------|-----------|-------------|------|-------------|-----------|
| `--quick` | 6 | 1 | 1 | 3 | ~18 | ~$0.09 |
| Default (33 facts) | 33 | 4 | 5 | 10 | ~6,600 | ~$34 |
| Single company | ~7 | 4 | 5 | 10 | ~1,400 | ~$7 |

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

## Extending the System

### Adding a New LLM Provider

The adapter pattern makes this a single-file change:

1. Create `src/adapters/openai_adapter.py`:

```python
from src.adapters.base_adapter import BaseAdapter, LLMResponse

class OpenAIAdapter(BaseAdapter):
    def __init__(self, model_name, max_tokens, api_config):
        # Initialize OpenAI client
        ...

    def generate(self, prompt, temperature):
        # Call OpenAI API, return LLMResponse
        ...

    def estimate_cost(self, n_calls):
        # Compute cost estimate
        ...

    def provider_name(self):
        return "openai"
```

2. Register in `src/adapters/__init__.py`:

```python
register_adapter("openai", OpenAIAdapter)
```

3. Update `config.yaml`:

```yaml
model:
  provider: "openai"
  name: "gpt-4o"
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
| `test_extractor.py` | 50 | All numeric formats: percentages, currencies, scales, negatives, ratios, cents, edge cases |
| `test_config.py` | 29 | Config loading, quick mode overrides, validation (weights, temperatures, thresholds), snapshot |
| `test_scorer.py` | 20+4 | Factual consistency (modal value), hallucination rate (tolerance, None handling), composite, semantic (skipped without sentence-transformers) |
| `test_prompts.py` | 19 | Template rendering, missing fields, unknown templates, all-prompts generation |
| `test_reporter.py` | 16 | Flag logic, CSV generation, summary content, JSON format |

All 148 tests pass when `sentence-transformers` is installed (included in `requirements.txt`). The semantic consistency tests in `test_scorer.py::TestSemanticConsistency` require that package and will be skipped if it is absent.

---

## Cost Management

### Estimation Formula

```
total_calls = n_facts x n_templates x n_temperatures x n_runs
cost_per_call = (avg_input_tokens / 1000 x price_input) + (avg_output_tokens / 1000 x price_output)
total_cost = total_calls x cost_per_call
```

With defaults (Claude Sonnet):

```
cost_per_call = (200/1000 x $0.003) + (300/1000 x $0.015) = $0.0051
```

### Safety Rails

1. **`--dry-run`** — Always preview cost before spending
2. **Confirmation gate** — Runs exceeding `$1.00` (configurable) require explicit `y` confirmation
3. **`--quick` mode** — Development iteration at ~$0.10-0.50 per run
4. **Checkpointing** — Crash at call 2500? Resume without re-running the first 2500

### Cost by Scenario

| Scenario | Calls | Est. Cost |
|----------|-------|-----------|
| Quick mode (default) | ~18 | ~$0.09 |
| Single company, full sweep | ~1,400 | ~$7 |
| Full suite (33 facts) | ~6,600 | ~$34 |
| Quick + single company | ≤18 | ≤$0.09 |

---

## Project Documents

### In This Repository

| Document | Location | Purpose |
|----------|----------|---------|
| **README.md** | `/README.md` | This file — deployment, tech stack, usage guide |
| **Ground Truth README** | `/ground_truth/README.md` | How financial facts were sourced, verification methodology, data schema documentation |
| **Lessons Learned** | `/tasks/lessons.md` | Post-correction patterns to prevent repeated mistakes |
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
| `ModuleNotFoundError: anthropic` | SDK not installed | `pip install -r requirements.txt` |
| `ANTHROPIC_API_KEY not set` | Missing env var | `export ANTHROPIC_API_KEY=sk-ant-...` or add to `.env` |
| `facts.json must contain a JSON array` | Old format | The loader supports both raw arrays and `{"facts": [...]}` wrapper objects |
| 4 tests skipped | sentence-transformers not installed | Install with `pip install sentence-transformers` (downloads ~80MB model on first use) |
| Rate limit errors | Too many requests | Reduce `api.rate_limit_rpm` in config.yaml, or increase `api.base_backoff_seconds` |
| Resume finds no checkpoint | Wrong run_id | Check `results/` directory for available run IDs |
| High extraction failure rate | LLM responses not parseable | Review prompt templates — ensure they ask for "just the number and unit" |

### Getting Help

- Check the test suite for usage examples: `tests/test_*.py`
- Read `config.yaml` comments for parameter documentation
- Open an issue on the repository for bugs
