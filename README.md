# llm-eval-platform

An enterprise-grade evaluation and regression platform for LLM and GenAI
applications. Define datasets, run a metric suite, compare runs against a
baseline, and enforce quality gates in CI.

## Why this exists

Shipping an LLM feature without evaluation is shipping blind. Teams need a
repeatable way to answer three questions:

1. Did quality regress since the last release?
2. Which examples got worse, and by how much?
3. Should this build be blocked?

This platform answers all three with deterministic metrics, persisted runs, and
machine-readable gate results.

## Features

- **Dataset registry** — load evaluation datasets from JSON/JSONL/CSV, validate
  schemas, and summarize them before running.
- **Metric library** — exact match, contains, BLEU, ROUGE, faithfulness, answer
  relevance, plus a registry for custom metrics.
- **LLM-as-judge** — rubric-based scoring with a pluggable judge interface.
- **Run comparison** — diff two runs per example and classify each change as
  improvement, regression, unchanged, or broken.
- **Quality gates** — threshold-based pass/fail decisions designed for CI.
- **SQLite persistence** — every run is stored and queryable; no external
  service required.
- **REST API** — datasets, metrics, runs, comparisons, and gate evaluation.
- **CLI** — the same capabilities from a terminal or a pipeline step.

## Architecture

```
Dataset file ──► Loader ──► Runner ──► Metrics ──► SQLite run store
                                  │                     │
                                  └──► Judge            ├──► Compare (baseline ↔ candidate)
                                                        ├──► Quality gate
                                                        └──► API / CLI
```

## Project layout

```
llm-eval-platform/
├── src/llm_eval/
│   ├── api/app.py              # FastAPI application
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Settings
│   ├── datasets/               # Registry + loaders (JSON/JSONL/CSV)
│   ├── metrics/                # Metric implementations + registry
│   ├── judges/                 # LLM judge + rubric scoring
│   ├── runners/                # Runner + run comparison
│   ├── gates/quality_gate.py   # CI quality gate
│   └── store/sqlite_store.py   # Persistence
├── tests/                      # pytest suite
└── examples/                   # Runnable example
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

### CLI

```bash
# Evaluate a dataset with the default metric suite
llm-eval run examples/sample.jsonl --dataset sample

# List registered metrics and datasets
llm-eval metrics
llm-eval datasets

# Compare a candidate run against a baseline
llm-eval compare <baseline_run_id> <candidate_run_id>

# Enforce thresholds (exits non-zero on failure)
llm-eval gate <run_id> --min faithfulness=0.8 --min relevance=0.7
```

### API

```bash
uvicorn llm_eval.api.app:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/datasets` | List dataset summaries |
| `POST` | `/datasets` | Register a dataset |
| `GET` | `/metrics` | List available metrics |
| `POST` | `/runs` | Execute an evaluation run |
| `GET` | `/runs` | List stored runs |
| `GET` | `/runs/{run_id}` | Fetch a single run |
| `POST` | `/compare` | Compare two runs |
| `POST` | `/gates/evaluate` | Evaluate a run against thresholds |

## Metrics

| Metric | What it measures |
| --- | --- |
| `exact_match` | Normalized string equality |
| `contains` | Reference answer present in the prediction |
| `bleu` | n-gram precision against the reference |
| `rouge` | n-gram recall overlap |
| `faithfulness` | Prediction supported by provided context |
| `relevance` | Prediction addresses the question |
| `llm_judge` | Rubric-based model grading |

Metrics implement a common interface, so custom metrics can be registered and
used by both the runner and the API without changes elsewhere.

## Quality gates in CI

Gate results are machine-readable and return a non-zero exit code on failure:

```yaml
- name: Evaluate
  run: |
    pip install -e ".[dev]"
    llm-eval run tests/fixtures/golden.jsonl --dataset golden
    llm-eval gate "$(llm-eval runs --latest --json | jq -r .id)" \
      --min faithfulness=0.85 --min relevance=0.8
```

## Testing

```bash
pytest
pytest --cov=llm_eval --cov-report=term-missing
```

The suite covers loaders, metric correctness, judge behavior, run comparison
classification, gates, the SQLite store, the CLI, and the HTTP API.

## Design principles

- **Deterministic first.** Core metrics are reproducible and require no model
  calls; judges are opt-in.
- **Runs are immutable.** Comparison always reads persisted runs, never
  in-memory state.
- **Fail closed in CI.** Gate errors are treated as failures, not passes.
- **No hidden services.** SQLite keeps local development and CI self-contained.

## Limitations

- Judge quality depends on the configured model and rubric.
- BLEU/ROUGE are lexical metrics and correlate imperfectly with human judgment.
- SQLite is intended for single-node use.

## License

MIT
