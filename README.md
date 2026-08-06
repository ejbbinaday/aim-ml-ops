# aim-ml-ops — Revenue Forecasting Pipeline (Milestones 1 & 2)

[![CI](https://github.com/ejbbinaday/aim-ml-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/ejbbinaday/aim-ml-ops/actions/workflows/ci.yml)

## Project & model

This repository operationalizes an existing **revenue-forecasting model** for an
Australian medical clinic and digital-health platform (preventative medicine,
longevity, peak-performance). The model is a monthly, multi-series time-series
forecaster: raw Stripe charge history is split into three revenue streams —
**Core MRR** (recurring subscriptions), **IV League** (Calendly bookings), and
**Core one-time** (packages, corporate/Keyman, invoices, consults) — and each
stream is forecast independently (EAT ensemble / Level / Croston, auto-selected
by backtested MASE) then summed into a bottom-up total with prediction
intervals and Bear/Base/Bull scenarios. The output feeds leadership budgeting,
hiring, and growth-target decisions via a Streamlit revenue page; it is
decision-support, so a wrong forecast mis-sets expectations rather than causing
direct customer harm.

**Milestone 1 scope** is the *data pipeline* — the `extract → validate → load`
layer that turns raw payment records into the clean, model-ready monthly series
every downstream forecasting phase depends on. See
[`docs/ML_PROBLEM_FRAMING.md`](docs/ML_PROBLEM_FRAMING.md) for the full ML
problem framing document (Deliverable 1).

**Milestone 2 scope** adds the MLOps layers on top of that pipeline:
MLflow experiment tracking + Model Registry (`models/train.py`), an automated
pytest suite covering data validation / model quality / integration
(`tests/`), and a GitHub Actions CI workflow that lints and tests every push
and pull request (`.github/workflows/ci.yml`).

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.11)
- Docker (only if you want to run the Airflow UI)

## Quickstart

```bash
# 1. Install dependencies into a local .venv (exact versions from uv.lock)
uv sync

# 2. Generate the synthetic, PII-free sample data (already committed, but
#    reproducible byte-for-byte)
uv run python scripts/generate_synthetic_data.py

# 3. Run the Milestone 1 data pipeline (extract → validate → load)
uv run python scripts/run_pipeline.py
```

### Run via Airflow (`docker compose up`)

The same three tasks run as an Airflow DAG (`dags/revenue_pipeline.py`):

```bash
docker compose up
```

Open <http://localhost:8080> (user `admin`; password is printed in the logs and
written to `standalone_admin_password.txt`), unpause **`revenue_pipeline`**, and
trigger it. Clean output lands in `./outputs/tables/`.

## MLflow experiment tracking (Milestone 2)

Training/evaluation runs are tracked with **MLflow 3.x** under the
`revenue-forecast` experiment: pipeline configuration as params, per-series
cross-validation metrics (MASE, RMSE, Winkler-80) as metrics, and the fitted
model bundle registered to the **Model Registry** as `revenue-forecast-bundle`
with a `milestone` version tag.

The tracking URI always comes from the `MLFLOW_TRACKING_URI` environment
variable — it is never hardcoded.

### Option A — serverless (SQLite, zero setup)

```bash
# Run the tracked training pipeline (phases 1–4 + registry)
uv run python models/train.py

# Inspect runs in the MLflow UI backed by the same local SQLite store
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

Open <http://localhost:5000> — the `revenue-forecast` experiment and the
registered `revenue-forecast-bundle` model appear after one training run.

### Option B — tracking server via docker compose

```bash
docker compose up mlflow                    # server + UI at http://localhost:5000
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python models/train.py
```

Both stores live under the gitignored `mlflow/` directory.

## Test suite (Milestone 2)

```bash
uv run pytest                                              # full suite
uv run pytest --cov=src --cov=models --cov-report=term-missing  # with coverage
```

The suite covers the three spec-mandated categories:

| File | Category | What it proves |
|------|----------|----------------|
| `tests/test_data_validation.py` | Data validation | The Pandera gate rejects missing columns, wrong types, and out-of-range dates — and passes valid data |
| `tests/test_model_quality.py` | Model quality | Held-out 12-month MASE beats the seasonal-naïve threshold (`MASE_THRESHOLD = 1.0`, derived from Milestone 1 CV results); forecasts have the right shape, no NaNs, non-negative values, ordered intervals |
| `tests/test_pipeline_integration.py` | Integration | extract → validate → load end-to-end on synthetic data produces the versioned artifact; `models/train.py` end-to-end logs params/metrics and registers the model against a temp MLflow store |
| `tests/test_storage.py` | (support) | Round-trips for the artifact I/O layer every test above depends on |

Coverage is scoped to pipeline + training code (`src/`, `models/`); thin
entry-point wrappers (`scripts/`, `dags/`, `dashboard.py`), the Streamlit UI,
and the live-Stripe ingest (needs an API key) are excluded — see
`[tool.coverage.run]` in `pyproject.toml`.

## Continuous integration (Milestone 2)

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and every pull request, on GitHub-hosted `ubuntu-latest` runners:
install `uv` → `uv sync` → `ruff check .` (hard fail) → `pytest --cov`
(hard fail; coverage report visible in the job output). Recent runs:
[Actions tab](https://github.com/ejbbinaday/aim-ml-ops/actions/workflows/ci.yml).

### View the Streamlit revenue page

```bash
# Run the full forecast pipeline first so the page has forecast tables to read
uv run python scripts/run_revenue_pipeline.py
uv run streamlit run dashboard.py
```

## The pipeline (Extract → Validate → Load)

| Task | What it does | Artifact |
|------|--------------|----------|
| **extract** | Read raw payments (Stripe snapshot if configured, else the synthetic CSV under `data/`) | `outputs/tables/00_raw_charges.csv` |
| **validate** | Enforce the raw-charges schema with **Pandera** — the pipeline **fails** on invalid data | *(gate — no artifact)* |
| **load** | Clean (Paid-only, AUD-only), categorize into the 3 streams, aggregate to month-start buckets, engineer features (COVID flag, log1p, Total) | `outputs/tables/01_monthly_series_<run_id>.csv` |

The task logic lives in `src/revenue/pipeline_m1.py` (unit-testable, Airflow-free);
the DAG and `scripts/run_pipeline.py` are thin wrappers over it.

## Output artifact

- **Versioned clean dataset:** `outputs/tables/01_monthly_series_<run_id>.csv`,
  where `run_id` is a UTC timestamp (e.g. `01_monthly_series_20260716T101500Z.csv`).
  Wide-format monthly DataFrame indexed by month-start: the three revenue
  streams, their `log1p` variants, an `is_covid_startup` flag, and `Total`.
- A stable **`outputs/tables/01_monthly_series.csv`** copy (what the forecasting
  phases and the Streamlit page read).
- `outputs/tables/01_category_summary.csv` — per-category transaction audit.
- `outputs/tables/01_run_manifest_<run_id>.json` — run provenance
  (row counts, month range, artifact paths).

Artifacts are written through `src/storage.py`, a single I/O switch: local
`outputs/` by default, or `s3://$OUTPUTS_BUCKET/...` when `OUTPUTS_BUCKET` is set.

## Data

`data/historical_unified_payments.csv` is **synthetic and PII-free** — a
structurally faithful stand-in for the confidential Stripe export, generated
deterministically by `scripts/generate_synthetic_data.py`. No real customer data
is ever committed.

## Development

```bash
uv run ruff check .              # lint — returns zero violations
uv run pytest                    # tests
uv run pre-commit install        # enable pre-commit hooks
uv run pre-commit run --all-files
```

## Layout

```
aim-ml-ops/
├── .github/workflows/ci.yml         # CI: lint + tests + coverage (M2)
├── dags/revenue_pipeline.py         # Airflow DAG: extract → validate → load
├── data/historical_unified_payments.csv   # synthetic sample (no PII)
├── mlflow/                          # gitignored MLflow store (SQLite + artifacts)
├── models/train.py                  # MLflow-tracked training entry point (M2)
├── scripts/
│   ├── generate_synthetic_data.py   # reproduce the sample data
│   ├── run_pipeline.py              # run M1 pipeline without Airflow (uv run)
│   └── run_revenue_pipeline.py      # full forecast pipeline (phases 0–5)
├── src/
│   ├── storage.py                   # single artifact-I/O switch
│   ├── auth.py                      # no-op page decorator (standalone shim)
│   ├── pages/revenue.py             # Streamlit revenue page
│   └── revenue/                     # pipeline + validation + forecasting phases
├── tests/
│   ├── conftest.py                  # shared fixtures (isolated storage, sample frame)
│   ├── test_data_validation.py      # Deliverable 2a
│   ├── test_model_quality.py        # Deliverable 2b
│   ├── test_pipeline_integration.py # Deliverable 2c
│   └── test_storage.py              # artifact I/O round-trips
├── dashboard.py                     # standalone Streamlit entrypoint
├── docker-compose.yaml              # Airflow UI + MLflow tracking server
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```
