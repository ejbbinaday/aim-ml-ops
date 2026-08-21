# aim-ml-ops — Revenue Forecasting MLOps System

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

**Final Project scope** turns that working baseline into a deployable and
monitorable service: a strict FastAPI serving contract, environment-driven
MLflow Model Registry loading, a Docker/Compose stack, API tests, and an
Evidently input-and-prediction drift report with documented findings.

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.11)
- Docker Desktop (for the complete local serving stack and the Airflow UI)

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

## Forecast API (Final Project)

The API loads one compatible model version from the MLflow Registry during
startup and reuses it for every request. The exact JSON fields, validation
rules, response semantics, and environment variables are documented in
[`docs/SERVING_CONTRACT.md`](docs/SERVING_CONTRACT.md).

### Run locally

```bash
# Ensure the final-project model contract is registered (creates Version 2+
# when the registry contains only the Milestone 2 Version 1 contract).
uv run python models/train.py

# Start the API against the repo-local SQLite registry.
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"horizon_months":12,"interval_level":80}'
```

Interactive OpenAPI documentation is available at <http://localhost:8000/docs>.

### Run the containerized stack

Open Docker Desktop, then run:

```bash
docker compose up --build mlflow trainer api
```

Compose starts MLflow at <http://localhost:5000>, reuses or trains one
compatible registered model, then starts the API at <http://localhost:8000>.
Configuration defaults are shown in `.env.example`; copy it to the gitignored
`.env` file only when an override is needed. Stop the stack with
`docker compose down` (the local MLflow state remains under `mlflow/`).

If macOS AirPlay Receiver is using host port `5000`, start the same stack with
`MLFLOW_PORT=5001 docker compose up --build mlflow trainer api` and open MLflow
at <http://localhost:5001>. Container-to-container tracking remains on port
`5000`, so no other configuration changes are required.

## Drift monitoring (Final Project)

On a fresh clone, train and register the model first:

```bash
uv run python models/train.py
```

Then generate the monitoring report:

```bash
uv run python monitoring/run_report.py
```

This command uses the configured registered model to create deterministic,
PII-free reference/current serving datasets and writes:

- `reports/evidently_report.html` — visual Evidently report;
- `reports/evidently_report.json` — machine-readable metric results;
- `reports/findings.md` — interpretation, response, and limitations;
- `reports/monitoring_summary.json` and the two source CSV files — provenance.

The current example intentionally shifts requests from shorter to longer
forecast horizons so both API-input and cumulative-prediction drift are
visible. It is a course demonstration, not claimed production behavior or
evidence of model inaccuracy.

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

The tracking URI is taken from the `MLFLOW_TRACKING_URI` environment variable
when set (Option B below). When unset, `models/train.py` intentionally falls
back to a local SQLite store at `mlflow/mlflow.db` (Option A), so a fresh
clone can train and inspect runs with zero setup.

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

> **macOS note:** AirPlay Receiver can occupy port `5000`. If the UI fails to
> start, use port `5001` instead, e.g.
> `uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5001`
> (and `MLFLOW_TRACKING_URI=http://localhost:5001` for Option B).

## Test suite (Milestone 2)

```bash
uv run pytest                                  # full suite
uv run pytest --cov --cov-report=term-missing  # coverage with 60% gate
```

The suite covers the three spec-mandated categories:

| File | Category | What it proves |
|------|----------|----------------|
| `tests/test_data_validation.py` | Data validation | The Pandera gate rejects missing columns, wrong types, and out-of-range dates — and passes valid data |
| `tests/test_model_quality.py` | Model quality | Held-out 12-month MASE beats the seasonal-naïve threshold (`MASE_THRESHOLD = 1.0`, derived from Milestone 1 CV results); forecasts have the right shape, no NaNs, non-negative values, ordered intervals |
| `tests/test_pipeline_integration.py` | Integration | extract → validate → load end-to-end on synthetic data produces the versioned artifact; `models/train.py` end-to-end logs params/metrics and registers the model against a temp MLflow store |
| `tests/test_storage.py` | (support) | Round-trips for the artifact I/O layer every test above depends on |
| `tests/test_api.py` | Serving contract | Health, valid 80/95% forecasts, invalid 422 payloads, startup failure, incompatible model output |
| `tests/test_monitoring.py` | Monitoring | Deterministic PII-free serving samples and a known PSI shift above the documented threshold |

Coverage is scoped to serving + training + monitoring + pipeline code
(`app/`, `models/`, `monitoring/`, `src/`); thin
entry-point wrappers (`scripts/`, `dags/`, `dashboard.py`), the Streamlit UI,
and the live-Stripe ingest (needs an API key) are excluded — see
`[tool.coverage.run]` in `pyproject.toml`.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to
`main` and every pull request, on GitHub-hosted `ubuntu-latest` runners:
install `uv` → `uv sync` → `ruff check .` (hard fail) → `pytest --cov`
(hard fail, including the 60% coverage gate; report visible in the job output). Recent runs:
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
├── app/                              # FastAPI contract + MLflow model service
├── dags/revenue_pipeline.py         # Airflow DAG: extract → validate → load
├── data/historical_unified_payments.csv   # synthetic sample (no PII)
├── mlflow/                          # gitignored MLflow store (SQLite + artifacts)
├── models/train.py                  # MLflow-tracked training entry point (M2)
├── monitoring/run_report.py         # Evidently input/prediction drift report
├── reports/                          # PII-free monitoring data, report, findings
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
│   ├── test_api.py                  # Final Project serving-contract tests
│   ├── test_model_quality.py        # Deliverable 2b
│   ├── test_pipeline_integration.py # Deliverable 2c
│   ├── test_monitoring.py           # Drift-data reproducibility tests
│   └── test_storage.py              # artifact I/O round-trips
├── dashboard.py                     # standalone Streamlit entrypoint
├── docker-compose.yaml              # Airflow UI + MLflow tracking server
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```
