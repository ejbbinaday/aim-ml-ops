# Milestone 1: ML Problem Framing Document
### Revenue Forecasting Pipeline for an Australian Health & Longevity Platform
**MLOps CPT3**

---

## The ML System

This model is a monthly, multi-series time-series forecasting pipeline that projects the recurring revenue of an Australian medical clinic and digital health platform focused on preventative medicine, longevity, and peak-performance optimization (hereafter *the clinic*). It is not one model but a suite: the raw Stripe transaction history is split into three revenue streams, and each stream gets its own model, chosen automatically by backtested accuracy.

| Series | What it captures | Candidate Models |
|---|---|---|
| **Core_MRR** | True recurring subscription revenue (Subscription creation / Subscription update charges) | EAT ensemble (AutoARIMA + AutoETS + Theta) vs. Level |
| **IV_League** | Calendly "IV League" bookings | EAT ensemble vs. Croston (sparse/intermittent demand) |
| **Core_OneTime** | Everything else — packages, corporate/Keyman, invoices, consultations | Level (mean-reverting flat) vs. EAT vs. Croston |

**What it predicts:** for each series, a 12-month and 24-month forward forecast in AUD, with 80% and 95% prediction intervals, plus a bottom-up Total Revenue projection (the sum of the three series) and Bear / Base / Bull scenarios.

**Feature set:** this is a univariate-per-series problem, so "features" are engineered mostly from each series' own history:

- Month-start aggregated revenue per category (log1p-transformed variants available).
- A COVID / startup structural-break flag (Aug 2020 – Oct 2021).
- Outlier intervention dummies — Hampel detection on the STL remainder (Phase 2).
- Structural-break step dummies — CUSUM/PELT changepoint detection (Phase 2).
- Model-internal terms: seasonal period = 12, AICc-selected ARIMA/ETS orders.

There are deliberately no cross-series or external regressors beyond the break/outlier dummies — the three series are forecast independently and then summed.

---

## System Architecture

The full forecasting system is a linear chain of six idempotent phases that communicate only through persisted artifacts, guarded by two safety gates (a *no-op gate* that short-circuits an unchanged input, and a *regression gate* that blocks a MASE regression before any write), and closed by a provenance manifest whose hashes seed the next run's gates.

**The Milestone 1 deliverable is the data pipeline — Phase 0 → Phase 1, raw charges → clean model-ready monthly series — re-expressed as a portable, orchestrated `extract → validate → load` DAG in this repository (`aim-ml-ops`).** The downstream forecasting phases (2–5) are carried along so the pipeline runs end-to-end and the walkthrough holds, but the engineering work graded for this milestone is the ingestion and data-engineering layer.

Two cross-cutting concerns are noted here rather than drawn into the flow: (1) every artifact is read/written through the single storage switch `src/storage.py` (local `outputs/` when `OUTPUTS_BUCKET` is unset, else `s3://$OUTPUTS_BUCKET`); (2) in production the whole run is scheduled monthly on AWS Fargate via an EventBridge cron — in this repo the same three tasks run as an Airflow DAG or a single-process `uv run` invocation.

```
Milestone 1 scope
┌─────────────────────────────────────────────────────────┐
│  extract  →  validate (Pandera)  →  load (versioned)     │   dags/revenue_pipeline.py
└─────────────────────────────────────────────────────────┘
        │ 01_monthly_series_<run_id>.csv
        ▼
Phase 2 EDA → Phase 3 fit → Phase 4 eval (regression gate) → Phase 5 forecast → manifest ↻
```

---

## The Business Context

Revenue planning and financial visibility for a subscription-based digital-health business. The forecast answers *"what recurring revenue can we expect over the next 1–2 years, and how much of it is durable MRR versus lumpy one-time income?"* — feeding budgeting, hiring, and growth-target decisions. Separating MRR from one-time revenue is the whole point: MRR is the health signal for a subscription business, and lumping it with volatile package/corporate income would distort it.

**Primary users:** the clinic's leadership/operators, via the Streamlit dashboard's revenue page, which reads the forecast tables and the run manifest. Deliverables are stakeholder-facing tables and figures (`05_forecast_12m.csv`, `05_forecast_24m.csv`, `05_scenarios.csv`, and per-series / Total fan charts).

No automated action is taken on the forecast when it is wrong — it is decision-support, so errors surface as mis-set expectations (over/under-hiring, bad budget targets) rather than direct customer harm. The main failure modes are subtle data errors, not model blow-ups, and the pipeline guards against them explicitly:

- A stray non-AUD charge could silently inflate totals → non-AUD rows are dropped with a loud warning.
- Trailing months padded with zeros could fake a "revenue collapse" that anchors the forecast → the series is capped to the last month with real data.
- Overconfident intervals → mitigated by reporting Winkler scores and Bear/Base/Bull scenarios instead of a single point estimate, and by taking the median (not mean) across ensemble members so one explosive model can't blow the interval out.

---

## The Current MLOps State

This model is more mature than a notebook on a laptop — it is a productionized, reproducible pipeline — but it still has gaps typical of a small ML system. An honest inventory:

### What exists

- **Structured, orchestrated pipeline.** Six idempotent phases (0 ingest → 1 data eng → 2 EDA → 3 fit → 4 eval → 5 forecast). The Milestone 1 data layer is now an explicit three-task **Airflow DAG** (`dags/revenue_pipeline.py`: `extract → validate → load`); the task logic lives in `src/revenue/pipeline_m1.py` so it is unit-testable and also runnable without a scheduler (`scripts/run_pipeline.py`).
- **Enforced data-quality gate.** A **Pandera** schema (`src/revenue/validation.py`) contracts the raw charges (status, amount, description, timestamp); the `validate` task — and therefore the DAG run — *fails* on invalid data, so nothing downstream runs on a corrupt extract.
- **Versioned artifacts.** The clean dataset is written with a run-id in the filename (`01_monthly_series_<UTC-run-id>.csv`), alongside a stable `latest` copy and a per-run JSON manifest.
- **Versioning / provenance.** Every successful *forecast* run stamps a manifest (`00_manifest.json`) with input SHA-256, git SHA, pipeline version, data-through date, model selections, and metrics.
- **Automated gates.** An idempotent no-op gate (skip if input bytes and code SHA are unchanged) and a MASE regression gate (hard-fail only if the new error both worsens past a configurable ratio and loses to the seasonal-naïve baseline).
- **Honest validation.** Rolling-origin (expanding-window) cross-validation with MASE (primary), RMSE, Winkler score, and residual diagnostics (Ljung-Box, Shapiro-Wilk). Per-series model choice is selected by this backtest, not hand-picked.
- **Reproducible environment & code quality.** Dependencies are locked with **uv** (`uv.lock`, pinned versions); **Ruff** lints and formats to zero violations; **pre-commit** hooks enforce it; **pytest** covers the pipeline (categorisation, the Pandera gate, cleaning/aggregation, and the end-to-end versioned-artifact contract).
- **Deployment.** Runs on AWS Fargate on a monthly EventBridge cron (with a SIGALRM wall-clock timeout); artifacts are written through the single storage switch (local `outputs/` or `s3://$OUTPUTS_BUCKET`). The fitted model bundle is persisted via joblib with a stale-input guard.

### Honest gaps

- **No CI runs the test suite yet** — tests run manually / via pre-commit before merge. *(Milestone 2 adds a GitHub Actions workflow.)*
- **No model registry / experiment tracking** — "versioning" is git SHA + joblib bundle + JSON manifest, not MLflow/W&B. *(Milestone 2 adds MLflow.)*
- **No production monitoring** of forecast quality beyond the run-time regression gate; no drift dashboard or accuracy-over-time trend.
- **Real data is external to the repo** — this deliverable ships a synthetic, PII-free sample; when Stripe isn't configured the pipeline reads `data/historical_unified_payments.csv` (the real Stripe export is never committed).
- **Small-data regime** — ~5–6 years of monthly points per series constrains model complexity and the reliability of long-horizon intervals.

---

## Milestone 1 Scope

**What we are building this milestone:** the data pipeline (Phase 0 → Phase 1) — the ingestion and data-engineering layer that turns raw payment records into the clean, model-ready monthly series every downstream phase depends on — orchestrated as an `extract → validate → load` Airflow DAG with an enforced Pandera quality gate and a versioned output artifact.

### Inputs

- **Primary:** a fresh Stripe charges snapshot pulled via API (Phase 0, when `STRIPE_API_KEY` is set) → `tables/00_stripe_charges.csv`.
- **Fallback / this deliverable:** `data/historical_unified_payments.csv` — a synthetic, PII-free sample generated deterministically by `scripts/generate_synthetic_data.py`.
- **Required columns (validated fail-loud by Pandera):** `Status`, `Amount`, `Description`, `Created date (UTC)`; optional `Currency`.

### Pipeline tasks

| Task | What it does |
|---|---|
| **extract** | Read the raw payments source (Stripe snapshot if configured, else the synthetic CSV) and persist a raw snapshot artifact (`tables/00_raw_charges.csv`). |
| **validate** | Enforce the raw-charges Pandera schema. **Raises and fails the run on invalid data** — the enforced data-quality check. |
| **load** | Filter to Paid + AUD-only rows (loud warnings on drops); categorise each charge by description → `IV_League` (regex), `Core_MRR` (exact subscription strings), or `Core_OneTime`; aggregate to month-start buckets (impute zero for empty months, cap at the last real month — no trailing-zero padding); engineer features (COVID flag, log1p columns, Total); write the versioned clean dataset + audit summary + run manifest. |

### Outputs

- **`tables/01_monthly_series_<run_id>.csv`** — the *versioned* wide-format monthly DataFrame (three series + engineered columns); the run-id is a UTC timestamp.
- **`tables/01_monthly_series.csv`** — a stable copy; the single artifact Phases 2–5 and the Streamlit page consume.
- **`tables/01_category_summary.csv`** — a per-category transaction audit (counts, totals, % of revenue).
- **`tables/01_run_manifest_<run_id>.json`** — run provenance (row counts, month range, artifact paths).

---

## Methodology

This section walks a viewer through the end-to-end methodology: the phased design, the modelling choices behind it, and the runtime behaviour.

### Design principle: phased, idempotent pipeline

The system is decomposed into six phases, each a module under `src/revenue/`. Phases communicate through persisted artifacts (CSV/joblib/JSON), not in-memory hand-offs, so any phase can be re-run in isolation (`--phase N`) and the whole run is reproducible from its manifest. The Milestone 1 layer wraps Phase 0 → Phase 1 into three discrete, individually runnable tasks that likewise hand off through artifacts, so the Airflow tasks are reproducible in isolation.

```
Phase0_ingest   Stripe API → tables/00_stripe_charges.csv          (skipped if no API key)
Phase1_data     raw charges → clean monthly series + features      (M1: extract→validate→load)
Phase2_eda      STL, outliers, structural breaks, ACF/PACF
Phase3_models   fit candidate models per series → joblib bundle
Phase4_eval     rolling-origin CV, select winner, regression gate
Phase5_forecast 12m / 24m forecasts, intervals, scenarios, charts
```

Two gates make it safe to run unattended monthly:

- **Idempotent no-op gate** — before doing work, compare the SHA-256 of the input Phase 1 would read and the current git SHA against the prior manifest. If both match, log a no-op and exit 0.
- **Regression gate** — after CV, block publication only if a series' MASE both worsens by more than `MPD_REVENUE_MASE_REGRESSION_RATIO` (default 1.30×) *and* falls behind the seasonal-naïve baseline. This runs before any artifact is persisted, so a blocked run leaves the outputs store internally consistent.

### Prototyping / EDA

Before any model is fit, Phase 2 characterizes each series and produces the evidence that drives the model choices in Phase 3:

- STL decomposition (`robust=True`) separates trend / seasonal / remainder.
- Hampel outlier detection on the STL remainder (median ± n·MAD) flags extreme months; capped at the 5 most extreme per series to avoid over-parameterizing. These become intervention dummies.
- Structural-break detection via `ruptures` PELT (RBF kernel) locates level shifts; the first break per series is later injected as a `post_break` step dummy.
- ACF/PACF plots expose the autocorrelation structure that justifies ARIMA-family models.

The prototyping outcome is baked into Phase 3's docstrings as design notes — e.g. MRR was switched away from a BSTS local-level model after a backtest showed worse MASE (2.38 vs. baseline 2.06) and Ljung-Box p≈0 (residual autocorrelation), in favour of the EAT ensemble.

### How each model works

- **EAT Ensemble** (Core_MRR, IV_League) — AutoARIMA + AutoETS + Theta fit via `statsforecast`, each AICc-selected, combined equal-weight at forecast time. A `post_break` exogenous step dummy is injected when Phase 2 found a break. Point forecast = mean of the three; intervals = median across the three (robust to a multiplicative-error ETS whose bounds grow explosively at long horizons).
- **Level model** (Core_OneTime) — one-time revenue is lumpy and mean-reverting with no durable trend, so the forecast is a flat exponentially-weighted level (α = 0.25) computed on clean, non-COVID, non-outlier months. Intervals are fixed-width (empirical σ) — no horizon compounding, because each month is an independent draw around a stable level, not a random walk.
- **Croston's method** (sparse fallback) — for intermittent demand, separately smooths demand size and inter-demand interval and forecasts the constant rate size/interval. Wide, data-driven intervals reflect the sparsity.

All fitted objects are packed into a `ModelBundle` (a dataclass) tagged with the input SHA-256 and serialized to `tables/03_bundle.joblib`.

### Model validation

- Rolling-origin (expanding-window) cross-validation: 8 windows, 12-month horizon, 3-month step, minimum 24-month train — the honest analogue of "forecast the next year, repeatedly, as history grows."
- Metrics: MASE (primary; <1.0 beats the seasonal-naïve baseline), RMSE (large-miss penalty), Winkler score (interval quality), plus Ljung-Box (residual white-noise) and Shapiro-Wilk (normality) diagnostics.
- Auto-selection: for each series the pipeline backtests the competing models and keeps the lower-MASE winner, rewriting `model_type` in the bundle. The re-selected bundle is re-persisted so a standalone `--phase 5` reads the post-eval choice.
- The regression gate runs here, before any write.

### How the forecast is produced

- Each series is forecast for 12 and 24 months via the model chosen in Phase 4; all outputs are clipped to ≥ 0 (revenue can't be negative) and EAT point forecasts get a soft floor at 25% of the recent nonzero median.
- Total Revenue is a bottom-up sum of the three series (points and interval bounds summed).
- Scenarios: Bear / Base / Bull map to the 80%-interval low / point / high.
- Deliverables: `05_forecast_12m.csv`, `05_forecast_24m.csv`, `05_scenarios.csv`, and fan charts per series + Total, plus a quarterly scenario bar chart.

### Reproducibility & provenance

The manifest (`00_manifest.json`) is the single source of truth for *"when was this forecast computed, against what data, by what code, and how good was it?"* — `run_at`, `git_sha`, `pipeline_version`, `input_source`, `input_sha256`, `data_through`, `model_selections`, and per-series metrics. It seeds both the no-op gate (next run's short-circuit) and the regression gate (next run's MASE comparison), closing the loop into a self-consistent monthly refresh.
