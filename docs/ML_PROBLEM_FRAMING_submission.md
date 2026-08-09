# Milestone 1 — ML Problem Framing
### Revenue Forecasting Pipeline · Australian Health & Longevity Platform
MLOps LT 3 · Repository: [`aim-ml-ops`](https://github.com/ejbbinaday/aim-ml-ops)

---

## 1. The ML System

A monthly, multi-series time-series forecasting pipeline for an Australian medical clinic and digital-health platform (preventative medicine, longevity, peak-performance). It projects the platform's revenue across three streams: recurring subscriptions, IV League bookings, and one-time income. Each stream is forecast independently using the model that performs best during backtesting, measured by MASE (Mean Absolute Scaled Error), and the forecasts are combined into a bottom-up total revenue projection.

| Series | Captures | Model (auto-selected by MASE) |
|---|---|---|
| Core_MRR | Recurring subscriptions (Subscription creation/update) | EAT ensemble (AutoARIMA, AutoETS, Theta) vs. Level |
| IV_League | Calendly "IV League" bookings | EAT ensemble vs. Croston (sparse demand) |
| Core_OneTime | Packages, corporate/Keyman, invoices, consults | Level vs. EAT vs. Croston |

Here, EAT is the ensemble of AutoARIMA, AutoETS, and Theta; Level is a simple level forecast; and Croston is a method for intermittent, sparse demand.

Predicts: per series, a 12- and 24-month AUD forecast with 80%/95% prediction intervals, plus a Total projection and Bear/Base/Bull scenarios.

Feature set (univariate per series): month-start aggregated revenue per category, with log1p variants; a COVID/startup structural-break flag; Hampel outlier-intervention dummies; PELT (Pruned Exact Linear Time) structural-break step dummies; and model-internal terms (seasonal period 12, ARIMA/ETS orders selected by AICc, the corrected Akaike Information Criterion). There are no cross-series or external regressors. The streams are modelled separately and summed.

## 2. The Business Context

The forecast answers *"what revenue can we expect over the next 1–2 years, and how much is durable MRR versus lumpy one-time income?"*, feeding budgeting, hiring, and growth-target decisions. Separating MRR from one-time revenue is the whole point: MRR is the health signal for a subscription business, and lumping it with volatile package income distorts it.

Who uses the output: the clinic's leadership and operators, via a Streamlit revenue page that reads the forecast tables and run manifest (stakeholder-facing tables and fan charts).

What happens when it's wrong: it is decision-support, with no automated action taken, so errors surface as mis-set expectations (over/under-hiring, bad budget targets), not direct customer harm. The real risk is subtle *data* errors, which the pipeline guards against explicitly. Non-AUD charges are dropped with a warning (no FX inflation); trailing zero-months are capped, not padded (no fake "revenue collapse"); and intervals use Bear/Base/Bull scenarios and a median-across-ensemble to avoid overconfidence.

## 3. The Current MLOps State

Before Milestone 1, the project already had forecasting logic, backtesting, model-selection procedures, and stakeholder-facing outputs. For Milestone 1, the team is formalizing the data layer as an automated and reproducible Airflow pipeline with Pandera validation, versioned outputs, and run-level provenance.

Before Milestone 1

- Forecasting logic: the suite of models (EAT ensemble, Level, Croston) with per-series auto-selection by backtest.
- Honest validation: rolling-origin cross-validation with MASE, RMSE, and Winkler score, plus Ljung-Box and Shapiro-Wilk diagnostics. Model choice is backtested, not hand-picked.
- Stakeholder-facing outputs: a Streamlit revenue page with forecast tables and fan charts.
- Reproducibility groundwork: uv-locked deps, Ruff-clean code, pre-commit hooks, and pytest.

For Milestone 1

- Airflow-orchestrated data pipeline: six idempotent phases (ingest, data-eng, EDA, fit, eval, forecast), with the data layer built as an explicit Airflow DAG (extract, validate, load) whose logic lives in `pipeline_m1.py` and is also runnable without a scheduler.
- Enforced data-quality gate: a Pandera schema that fails the run on invalid raw charges.
- Versioning and provenance: a run manifest stamps input SHA-256, git SHA, code version, data-through date, model selections, and metrics. Clean datasets carry a run-id in the filename.
- Safety gates: an idempotent no-op gate (skip unchanged input) and a MASE regression gate (block a forecast that both worsens past a ratio and loses to seasonal-naïve).
- Deployment path: the pipeline runs monthly on AWS Fargate (EventBridge cron), with artifacts routed through a single local/S3 storage switch.

Honest gaps

- No CI runs the tests yet (*M2: GitHub Actions*).
- No experiment tracking or model registry. Versioning is git SHA plus manifest, not MLflow (*M2*).
- No production monitoring of forecast quality (no drift or accuracy-over-time dashboard).
- Real Stripe data lives outside the repo. This deliverable ships a synthetic, PII-free sample.
- Small-data regime (~5–6 yrs of monthly points) limits model complexity and long-horizon intervals.

## 4. Milestone 1 Scope

What we are building for Milestone 1: an Airflow-orchestrated extract, validate, and load pipeline that converts raw payment records into clean, validated, and versioned monthly revenue series for downstream model training, forecasting, and reporting. This is the Phase 0 to Phase 1 data layer that every downstream phase depends on.

Inputs

- Primary: a Stripe charges snapshot pulled via API when `STRIPE_API_KEY` is set.
- This deliverable: `data/historical_unified_payments.csv`, a synthetic, PII-free sample (deterministically generated).
- Required columns (Pandera-validated, fail-loud): `Status`, `Amount`, `Description`, `Created date (UTC)`; optional `Currency`.

Pipeline tasks: extract (read source, persist raw snapshot), then validate (Pandera schema; the run fails on bad data), then load (Paid and AUD-only filtering, categorise into the 3 streams, aggregate to month-start buckets, engineer features).

Outputs

- `tables/01_monthly_series_<run_id>.csv`: the versioned wide-format monthly series (three streams plus engineered columns); run-id is a UTC timestamp.
- `tables/01_monthly_series.csv`: stable copy consumed by Phases 2–5 and the dashboard.
- `tables/01_category_summary.csv`: per-category transaction audit.
- `tables/01_run_manifest_<run_id>.json`: run provenance (row counts, month range, artifact paths).

---

*Full architecture, modelling rationale, and validation methodology: [`ML_PROBLEM_FRAMING.md`](ML_PROBLEM_FRAMING.md). Run instructions: repository `README.md`.*
