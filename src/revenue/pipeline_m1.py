"""Milestone 1 data pipeline: extract → validate → load.

This is the thin orchestration layer the Airflow DAG (`dags/revenue_pipeline.py`)
and the standalone runner (`scripts/run_pipeline.py`) both call. It reuses the
battle-tested cleaning/aggregation logic in `phase1_data` and the enforced
Pandera contract in `validation`, exposing them as three discrete, individually
runnable steps that hand off through persisted artifacts (never in-memory), so
the DAG tasks are reproducible in isolation.

Steps
-----
extract()   read the raw payments source → persist a raw snapshot artifact.
validate()  enforce the raw-charges data-quality contract (Pandera); raises on
            bad data so nothing downstream runs on a corrupt extract.
load(df)    clean + aggregate to a monthly series + engineer features → write a
            *versioned* clean dataset (run-id in the filename) plus a stable
            `latest` copy and a per-run manifest.

Output artifact naming
----------------------
The versioned clean dataset is written as
``tables/01_monthly_series_<run_id>.csv`` where ``run_id`` is a UTC timestamp
(e.g. ``01_monthly_series_20260716T101500Z.csv``). A stable
``tables/01_monthly_series.csv`` copy is also written so the downstream
forecasting phases and the Streamlit page keep reading a fixed key.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src import storage

from .phase1_data import (
    add_features,
    aggregate_monthly,
    category_summary,
    clean_raw,
    read_source,
)
from .validation import validate_raw_charges

RAW_SNAPSHOT_KEY = "tables/00_raw_charges.csv"
CLEAN_LATEST_KEY = "tables/01_monthly_series.csv"
CATEGORY_SUMMARY_KEY = "tables/01_category_summary.csv"


def make_run_id() -> str:
    """UTC timestamp run id, e.g. ``20260716T101500Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def versioned_clean_key(run_id: str) -> str:
    return f"tables/01_monthly_series_{run_id}.csv"


# ── Task 1: Extract ────────────────────────────────────────────────────────────


def extract() -> dict:
    """EXTRACT — read raw payments from the source and persist a snapshot.

    Prefers a Stripe API snapshot (``tables/00_stripe_charges.csv``) when it
    exists; otherwise reads the synthetic historical CSV under ``data/`` — via
    the shared ``phase1_data.read_source`` so the M1 extract and the full
    pipeline never diverge on source precedence. The untouched frame is
    persisted to ``RAW_SNAPSHOT_KEY`` so the validate and load tasks operate on
    a stable, re-readable artifact rather than a hand-off in memory. Returns
    provenance metadata for the manifest.
    """
    df, meta = read_source()
    storage.save_csv(RAW_SNAPSHOT_KEY, df, index=False)
    meta["rows"] = int(len(df))
    print(f"  extract: {len(df):,} rows from {meta['input_source']} → {RAW_SNAPSHOT_KEY}")
    return meta


# ── Task 2: Validate ───────────────────────────────────────────────────────────


def validate() -> int:
    """VALIDATE — enforce the raw-charges Pandera contract.

    Reads the extracted snapshot and validates it. Raises
    ``pandera.errors.SchemaErrors`` on any violation, which fails the pipeline
    (the whole point of an *enforced* data-quality check). Returns the row count
    on success.
    """
    df = storage.load_csv(RAW_SNAPSHOT_KEY, parse_dates=["Created date (UTC)"])
    validated = validate_raw_charges(df)
    print(f"  validate: {len(validated):,} rows passed the raw-charges schema")
    return int(len(validated))


# ── Task 3: Load ─────────────────────────────────────────────────────────────


def load(run_id: str | None = None) -> dict:
    """LOAD — clean, aggregate to monthly series, engineer features, persist.

    Writes a *versioned* clean dataset (``01_monthly_series_<run_id>.csv``), a
    stable ``01_monthly_series.csv`` copy for downstream consumers, and a
    per-category audit summary. Returns a small manifest dict.
    """
    run_id = run_id or make_run_id()

    raw = storage.load_csv(RAW_SNAPSHOT_KEY, parse_dates=["Created date (UTC)"])
    cleaned = clean_raw(raw)

    # Post-clean guard: the Pandera gate validates the raw extract, but it does
    # not constrain Status/Currency values, so a snapshot that is entirely
    # non-Paid or non-AUD passes validation yet leaves nothing after cleaning.
    # Fail loud rather than publish an all-zero monthly series to the stable
    # artifact the dashboard and forecast phases read.
    if cleaned.empty:
        raise ValueError(
            f"no rows survived cleaning ({len(raw):,} raw rows in) — the extract "
            "is entirely non-Paid and/or non-AUD; refusing to write an empty series."
        )

    monthly = add_features(aggregate_monthly(cleaned))

    versioned_key = versioned_clean_key(run_id)
    storage.save_csv_atomic(versioned_key, monthly)
    storage.save_csv_atomic(CLEAN_LATEST_KEY, monthly)
    storage.save_csv(CATEGORY_SUMMARY_KEY, category_summary(cleaned))

    manifest = {
        "run_id": run_id,
        "rows_in": int(len(raw)),
        "rows_clean": int(len(cleaned)),
        "months": int(len(monthly)),
        "month_start": monthly.index.min().strftime("%Y-%m-%d"),
        "month_end": monthly.index.max().strftime("%Y-%m-%d"),
        "versioned_artifact": versioned_key,
        "latest_artifact": CLEAN_LATEST_KEY,
    }
    storage.save_json(f"tables/01_run_manifest_{run_id}.json", manifest, indent=2)
    print(f"  load: {manifest['months']} months → {versioned_key} (+ latest copy)")
    return manifest


def run(run_id: str | None = None) -> dict:
    """Run the full extract → validate → load chain in order (no Airflow)."""
    run_id = run_id or make_run_id()
    print(f"[pipeline_m1] run_id={run_id}")
    extract()
    validate()
    return load(run_id)
