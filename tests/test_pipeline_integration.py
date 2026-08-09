"""Pipeline unit + integration tests (Milestone 2 — Deliverable 2c).

Covers the cleaning/aggregation logic and the extract → validate → load chain
running end-to-end against a small synthetic dataset, verifying the versioned
output artifact is produced.
"""

from __future__ import annotations

import pytest

from src import storage
from src.revenue import phase1_data, pipeline_m1
from src.revenue.phase1_data import (
    _categorize,
    add_features,
    aggregate_monthly,
    clean_raw,
)

# ── Categorisation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "desc,expected",
    [
        ("Subscription creation", "MPD_Core_MRR"),
        ("Subscription update", "MPD_Core_MRR"),
        ("IV League - Booking", "IV_League"),
        ("Package initiation", "MPD_Core_OneTime"),
        ("", "MPD_Core_OneTime"),
    ],
)
def test_categorize(desc, expected):
    assert _categorize(desc) == expected


# ── Cleaning + aggregation ────────────────────────────────────────────────────


def test_clean_raw_filters_non_paid_and_non_aud(raw_frame):
    cleaned = clean_raw(raw_frame)
    # Refunded row dropped, USD row dropped → 3 rows remain.
    assert len(cleaned) == 3
    assert set(cleaned.columns) == {"date", "amount", "category", "Description"}


def test_aggregate_and_features_shape(raw_frame):
    monthly = add_features(aggregate_monthly(clean_raw(raw_frame)))
    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime", "Total", "is_covid_startup"]:
        assert col in monthly.columns
    # Total == sum of the three streams, row-wise.
    streams = monthly[["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]].sum(axis=1)
    assert (streams - monthly["Total"]).abs().max() < 1e-6


# ── End-to-end extract → validate → load ─────────────────────────────────────


def test_extract_reads_source_and_persists_snapshot():
    # No Stripe snapshot in the isolated store → falls back to the synthetic
    # data/ CSV; extract must persist the raw snapshot the next tasks read.
    meta = pipeline_m1.extract()
    assert meta["input_source"] in {"historical_csv", "stripe_api"}
    assert meta["rows"] > 0
    assert storage.exists(pipeline_m1.RAW_SNAPSHOT_KEY)


def test_full_pipeline_end_to_end_on_synthetic_data(raw_frame):
    # Seed a small synthetic extract as the preferred source, then run the
    # whole extract → validate → load chain and verify the output artifact.
    storage.save_csv(phase1_data.SNAPSHOT_KEY, raw_frame, index=False)

    run_id = "testE2E0T000000Z"
    manifest = pipeline_m1.run(run_id=run_id)

    assert manifest["run_id"] == run_id
    assert manifest["rows_in"] == 5
    assert manifest["rows_clean"] == 3  # Refunded + USD rows dropped
    assert storage.exists(pipeline_m1.versioned_clean_key(run_id))
    assert storage.exists(pipeline_m1.CLEAN_LATEST_KEY)
    assert storage.exists(f"tables/01_run_manifest_{run_id}.json")


def test_pipeline_produces_versioned_artifact(raw_frame):
    # Seed the raw snapshot the validate/load tasks read (isolated storage).
    storage.save_csv(pipeline_m1.RAW_SNAPSHOT_KEY, raw_frame, index=False)

    assert pipeline_m1.validate() == 5

    run_id = "test0000T000000Z"
    manifest = pipeline_m1.load(run_id=run_id)

    assert manifest["run_id"] == run_id
    assert storage.exists(pipeline_m1.versioned_clean_key(run_id))
    assert storage.exists(pipeline_m1.CLEAN_LATEST_KEY)
    # Versioned filename carries the run id (spec requirement).
    assert run_id in pipeline_m1.versioned_clean_key(run_id)


def test_load_fails_when_nothing_survives_cleaning(raw_frame):
    # All rows non-Paid: passes the Pandera gate but clean_raw drops everything.
    bad = raw_frame
    bad["Status"] = "Refunded"
    storage.save_csv(pipeline_m1.RAW_SNAPSHOT_KEY, bad, index=False)
    with pytest.raises(ValueError, match="no rows survived cleaning"):
        pipeline_m1.load(run_id="testEmptyT000000Z")


# ── Full tracked training pipeline (phases 1–4 + MLflow) + phase 5 ───────────


def test_training_run_logs_to_mlflow_and_registers_model(tmp_path, monkeypatch):
    """models/train.py end-to-end against a temp MLflow store.

    Runs the tracked training pipeline (data → EDA → fit → CV evaluation)
    with a reduced CV budget (2 rolling-origin windows instead of 8 — enough
    to exercise every code path at a fraction of the fit time), then verifies
    the MLflow run carries the required params/metrics, the model is
    registered, and phase 5 can produce forecasts from the persisted bundle.
    """
    from mlflow.tracking import MlflowClient

    from models import train as train_module
    from src.revenue import phase4_eval, phase5_forecast

    # Shrink the CV budget. `_cv_windows` binds CV_WINDOWS as a default at
    # definition time, so wrap the function rather than patching the constant.
    orig_cv_windows = phase4_eval._cv_windows
    monkeypatch.setattr(
        phase4_eval,
        "_cv_windows",
        lambda n, h=12: orig_cv_windows(n, h, n_windows=2),
    )

    # Tracking URI comes from the environment — point it at a throwaway store.
    uri = f"sqlite:///{tmp_path}/mlflow-test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    results = train_module.train()
    assert set(results) == {"IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"}

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    exp = client.get_experiment_by_name(train_module.EXPERIMENT_NAME)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert len(run.data.params) >= 3, "spec requires ≥3 logged params"
    assert len(run.data.metrics) >= 2, "spec requires ≥2 logged metrics"

    # Model Registry: version 1 exists and carries the milestone version tag.
    mv = client.get_model_version(train_module.REGISTERED_MODEL_NAME, "1")
    assert mv.tags["milestone"] == "2"

    # Phase 5 consumes the bundle the tracked run persisted (staleness-checked).
    monthly = storage.load_csv(
        "tables/01_monthly_series.csv", index_col="month", parse_dates=["month"]
    )
    forecasts = phase5_forecast.run(None, monthly)
    assert set(forecasts) == {"12m", "24m"}
    assert storage.exists("tables/05_forecast_12m.csv")
    assert storage.exists("tables/05_forecast_24m.csv")
