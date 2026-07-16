"""Unit + integration tests for the Milestone 1 revenue data pipeline.

Covers the three things that matter for M1: the enforced Pandera data-quality
check (accepts good data, *fails* on bad), the cleaning/aggregation logic, and
the extract→validate→load chain producing a versioned artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import storage  # noqa: E402
from src.revenue import pipeline_m1  # noqa: E402
from src.revenue.phase1_data import (  # noqa: E402
    _categorize,
    add_features,
    aggregate_monthly,
    clean_raw,
)
from src.revenue.validation import validate_raw_charges  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Redirect the storage backend to a per-test temp dir.

    Without this, tests that call pipeline_m1.load()/extract() would write
    through the real local backend and clobber outputs/tables/01_monthly_series.csv
    — the stable artifact the dashboard and forecast phases read. Every test
    gets its own throwaway outputs/ dir instead.
    """
    monkeypatch.setattr(storage, "_LOCAL_DIR", tmp_path / "outputs")


def _raw_frame() -> pd.DataFrame:
    """A tiny valid raw-charges frame spanning three months + all guards."""
    return pd.DataFrame(
        {
            "Status": ["Paid", "Paid", "Paid", "Refunded", "Paid"],
            "Created date (UTC)": pd.to_datetime(
                [
                    "2021-01-10 09:00:00",
                    "2021-01-20 12:00:00",
                    "2021-02-05 15:00:00",
                    "2021-02-06 15:00:00",
                    "2021-03-01 10:00:00",
                ]
            ),
            "Amount": [349.0, 250.0, 599.0, 349.0, 2200.0],
            "Currency": ["aud", "usd", "aud", "aud", "aud"],
            "Description": [
                "Subscription update",
                "IV League - Booking",
                "Subscription creation",
                "Subscription update",
                "Package initiation",
            ],
        }
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


# ── Pandera data-quality gate ────────────────────────────────────────────────


def test_schema_accepts_valid_data():
    out = validate_raw_charges(_raw_frame())
    assert len(out) == 5


def test_schema_rejects_missing_required_column():
    bad = _raw_frame().drop(columns=["Amount"])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_charges(bad)


def test_schema_rejects_non_numeric_amount():
    bad = _raw_frame()
    bad.loc[0, "Amount"] = "not-a-number"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_charges(bad)


# ── Cleaning + aggregation ────────────────────────────────────────────────────


def test_clean_raw_filters_non_paid_and_non_aud():
    cleaned = clean_raw(_raw_frame())
    # Refunded row dropped, USD row dropped → 3 rows remain.
    assert len(cleaned) == 3
    assert set(cleaned.columns) == {"date", "amount", "category", "Description"}


def test_aggregate_and_features_shape():
    monthly = add_features(aggregate_monthly(clean_raw(_raw_frame())))
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


def test_pipeline_produces_versioned_artifact():
    # Seed the raw snapshot the validate/load tasks read (isolated storage).
    storage.save_csv(pipeline_m1.RAW_SNAPSHOT_KEY, _raw_frame(), index=False)

    assert pipeline_m1.validate() == 5

    run_id = "test0000T000000Z"
    manifest = pipeline_m1.load(run_id=run_id)

    assert manifest["run_id"] == run_id
    assert storage.exists(pipeline_m1.versioned_clean_key(run_id))
    assert storage.exists(pipeline_m1.CLEAN_LATEST_KEY)
    # Versioned filename carries the run id (spec requirement).
    assert run_id in pipeline_m1.versioned_clean_key(run_id)


def test_load_fails_when_nothing_survives_cleaning():
    # All rows non-Paid: passes the Pandera gate but clean_raw drops everything.
    bad = _raw_frame()
    bad["Status"] = "Refunded"
    storage.save_csv(pipeline_m1.RAW_SNAPSHOT_KEY, bad, index=False)
    with pytest.raises(ValueError, match="no rows survived cleaning"):
        pipeline_m1.load(run_id="testEmptyT000000Z")
