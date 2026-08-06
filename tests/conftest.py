"""Shared fixtures for the Milestone 2 test suite.

The suite is split into the three spec-mandated files:

  test_data_validation.py      — Pandera schema gate (Deliverable 2a)
  test_model_quality.py        — threshold + output-format checks (Deliverable 2b)
  test_pipeline_integration.py — end-to-end pipeline runs (Deliverable 2c)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path, monkeypatch):
    """Redirect the storage backend to a per-test temp dir.

    Without this, tests that call pipeline_m1.load()/extract() would write
    through the real local backend and clobber outputs/tables/01_monthly_series.csv
    — the stable artifact the dashboard and forecast phases read. Every test
    gets its own throwaway outputs/ dir instead.
    """
    monkeypatch.setattr(storage, "_LOCAL_DIR", tmp_path / "outputs")


@pytest.fixture
def raw_frame() -> pd.DataFrame:
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
