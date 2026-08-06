"""Model-quality tests (Milestone 2 — Deliverable 2b).

These test the model's *outputs* against business-relevant criteria — not
model internals: a forecast-accuracy threshold on a held-out test set, and
output-format contracts (shape, no NaNs, non-negative revenue, ordered
prediction intervals).

The model is a revenue *forecaster* (regression over time series), so the
classification metrics named in the spec (accuracy/F1) translate to the
project's headline accuracy metric, MASE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.revenue.config import DATA_FILE
from src.revenue.phase1_data import add_features, aggregate_monthly, clean_raw
from src.revenue.phase3_models import SeriesModels, croston_forecast, fit_eat_ensemble
from src.revenue.phase4_eval import mase
from src.revenue.phase5_forecast import _forecast_series

# ── Thresholds ───────────────────────────────────────────────────────────────
# MASE = MAE(forecast) / MAE(seasonal-naïve on the training series), so a
# value below 1.0 means the model out-forecasts the seasonal-naïve baseline
# on the held-out months. Milestone 1 rolling-origin CV measured MASE 0.39
# (MPD_Core_MRR) to 0.81 (IV_League) — see outputs/tables/04_evaluation_results.csv
# — so 1.0 is a conservative regression floor: it tolerates single-holdout
# noise above the CV point estimates, but fails if the model degrades to
# worse-than-baseline.
MASE_THRESHOLD = 1.0

# Held-out window = one full seasonal cycle (12 months) — the same horizon the
# business plans against (FORECAST_SHORT in config.py).
HOLDOUT_MONTHS = 12
FORECAST_HORIZON = 12


# ── Fixtures (fit once per session — EAT ensemble fitting is the slow part) ──


@pytest.fixture(scope="session")
def monthly_series() -> pd.DataFrame:
    """Monthly revenue series built from the committed synthetic sample data."""
    raw = pd.read_csv(DATA_FILE, parse_dates=["Created date (UTC)"])
    return add_features(aggregate_monthly(clean_raw(raw)))


@pytest.fixture(scope="session")
def fitted_mrr(monthly_series):
    """EAT ensemble fit on MPD_Core_MRR minus a 12-month holdout tail."""
    series = monthly_series["MPD_Core_MRR"]
    train, test = series[:-HOLDOUT_MONTHS], series[-HOLDOUT_MONTHS:]
    sf, sf_df = fit_eat_ensemble(train, "MPD_Core_MRR")
    sm = SeriesModels(
        name="MPD_Core_MRR",
        train=train,
        train_log=None,
        sf=sf,
        sf_df=sf_df,
        bsts_result=None,
        exog_train=None,
        outlier_dummies=pd.DataFrame(index=train.index),
        model_type="EAT",
    )
    return sm, train, test


# ── Accuracy threshold on a held-out test set ────────────────────────────────


def test_mrr_holdout_mase_beats_threshold(fitted_mrr):
    sm, train, test = fitted_mrr
    point, *_ = _forecast_series(sm, HOLDOUT_MONTHS)
    holdout_mase = mase(test.values, point, train.values)
    assert not np.isnan(holdout_mase)
    assert holdout_mase < MASE_THRESHOLD, (
        f"holdout MASE {holdout_mase:.3f} ≥ {MASE_THRESHOLD} — "
        "model no longer beats the seasonal-naïve baseline"
    )


# ── Output-format contracts ──────────────────────────────────────────────────


def test_forecast_output_format(fitted_mrr):
    sm, _, _ = fitted_mrr
    point, lo_80, hi_80, lo_95, hi_95 = _forecast_series(sm, FORECAST_HORIZON)

    for arr in (point, lo_80, hi_80, lo_95, hi_95):
        assert arr.shape == (FORECAST_HORIZON,)
        assert not np.isnan(arr).any(), "forecast contains NaNs"
        assert (arr >= 0).all(), "revenue forecast must be non-negative"

    # Prediction intervals must be internally ordered.
    assert (lo_80 <= hi_80).all()
    assert (lo_95 <= hi_95).all()


def test_croston_forecast_format(monthly_series):
    # The sparse-demand fallback path must honour the same output contract.
    fc = croston_forecast(monthly_series["IV_League"], FORECAST_HORIZON)
    assert fc.shape == (FORECAST_HORIZON,)
    assert not np.isnan(fc).any()
    assert (fc >= 0).all()
