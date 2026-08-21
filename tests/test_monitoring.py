"""Tests for reproducible, privacy-safe monitoring data."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.model_service import ModelLoadError
from monitoring.run_report import (
    DRIFT_THRESHOLD,
    MONITORED_COLUMNS,
    build_serving_datasets,
    load_monitoring_service,
    population_stability_index,
)


class DeterministicForecastService:
    config = SimpleNamespace(model_name="test-model")
    model_version = "1"
    run_id = "test-run"

    def forecast(self, horizon_months: int, interval_level: int):
        assert interval_level == 80
        return [
            {"total_prediction": float(month * 1_000)}
            for month in range(1, horizon_months + 1)
        ]


def test_serving_datasets_are_reproducible_pii_free_and_model_derived():
    first_reference, first_current = build_serving_datasets(
        DeterministicForecastService(), sample_size=100, seed=42
    )
    second_reference, second_current = build_serving_datasets(
        DeterministicForecastService(), sample_size=100, seed=42
    )

    pd.testing.assert_frame_equal(first_reference, second_reference)
    pd.testing.assert_frame_equal(first_current, second_current)
    assert list(first_reference.columns) == MONITORED_COLUMNS
    assert list(first_current.columns) == MONITORED_COLUMNS
    assert first_reference["horizon_months"].between(1, 24).all()
    assert first_reference["prediction"].gt(0).all()


def test_designed_shift_exceeds_documented_psi_threshold():
    reference, current = build_serving_datasets(
        DeterministicForecastService(), sample_size=400, seed=20260821
    )

    assert (
        population_stability_index(reference["horizon_months"], current["horizon_months"])
        > DRIFT_THRESHOLD
    )
    assert (
        population_stability_index(reference["prediction"], current["prediction"])
        > DRIFT_THRESHOLD
    )


def test_monitoring_requires_enough_rows():
    with pytest.raises(ValueError, match="at least 50"):
        build_serving_datasets(DeterministicForecastService(), sample_size=49)


def test_monitoring_missing_model_gives_setup_instructions(monkeypatch):
    def failed_load():
        raise ModelLoadError("No compatible registered model.")

    monkeypatch.setattr(
        "monitoring.run_report.ModelService.load",
        failed_load,
    )

    with pytest.raises(SystemExit, match="uv run python models/train.py"):
        load_monitoring_service()
