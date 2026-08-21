"""Tests for reproducible, privacy-safe monitoring data."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from monitoring.run_report import (
    DRIFT_THRESHOLD,
    INPUT_DRIFT_COLUMN,
    PREDICTION_DRIFT_COLUMN,
    build_serving_datasets,
    population_stability_index,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    first = build_serving_datasets(
        DeterministicForecastService(), sample_size=100, seed=42
    )
    second = build_serving_datasets(
        DeterministicForecastService(), sample_size=100, seed=42
    )

    for first_frame, second_frame in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_frame, second_frame)

    reference_requests, current_requests, reference_predictions, current_predictions = first
    assert list(reference_requests.columns) == [INPUT_DRIFT_COLUMN]
    assert list(current_requests.columns) == [INPUT_DRIFT_COLUMN]
    assert list(reference_predictions.columns) == [PREDICTION_DRIFT_COLUMN]
    assert list(current_predictions.columns) == [PREDICTION_DRIFT_COLUMN]
    assert reference_requests[INPUT_DRIFT_COLUMN].between(1, 24).all()
    assert reference_predictions[PREDICTION_DRIFT_COLUMN].gt(0).all()
    assert len(reference_predictions) > len(reference_requests)
    assert len(current_predictions) > len(current_requests)


def test_designed_shift_exceeds_documented_psi_threshold():
    reference_requests, current_requests, reference_predictions, current_predictions = (
        build_serving_datasets(
            DeterministicForecastService(), sample_size=400, seed=20260821
        )
    )

    input_score = population_stability_index(
        reference_requests[INPUT_DRIFT_COLUMN], current_requests[INPUT_DRIFT_COLUMN]
    )
    prediction_score = population_stability_index(
        reference_predictions[PREDICTION_DRIFT_COLUMN],
        current_predictions[PREDICTION_DRIFT_COLUMN],
    )
    assert input_score > DRIFT_THRESHOLD
    assert prediction_score > DRIFT_THRESHOLD
    assert prediction_score != pytest.approx(input_score)


def test_monitoring_requires_enough_rows():
    with pytest.raises(ValueError, match="at least 50"):
        build_serving_datasets(DeterministicForecastService(), sample_size=49)


def test_direct_script_entry_point_reports_missing_model_without_import_error(tmp_path: Path):
    environment = os.environ.copy()
    environment.update(
        {
            "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'empty-mlflow.db'}",
            "MODEL_VERSION": "champion",
        }
    )

    result = subprocess.run(
        [sys.executable, "monitoring/run_report.py"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "This report needs a compatible registered model" in output
    assert "ModuleNotFoundError" not in output
    assert "Traceback" not in output
