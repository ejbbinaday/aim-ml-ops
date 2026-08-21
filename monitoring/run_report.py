"""Generate PII-free input and prediction drift reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from evidently.core.report import Snapshot

    from app.model_service import ModelService

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
INPUT_DRIFT_COLUMN = "horizon_months"
PREDICTION_DRIFT_COLUMN = "prediction"
HORIZONS = np.array([3, 6, 12, 24])
REFERENCE_WEIGHTS = np.array([0.55, 0.30, 0.10, 0.05])
CURRENT_WEIGHTS = np.array([0.05, 0.10, 0.30, 0.55])
DRIFT_THRESHOLD = 0.2
DEFAULT_SAMPLE_SIZE = 400
DEFAULT_SEED = 20260821
REPORT_FILENAMES = {
    "input_drift": ("evidently_report.html", "evidently_report.json"),
    "prediction_drift": (
        "evidently_prediction_drift.html",
        "evidently_prediction_drift.json",
    ),
}


def _forecasts_by_horizon(
    service: ModelService,
) -> dict[int, list[dict[str, object]]]:
    """Load each representative API output once for deterministic sampling."""

    return {
        int(horizon): service.forecast(int(horizon), interval_level=80)
        for horizon in HORIZONS
    }


def build_serving_datasets(
    service: ModelService,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build deterministic request and output samples without personal information.

    These are labelled demonstration datasets. They model a shift from mostly
    short-horizon requests to mostly long-horizon requests; they are not
    presented as captured production traffic. Request-level input data and
    monthly forecast-output data are kept separate so prediction drift is not
    merely a relabelled copy of the requested horizon distribution.
    """

    if sample_size < 50:
        raise ValueError("sample_size must be at least 50 for a meaningful drift example")

    forecasts_by_horizon = _forecasts_by_horizon(service)
    rng = np.random.default_rng(seed)

    def sample_requests(weights: np.ndarray) -> pd.DataFrame:
        horizons = rng.choice(HORIZONS, size=sample_size, p=weights)
        return pd.DataFrame({INPUT_DRIFT_COLUMN: horizons.astype(int)})

    def expand_predictions(requests: pd.DataFrame) -> pd.DataFrame:
        predictions = [
            float(point["total_prediction"])
            for horizon in requests[INPUT_DRIFT_COLUMN]
            for point in forecasts_by_horizon[int(horizon)]
        ]
        return pd.DataFrame({PREDICTION_DRIFT_COLUMN: predictions})

    reference_requests = sample_requests(REFERENCE_WEIGHTS)
    current_requests = sample_requests(CURRENT_WEIGHTS)
    return (
        reference_requests,
        current_requests,
        expand_predictions(reference_requests),
        expand_predictions(current_requests),
    )


def population_stability_index(reference: pd.Series, current: pd.Series) -> float:
    """Calculate PSI on the shared set of observed values."""

    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    reference_share = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    current_share = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    epsilon = 1e-6
    reference_share = reference_share.clip(lower=epsilon)
    current_share = current_share.clip(lower=epsilon)
    score = ((current_share - reference_share) * np.log(current_share / reference_share)).sum()
    return float(score)


def create_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    column: str,
    name: str,
    drift_kind: str,
) -> Snapshot:
    """Run one explicitly labelled PSI-based drift report."""

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(
        [
            DataDriftPreset(
                columns=[column],
                method="psi",
                threshold=DRIFT_THRESHOLD,
            )
        ],
        metadata={
            "dataset_kind": "deterministic synthetic serving traffic",
            "purpose": "Final Project monitoring demonstration",
            "drift_kind": drift_kind,
        },
        tags=["final-project", drift_kind, "pii-free"],
    )
    return report.run(
        current_data=current,
        reference_data=reference,
        name=name,
    )


def _drift_score(snapshot: Snapshot, column: str) -> float:
    """Read the displayed ValueDrift score from an Evidently snapshot."""

    for metric in snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        if config.get("type") == "evidently:metric_v2:ValueDrift" and config.get(
            "column"
        ) == column:
            return float(metric["value"])
    raise RuntimeError(f"Evidently did not return a drift score for {column!r}")


def _write_findings(summary: dict[str, object], path: Path) -> None:
    input_score = float(summary["horizon_months_psi"])
    prediction_score = float(summary["prediction_psi"])
    input_state = "detected" if input_score >= DRIFT_THRESHOLD else "not detected"
    prediction_state = "detected" if prediction_score >= DRIFT_THRESHOLD else "not detected"
    model = summary["model"]
    assert isinstance(model, dict)

    text = f"""# Drift-monitoring findings

## Executive summary

This reproducible demonstration detected both input drift and prediction drift. The simulated
current period contains far more long-horizon forecast requests than the reference period. Because
longer requests return more future monthly forecasts, the per-month output distribution shifted too.
This is a monitoring signal to investigate, **not proof that the model has become inaccurate**.

## Results

| Signal | PSI | Threshold | Result |
|---|---:|---:|---|
| API input: `horizon_months` | {input_score:.3f} | {DRIFT_THRESHOLD:.1f} | Drift {input_state} |
| Model output: monthly `prediction` | {prediction_score:.3f} | {DRIFT_THRESHOLD:.1f} | Drift {prediction_state} |

The report used {summary['reference_rows']} reference requests and {summary['current_rows']}
current requests, producing {summary['reference_prediction_rows']} reference forecast rows and
{summary['current_prediction_rows']} current forecast rows. It evaluated registered model
`{model['name']}` Version `{model['version']}` (run `{model['run_id']}`).

## Interpretation and action

The input shift may mean users are planning farther ahead, the API's calling pattern changed, or
the serving mix changed. The prediction shift is related to that input shift because long-horizon
requests include more distant monthly outputs, but it is measured on the returned monthly forecast
values rather than by renaming the horizon categories. The team should first
validate the request logs and confirm whether the shift reflects a real business change. If it does,
segment monitoring by horizon and compare forecasts with actual revenue as it arrives. Retraining is
appropriate only if later accuracy monitoring shows sustained degradation; drift alone is not a
retraining trigger.

## Important limitation

The two datasets are deterministic, synthetic, and PII-free. They intentionally create a visible
shift for the course demonstration and must not be described as observed production behavior. This
system currently monitors request and prediction distributions without ground truth; production
operation should add delayed actual-revenue accuracy checks and alert ownership.
"""
    path.write_text(text, encoding="utf-8")


def run(output_dir: Path = REPORTS_DIR) -> dict[str, object]:
    """Load the configured model and write all monitoring deliverables."""

    from app.model_service import ModelService

    output_dir.mkdir(parents=True, exist_ok=True)
    service = ModelService.load()
    (
        reference_requests,
        current_requests,
        reference_predictions,
        current_predictions,
    ) = build_serving_datasets(service)
    reference_requests.to_csv(output_dir / "reference_serving_data.csv", index=False)
    current_requests.to_csv(output_dir / "current_serving_data.csv", index=False)
    reference_predictions.to_csv(output_dir / "reference_prediction_data.csv", index=False)
    current_predictions.to_csv(output_dir / "current_prediction_data.csv", index=False)

    snapshots = {
        "input_drift": create_evidently_report(
            reference_requests,
            current_requests,
            column=INPUT_DRIFT_COLUMN,
            name="Input data drift: requested forecast horizons",
            drift_kind="input-drift",
        ),
        "prediction_drift": create_evidently_report(
            reference_predictions,
            current_predictions,
            column=PREDICTION_DRIFT_COLUMN,
            name="Prediction drift: monthly revenue forecasts",
            drift_kind="prediction-drift",
        ),
    }
    for key, (html_name, json_name) in REPORT_FILENAMES.items():
        snapshots[key].save_html(str(output_dir / html_name))
        snapshots[key].save_json(str(output_dir / json_name))

    summary: dict[str, object] = {
        "dataset_kind": "deterministic synthetic serving traffic",
        "seed": DEFAULT_SEED,
        "reference_rows": len(reference_requests),
        "current_rows": len(current_requests),
        "reference_prediction_rows": len(reference_predictions),
        "current_prediction_rows": len(current_predictions),
        "drift_method": "population_stability_index",
        "drift_threshold": DRIFT_THRESHOLD,
        "horizon_months_psi": _drift_score(
            snapshots["input_drift"], INPUT_DRIFT_COLUMN
        ),
        "prediction_psi": _drift_score(
            snapshots["prediction_drift"], PREDICTION_DRIFT_COLUMN
        ),
        "model": {
            "name": service.config.model_name,
            "version": service.model_version,
            "run_id": service.run_id,
        },
    }
    for key in ("horizon_months_psi", "prediction_psi"):
        value = float(summary[key])
        if not math.isfinite(value):
            raise RuntimeError(f"Monitoring produced a non-finite {key} score")

    (output_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_findings(summary, output_dir / "findings.md")
    return summary


def main() -> None:
    """Run from a direct file path with actionable model-setup errors."""

    import sys

    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from app.model_service import ModelLoadError

    try:
        result = run()
    except ModelLoadError as error:
        raise SystemExit(
            f"{error}\n\n"
            "This report needs a compatible registered model. Either train one locally:\n"
            "    uv run python models/train.py\n"
            "or point at the running Compose registry (adjust the host port if overridden):\n"
            "    MLFLOW_TRACKING_URI=http://localhost:5000 "
            "uv run python monitoring/run_report.py"
        ) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
