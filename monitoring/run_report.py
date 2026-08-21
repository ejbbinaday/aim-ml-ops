"""Generate PII-free reference/current data and an Evidently drift report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from app.model_service import ModelService

if TYPE_CHECKING:
    from evidently.core.report import Snapshot

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MONITORED_COLUMNS = ["horizon_months", "prediction"]
HORIZONS = np.array([3, 6, 12, 24])
REFERENCE_WEIGHTS = np.array([0.55, 0.30, 0.10, 0.05])
CURRENT_WEIGHTS = np.array([0.05, 0.10, 0.30, 0.55])
DRIFT_THRESHOLD = 0.2
DEFAULT_SAMPLE_SIZE = 400
DEFAULT_SEED = 20260821


def _prediction_by_horizon(service: ModelService) -> dict[int, float]:
    """Map each monitored request horizon to its cumulative point forecast."""

    return {
        int(horizon): round(
            sum(
                point["total_prediction"]
                for point in service.forecast(int(horizon), interval_level=80)
            ),
            2,
        )
        for horizon in HORIZONS
    }


def build_serving_datasets(
    service: ModelService,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic request/output samples without personal information.

    These are labelled demonstration datasets. They model a shift from mostly
    short-horizon requests to mostly long-horizon requests; they are not
    presented as captured production traffic.
    """

    if sample_size < 50:
        raise ValueError("sample_size must be at least 50 for a meaningful drift example")

    prediction_by_horizon = _prediction_by_horizon(service)
    rng = np.random.default_rng(seed)

    def sample(weights: np.ndarray) -> pd.DataFrame:
        horizons = rng.choice(HORIZONS, size=sample_size, p=weights)
        return pd.DataFrame(
            {
                "horizon_months": horizons.astype(int),
                "prediction": [prediction_by_horizon[int(value)] for value in horizons],
            }
        )

    return sample(REFERENCE_WEIGHTS), sample(CURRENT_WEIGHTS)


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
) -> Snapshot:
    """Run an explicit PSI-based drift report for API input and prediction."""

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(
        [
            DataDriftPreset(
                columns=MONITORED_COLUMNS,
                method="psi",
                threshold=DRIFT_THRESHOLD,
            )
        ],
        metadata={
            "dataset_kind": "deterministic synthetic serving traffic",
            "purpose": "Final Project monitoring demonstration",
        },
        tags=["final-project", "drift-monitoring", "pii-free"],
    )
    return report.run(
        current_data=current,
        reference_data=reference,
        name="Revenue forecast serving drift",
    )


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
longer requests produce larger cumulative revenue forecasts, the output distribution shifted too.
This is a monitoring signal to investigate, **not proof that the model has become inaccurate**.

## Results

| Signal | PSI | Threshold | Result |
|---|---:|---:|---|
| API input: `horizon_months` | {input_score:.3f} | {DRIFT_THRESHOLD:.1f} | Drift {input_state} |
| Model output: cumulative `prediction` | {prediction_score:.3f} | {DRIFT_THRESHOLD:.1f} | Drift {prediction_state} |

The report used {summary['reference_rows']} reference requests and {summary['current_rows']}
current requests. It evaluated registered model `{model['name']}` Version `{model['version']}`
(run `{model['run_id']}`).

## Interpretation and action

The input shift may mean users are planning farther ahead, the API's calling pattern changed, or
the serving mix changed. The prediction shift is expected to move with that input because the
monitored output is the cumulative forecast over the requested horizon. The team should first
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

    output_dir.mkdir(parents=True, exist_ok=True)
    service = ModelService.load()
    reference, current = build_serving_datasets(service)
    reference.to_csv(output_dir / "reference_serving_data.csv", index=False)
    current.to_csv(output_dir / "current_serving_data.csv", index=False)

    snapshot = create_evidently_report(reference, current)
    snapshot.save_html(str(output_dir / "evidently_report.html"))
    snapshot.save_json(str(output_dir / "evidently_report.json"))

    summary: dict[str, object] = {
        "dataset_kind": "deterministic synthetic serving traffic",
        "seed": DEFAULT_SEED,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_method": "population_stability_index",
        "drift_threshold": DRIFT_THRESHOLD,
        "horizon_months_psi": population_stability_index(
            reference["horizon_months"], current["horizon_months"]
        ),
        "prediction_psi": population_stability_index(
            reference["prediction"], current["prediction"]
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


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
