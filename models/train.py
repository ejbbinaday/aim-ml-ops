"""Milestone 2 — MLflow-tracked model training & evaluation entry point.

Wraps the existing pipeline phases (1 data → 2 EDA → 3 fit → 4 CV evaluation)
in a single MLflow run under the ``revenue-forecast`` experiment:

  • params  — the modelling configuration from ``src/revenue/config.py``
              (seasonality, CV windows/step, horizons) plus input provenance.
  • metrics — per-series rolling-origin CV metrics from phase 4
              (MASE, RMSE, Winkler-80) and the seasonal-naïve baseline MASE.
  • model   — the fitted ``ModelBundle`` wrapped as an ``mlflow.pyfunc`` model
              and registered to the Model Registry as
              ``revenue-forecast-bundle`` with a version tag.

Tracking URI comes from the ``MLFLOW_TRACKING_URI`` environment variable —
never hardcoded. When unset, a serverless local SQLite store under ``mlflow/``
is used so a fresh clone can train and inspect runs without any server:

    uv run python models/train.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

Against the docker-compose server instead:

    docker compose up mlflow
    MLFLOW_TRACKING_URI=http://localhost:5000 uv run python models/train.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.revenue import phase1_data, phase2_eda, phase3_models, phase4_eval
from src.revenue.config import (
    CV_STEP,
    CV_WINDOWS,
    FORECAST_LONG,
    FORECAST_SHORT,
    MIN_TRAIN_MONTHS,
    SEASON,
)
from src.revenue.phase3_models import ModelBundle
from src.revenue.pipeline_m1 import make_run_id

EXPERIMENT_NAME = "revenue-forecast"
REGISTERED_MODEL_NAME = "revenue-forecast-bundle"
# Serverless fallback: SQLite backend under mlflow/ (registry-capable, unlike
# a plain file store). The real URI must come from MLFLOW_TRACKING_URI.
DEFAULT_TRACKING_URI = f"sqlite:///{_REPO_ROOT / 'mlflow' / 'mlflow.db'}"

# series name → ModelBundle attribute
SERIES_ATTRS = {
    "IV_League": "iv_league",
    "MPD_Core_MRR": "mpd_core_mrr",
    "MPD_Core_OneTime": "mpd_core_onetime",
}
SERIES = list(SERIES_ATTRS)


def resolve_tracking_uri() -> str:
    """Tracking URI from the environment, with a local SQLite fallback."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    (_REPO_ROOT / "mlflow").mkdir(exist_ok=True)
    return DEFAULT_TRACKING_URI


def bundle_forecast(bundle: ModelBundle, h: int) -> pd.DataFrame:
    """Point forecasts for all three streams + Total over the next `h` months."""
    from src.revenue.phase5_forecast import _forecast_index, _forecast_series

    idx = _forecast_index(bundle.iv_league.train.index[-1], h)
    out = pd.DataFrame({"month": idx})
    for name, attr in SERIES_ATTRS.items():
        point, *_ = _forecast_series(getattr(bundle, attr), h)
        out[name] = point
    out["Total"] = out[SERIES].sum(axis=1)
    return out


class RevenueForecastModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper: input is a DataFrame with an integer ``h`` column
    (forecast horizon in months); output is the point-forecast frame from
    :func:`bundle_forecast`."""

    def load_context(self, context):
        self._bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        h = int(model_input["h"].iloc[0])
        return bundle_forecast(self._bundle, h)


def _log_registered_model(bundle: ModelBundle, run_id: str) -> None:
    """Log the fitted bundle as a pyfunc model and register a tagged version."""
    with tempfile.TemporaryDirectory() as td:
        bundle_path = Path(td) / "03_bundle.joblib"
        joblib.dump(bundle, bundle_path)

        example_in = pd.DataFrame({"h": [FORECAST_SHORT]})
        signature = infer_signature(example_in, bundle_forecast(bundle, 3))

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=RevenueForecastModel(),
            artifacts={"bundle": str(bundle_path)},
            signature=signature,
            input_example=example_in,
            # Ship src/ with the model so joblib can unpickle the bundle and
            # predict() can import the forecast dispatch outside this repo.
            code_paths=[str(_REPO_ROOT / "src")],
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    version = model_info.registered_model_version
    client = MlflowClient()
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "milestone", "2")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "pipeline_run_id", run_id)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version, "input_sha256", bundle.input_sha256[:12]
    )
    print(f"  MLflow: registered {REGISTERED_MODEL_NAME} v{version} (milestone=2)")


def train() -> dict[str, dict]:
    """Run phases 1–4 inside a tracked MLflow run. Returns the eval results."""
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_id = make_run_id()

    with mlflow.start_run(run_name=f"train-{run_id}"):
        # ── Phase 1: data ────────────────────────────────────────────────────
        monthly, meta = phase1_data.run()

        # Pipeline configuration — the knobs that shape what gets fit and how
        # it is cross-validated (≥3 params, all manual log_param calls).
        mlflow.log_params(
            {
                "season_length": SEASON,
                "cv_windows": CV_WINDOWS,
                "cv_step_months": CV_STEP,
                "min_train_months": MIN_TRAIN_MONTHS,
                "forecast_horizon_short": FORECAST_SHORT,
                "forecast_horizon_long": FORECAST_LONG,
                "input_source": meta["input_source"],
                "train_months": len(monthly),
            }
        )

        # ── Phases 2–4: EDA → fit → rolling-origin CV ────────────────────────
        eda = phase2_eda.run(monthly)
        bundle = phase3_models.run(monthly, eda.outlier_dates, eda.break_dates)
        results = phase4_eval.run(bundle, monthly)

        # Per-series CV metrics (manual log_metric calls). MASE is the headline
        # accuracy metric; RMSE gives scale-aware error; Winkler-80 scores the
        # prediction intervals leadership actually plans against.
        for series, attr in SERIES_ATTRS.items():
            r = results.get(series, {})
            for metric in ("MASE", "RMSE", "Winkler", "Baseline_MASE"):
                value = r.get(metric)
                if value is not None and pd.notna(value):
                    mlflow.log_metric(f"{series}_{metric}", float(value))
            # Phase 4 may flip the model type per the CV winner — log the final choice.
            mlflow.log_param(f"{series}_model_type", getattr(bundle, attr).model_type)

        # Evaluation table + diagnostics figures as run artifacts.
        for artifact in (
            "outputs/tables/04_evaluation_results.csv",
            "outputs/figures/04_cv_performance.png",
            "outputs/figures/04_residuals.png",
        ):
            path = _REPO_ROOT / artifact
            if path.exists():
                mlflow.log_artifact(str(path))

        # ── Model Registry ───────────────────────────────────────────────────
        _log_registered_model(bundle, run_id)

    print(f"  MLflow: run logged to {tracking_uri} (experiment '{EXPERIMENT_NAME}')")
    return results


if __name__ == "__main__":
    train()
