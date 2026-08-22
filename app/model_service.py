"""Environment-driven loading and inference for the MLflow forecast model."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "revenue-forecast-bundle"
DEFAULT_MODEL_VERSION = "champion"
MAX_HORIZON_MONTHS = 24
SERVING_CONTRACT_TAG = "forecast-intervals-v1"


class ModelLoadError(RuntimeError):
    """Raised when the configured registry model cannot be loaded."""


class ForecastContractError(RuntimeError):
    """Raised when a loaded model does not implement the serving contract."""


@dataclass(frozen=True)
class ModelConfig:
    tracking_uri: str
    model_name: str
    model_version: str

    @classmethod
    def from_environment(cls) -> ModelConfig:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if not tracking_uri:
            repo_root = Path(__file__).resolve().parent.parent
            database = repo_root / "mlflow" / "mlflow.db"
            tracking_uri = f"sqlite:///{database}"

        return cls(
            tracking_uri=tracking_uri,
            model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()
            or DEFAULT_MODEL_NAME,
            model_version=os.getenv("MODEL_VERSION", DEFAULT_MODEL_VERSION).strip()
            or DEFAULT_MODEL_VERSION,
        )


class ModelService:
    """Load one concrete registry version once and reuse it for requests."""

    def __init__(
        self,
        *,
        config: ModelConfig,
        concrete_version: str,
        run_id: str,
        model: Any,
    ) -> None:
        self.config = config
        self.model_version = concrete_version
        self.run_id = run_id
        self._model = model
        self._predict_lock = threading.Lock()

    @classmethod
    def load(cls, config: ModelConfig | None = None) -> ModelService:
        config = config or ModelConfig.from_environment()
        mlflow.set_tracking_uri(config.tracking_uri)
        client = MlflowClient(
            tracking_uri=config.tracking_uri,
            registry_uri=config.tracking_uri,
        )

        try:
            concrete_version = cls._resolve_version(client, config)
            version = client.get_model_version(config.model_name, concrete_version)
            contract = version.tags.get("serving_contract")
            if contract != SERVING_CONTRACT_TAG:
                raise ModelLoadError(
                    f"{config.model_name} v{concrete_version} does not provide "
                    f"the required {SERVING_CONTRACT_TAG!r} contract. "
                    "Run the final-project training pipeline and select its version."
                )
            model_uri = f"models:/{config.model_name}/{concrete_version}"
            model = mlflow.pyfunc.load_model(model_uri)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Unable to load {config.model_name!r} version "
                f"{config.model_version!r} from {config.tracking_uri!r}."
            ) from exc

        return cls(
            config=config,
            concrete_version=concrete_version,
            run_id=version.run_id or "unknown",
            model=model,
        )

    @staticmethod
    def _resolve_version(client: MlflowClient, config: ModelConfig) -> str:
        requested = config.model_version
        if requested.isdigit():
            if int(requested) < 1:
                raise ModelLoadError("MODEL_VERSION must be a positive integer.")
            return requested

        if requested.lower() != "latest":
            try:
                aliased = client.get_model_version_by_alias(config.model_name, requested)
            except Exception as exc:
                raise ModelLoadError(
                    f"No {config.model_name!r} model version is assigned to alias "
                    f"{requested!r}. Register one first: uv run python models/train.py"
                ) from exc
            return str(aliased.version)

        versions = client.search_model_versions(f"name='{config.model_name}'")
        compatible = [
            version
            for version in versions
            if version.tags.get("serving_contract") == SERVING_CONTRACT_TAG
        ]
        if not compatible:
            raise ModelLoadError(
                f"No {config.model_name!r} version has the required "
                f"{SERVING_CONTRACT_TAG!r} contract. "
                "Register one first: uv run python models/train.py"
            )
        return str(max(int(version.version) for version in compatible))

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "model_name": self.config.model_name,
            "model_version": self.model_version,
            "model_run_id": self.run_id,
        }

    def forecast(self, horizon_months: int, interval_level: int) -> list[dict[str, Any]]:
        if not 1 <= horizon_months <= MAX_HORIZON_MONTHS:
            raise ForecastContractError(
                f"horizon_months must be between 1 and {MAX_HORIZON_MONTHS}."
            )
        if interval_level not in {80, 95}:
            raise ForecastContractError("interval_level must be either 80 or 95.")

        with self._predict_lock:
            output = self._model.predict(pd.DataFrame({"h": [horizon_months]}))

        if not isinstance(output, pd.DataFrame):
            output = pd.DataFrame(output)

        lower_column = f"Total_lo_{interval_level}"
        upper_column = f"Total_hi_{interval_level}"
        required = {
            "month",
            "IV_League",
            "MPD_Core_MRR",
            "MPD_Core_OneTime",
            "Total_Base",
            lower_column,
            upper_column,
        }
        missing = sorted(required - set(output.columns))
        if missing:
            raise ForecastContractError(
                "The registered model output is missing required confidence fields: "
                + ", ".join(missing)
            )
        if len(output) != horizon_months:
            raise ForecastContractError(
                f"The model returned {len(output)} rows for a {horizon_months}-month request."
            )

        forecasts: list[dict[str, Any]] = []
        for _, row in output.iterrows():
            values = {
                "iv_league": float(row["IV_League"]),
                "mpd_core_mrr": float(row["MPD_Core_MRR"]),
                "mpd_core_one_time": float(row["MPD_Core_OneTime"]),
                "total_prediction": float(row["Total_Base"]),
                "lower_bound": float(row[lower_column]),
                "upper_bound": float(row[upper_column]),
            }
            if not all(math.isfinite(value) and value >= 0 for value in values.values()):
                raise ForecastContractError("The model returned a non-finite or negative forecast.")
            if not values["lower_bound"] <= values["upper_bound"]:
                raise ForecastContractError("The model returned an invalid prediction interval.")

            forecasts.append(
                {
                    "month": pd.Timestamp(row["month"]).date(),
                    **values,
                    "confidence_level": interval_level,
                }
            )

        return forecasts
