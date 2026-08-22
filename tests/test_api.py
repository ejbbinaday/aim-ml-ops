"""Contract tests for the FastAPI forecast service."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model_service import ForecastContractError, ModelConfig, ModelLoadError, ModelService


class StubModelService:
    """Small deterministic stand-in that keeps API tests independent of MLflow."""

    config = ModelConfig(
        tracking_uri="sqlite:///test.db",
        model_name="revenue-forecast-bundle",
        model_version="2",
    )
    model_version = "2"
    run_id = "test-run-id"

    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "model_name": self.config.model_name,
            "model_version": self.model_version,
            "model_run_id": self.run_id,
        }

    def forecast(self, horizon_months: int, interval_level: int):
        return [
            {
                "month": date(2026, month, 1),
                "iv_league": 10_000.0 + month,
                "mpd_core_mrr": 20_000.0,
                "mpd_core_one_time": 5_000.0,
                "total_prediction": 35_000.0 + month,
                "lower_bound": 30_000.0,
                "upper_bound": 40_000.0,
                "confidence_level": interval_level,
            }
            for month in range(1, horizon_months + 1)
        ]


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(service_loader=StubModelService)) as test_client:
        yield test_client


def test_health_reports_loaded_model_identity(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "revenue-forecast-bundle",
        "model_version": "2",
        "model_run_id": "test-run-id",
    }


@pytest.mark.parametrize("interval_level", [80, 95])
def test_predict_returns_one_versioned_forecast_per_month(
    client: TestClient, interval_level: int
):
    response = client.post(
        "/predict",
        json={"horizon_months": 3, "interval_level": interval_level},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "revenue-forecast-bundle"
    assert body["model_version"] == "2"
    assert body["model_run_id"] == "test-run-id"
    assert body["horizon_months"] == 3
    assert body["currency"] == "AUD"
    assert len(body["forecasts"]) == 3
    assert body["forecasts"][0] == {
        "month": "2026-01-01",
        "iv_league": 10001.0,
        "mpd_core_mrr": 20000.0,
        "mpd_core_one_time": 5000.0,
        "total_prediction": 35001.0,
        "lower_bound": 30000.0,
        "upper_bound": 40000.0,
        "confidence_level": interval_level,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"horizon_months": 0},
        {"horizon_months": 25},
        {"horizon_months": 12, "interval_level": 90},
        {"horizon_months": 12, "unexpected": True},
    ],
)
def test_predict_rejects_invalid_contract_payloads(client: TestClient, payload: dict):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        '{"horizon_months":NaN}',
        '{"horizon_months":Infinity}',
        '{"horizon_months":-Infinity}',
        '{"horizon_months":2,"interval_level":NaN}',
    ],
)
def test_predict_returns_serializable_422_for_non_finite_numbers(
    client: TestClient, body: str
):
    response = client.post(
        "/predict",
        content=body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"]


def test_application_warms_forecast_engine_before_reporting_ready():
    service = StubModelService()
    calls: list[tuple[int, int]] = []
    original_forecast = service.forecast

    def tracked_forecast(horizon_months: int, interval_level: int):
        calls.append((horizon_months, interval_level))
        return original_forecast(horizon_months, interval_level)

    service.forecast = tracked_forecast  # type: ignore[method-assign]
    with TestClient(create_app(service_loader=lambda: service)) as test_client:
        assert test_client.get("/health").status_code == 200

    assert calls == [(1, 80)]


def test_application_fails_fast_when_model_cannot_load():
    def failed_loader():
        raise RuntimeError("registry unavailable")

    with pytest.raises(RuntimeError, match="Forecast model startup failed"):
        with TestClient(create_app(service_loader=failed_loader)):
            pass


class MissingIntervalModel:
    def predict(self, request: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "month": ["2026-01-01"],
                "IV_League": [10_000.0],
                "MPD_Core_MRR": [20_000.0],
                "MPD_Core_OneTime": [5_000.0],
                "Total_Base": [35_000.0],
            }
        )


def test_model_service_rejects_incompatible_model_output():
    service = ModelService(
        config=StubModelService.config,
        concrete_version="2",
        run_id="test-run-id",
        model=MissingIntervalModel(),
    )

    with pytest.raises(ForecastContractError, match="missing required confidence fields"):
        service.forecast(horizon_months=1, interval_level=80)


def test_model_service_resolves_configured_registry_alias():
    class AliasClient:
        def get_model_version_by_alias(self, name: str, alias: str):
            assert name == "revenue-forecast-bundle"
            assert alias == "champion"
            return SimpleNamespace(version="7")

    config = ModelConfig(
        tracking_uri="sqlite:///test.db",
        model_name="revenue-forecast-bundle",
        model_version="champion",
    )

    assert ModelService._resolve_version(AliasClient(), config) == "7"  # type: ignore[arg-type]


def test_model_service_reports_missing_registry_alias_actionably():
    class MissingAliasClient:
        def get_model_version_by_alias(self, _name: str, _alias: str):
            raise KeyError("missing")

    config = ModelConfig(
        tracking_uri="sqlite:///test.db",
        model_name="revenue-forecast-bundle",
        model_version="champion",
    )

    with pytest.raises(ModelLoadError, match="Register one first"):
        ModelService._resolve_version(MissingAliasClient(), config)  # type: ignore[arg-type]
