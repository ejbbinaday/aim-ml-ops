"""Public request and response contracts for the forecast API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """A request for a future monthly revenue forecast."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "horizon_months": 12,
                "interval_level": 80,
            }
        },
    )

    horizon_months: int = Field(
        ge=1,
        le=24,
        description="Number of future calendar months to forecast.",
    )
    interval_level: Literal[80, 95] = Field(
        default=80,
        description="Prediction-interval coverage to return.",
    )


class ForecastPoint(BaseModel):
    """One monthly forecast with an uncertainty interval in AUD."""

    month: date
    iv_league: float = Field(ge=0)
    mpd_core_mrr: float = Field(ge=0)
    mpd_core_one_time: float = Field(ge=0)
    total_prediction: float = Field(ge=0)
    lower_bound: float = Field(ge=0)
    upper_bound: float = Field(ge=0)
    confidence_level: Literal[80, 95]


class PredictionResponse(BaseModel):
    """Versioned forecast response returned by ``POST /predict``."""

    model_name: str
    model_version: str
    model_run_id: str
    horizon_months: int
    currency: Literal["AUD"] = "AUD"
    forecasts: list[ForecastPoint]


class HealthResponse(BaseModel):
    """Readiness response returned by ``GET /health``."""

    status: Literal["ok"] = "ok"
    model_name: str
    model_version: str
    model_run_id: str
