"""FastAPI application serving the registered revenue forecast model."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.model_service import ForecastContractError, ModelService
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

LOGGER = logging.getLogger(__name__)


def _json_safe(value: object) -> object:
    """Replace non-finite floats that strict JSON cannot represent."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)


def get_model_service(request: Request) -> ModelService:
    """Return the model loaded during application startup."""

    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast model is not ready.",
        )
    return service


ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]


def create_app(service_loader: Callable[[], ModelService] | None = None) -> FastAPI:
    """Application factory that keeps model startup explicit and testable."""

    loader = service_loader or ModelService.load

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            service = loader()
            # Pay the forecast engine's one-off initialization cost before the
            # service reports ready, so the first real caller does not.
            service.forecast(1, 80)
        except Exception as exc:
            raise RuntimeError(f"Forecast model startup failed: {exc}") from exc
        application.state.model_service = service
        yield
        application.state.model_service = None

    application = FastAPI(
        title="Revenue Forecast API",
        version="1.0.0",
        description=(
            "Serves versioned monthly revenue forecasts and calibrated prediction intervals "
            "from the MLflow Model Registry."
        ),
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Return a serializable 422 response for every invalid request body."""

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": _json_safe(exc.errors())},
        )

    @application.get("/health", response_model=HealthResponse)
    def health(service: ModelServiceDependency) -> dict[str, str]:
        return service.health()

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        request: PredictionRequest,
        service: ModelServiceDependency,
    ) -> dict[str, object]:
        try:
            forecasts = service.forecast(request.horizon_months, request.interval_level)
        except ForecastContractError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            LOGGER.exception("Unexpected forecast failure")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Forecast generation failed.",
            ) from exc

        return {
            "model_name": service.config.model_name,
            "model_version": service.model_version,
            "model_run_id": service.run_id,
            "horizon_months": request.horizon_months,
            "currency": "AUD",
            "forecasts": forecasts,
        }

    return application


app = create_app()
