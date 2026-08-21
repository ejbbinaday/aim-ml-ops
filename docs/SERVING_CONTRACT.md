# Revenue Forecast API Contract

The service exposes the registered `revenue-forecast-bundle` model without
retraining it during a request. The model is loaded once when FastAPI starts.

## `GET /health`

Successful response (`200`):

```json
{
  "status": "ok",
  "model_name": "revenue-forecast-bundle",
  "model_version": "2",
  "model_run_id": "c404df276c8946cd886502763d264815"
}
```

The version and run ID come from MLflow. They are not release labels invented
by the API.

## `POST /predict`

Request:

```json
{
  "horizon_months": 2,
  "interval_level": 80
}
```

- `horizon_months` is required and must be an integer from 1 through 24.
- `interval_level` is optional and must be either `80` or `95`; it defaults to
  `80`.
- Additional request fields are rejected.

Successful response (`200`):

```json
{
  "model_name": "revenue-forecast-bundle",
  "model_version": "2",
  "model_run_id": "c404df276c8946cd886502763d264815",
  "horizon_months": 2,
  "currency": "AUD",
  "forecasts": [
    {
      "month": "2026-07-01",
      "iv_league": 2579.02,
      "mpd_core_mrr": 87769.83,
      "mpd_core_one_time": 9193.17,
      "total_prediction": 99542.02,
      "lower_bound": 84553.66,
      "upper_bound": 116125.23,
      "confidence_level": 80
    },
    {
      "month": "2026-08-01",
      "iv_league": 3512.36,
      "mpd_core_mrr": 89025.11,
      "mpd_core_one_time": 8893.14,
      "total_prediction": 101430.61,
      "lower_bound": 85334.53,
      "upper_bound": 117792.06,
      "confidence_level": 80
    }
  ]
}
```

### What “confidence” means

The API does not return a classification probability. `lower_bound` and
`upper_bound` are the model's prediction interval at the requested coverage
level. The interval combines the component-series bounds using the same
bottom-up aggregation used by the forecasting pipeline. Wider intervals mean
greater forecast uncertainty.

Prediction intervals describe uncertainty under the fitted model and observed
history. They do not guarantee that a future value will fall within the bounds,
especially after material business or data-generating changes.

## Error behavior

- Invalid, missing, or additional fields: HTTP `422`, generated from the
  Pydantic schema.
- Model not ready: HTTP `503`.
- Registered model does not satisfy the interval contract: HTTP `500` with a
  specific contract error.
- Unexpected prediction failure: HTTP `500` with details written to service
  logs, without exposing an internal stack trace to the caller.

## Runtime configuration

The service reads:

- `MLFLOW_TRACKING_URI`
- `MODEL_NAME` (default: `revenue-forecast-bundle`)
- `MODEL_VERSION` (positive integer or `latest`)

When `MODEL_VERSION=latest`, the service selects the highest registry version
tagged with `serving_contract=forecast-intervals-v1`, then reports the concrete
version in `/health` and `/predict`.
