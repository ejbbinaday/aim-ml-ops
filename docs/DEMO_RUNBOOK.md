# Final Project demo runbook

Target length: about 2–3 minutes. Run the preflight once before recording so
Docker does not spend the demo downloading or building images.

## Preflight (before recording)

```bash
uv sync
uv run ruff check .
uv run pytest --cov --cov-report=term-missing
docker compose up --build mlflow trainer api
```

Wait until the API log says `Application startup complete`. Keep the Compose
terminal open, then prepare these browser tabs:

1. <http://localhost:8000/docs>
2. <http://localhost:5000> (MLflow)
3. `reports/evidently_report.html` opened in the browser

## Recording sequence

### 1. Introduce the system (15 seconds)

“This project operationalizes our revenue forecast from data pipeline and
tracked training through versioned serving, automated testing, CI, and drift
monitoring.”

### 2. Prove readiness and model identity (20 seconds)

```bash
curl http://localhost:8000/health
```

Point out `status: ok`, `revenue-forecast-bundle`, the concrete model version,
and the MLflow run ID. Explain that the service loaded this model once at
startup rather than reloading it for every request.

### 3. Make a forecast (35 seconds)

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"horizon_months":12,"interval_level":80}'
```

Point out the 12 monthly rows, three component revenue streams, total forecast,
80% lower/upper bounds, currency, model version, and run ID. Say that 80% is
prediction-interval coverage, not the probability that the point estimate is
correct.

### 4. Show model governance (25 seconds)

In MLflow, open **Models → revenue-forecast-bundle → latest compatible
version**. Show the `serving_contract=forecast-intervals-v1` and
`final_project=true` tags, run link, parameters, metrics, and artifacts.

### 5. Show automated quality controls (20 seconds)

Show the green GitHub Actions run. Mention 42 local tests across validation,
model quality, integration, API behavior, storage, and monitoring, plus the
60% CI coverage gate (the verified local result is about 80%).

### 6. Show monitoring and responsible interpretation (35 seconds)

Open `reports/evidently_report.html`. Show that both `horizon_months` input
drift and cumulative `prediction` drift are detected. State that the datasets
are deterministic, synthetic, and PII-free and intentionally demonstrate a
request-mix shift; drift is an investigation signal, not proof of bad accuracy
and not an automatic retraining trigger.

### 7. Close (10 seconds)

“The repository is reproducible from the lockfile and Compose configuration,
and the README plus serving contract document the exact operating procedure.”

## After recording

```bash
docker compose down
```

Confirm the recording clearly shows successful API responses, the registered
model identity, a green CI run, and the Evidently results.
