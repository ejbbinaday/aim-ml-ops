# Final Project submission checklist

## Repository evidence

- [x] Work is isolated on the `final-project` branch.
- [x] Strict request/response contract is documented in `docs/SERVING_CONTRACT.md`.
- [x] FastAPI implements `/health`, `/predict`, startup model loading, and 422 validation.
- [x] MLflow model versions carry the serving-contract tag and exact run identity.
- [x] Dockerfile, Compose services, MLflow/API health checks, trainer gate, and `.env.example` exist.
- [x] API and monitoring tests are included.
- [x] CI runs lint plus tests with a 60% coverage gate.
- [x] Separate Evidently input/prediction HTML/JSON reports, PII-free datasets,
      summary, and findings are included.
- [x] README contains local, Docker, API, monitoring, and test commands.
- [x] Demo sequence is documented in `docs/DEMO_RUNBOOK.md`.

## Final laptop checks

Run these from the repository root with normal internet access:

```bash
uv lock
uv sync --locked
uv run ruff check .
uv run pytest --cov --cov-report=term-missing
```

- [ ] Confirm `uv.lock` changes and `uv lock --check` passes.
- [ ] Confirm all tests pass and coverage remains at least 60%.
- [ ] Open Docker Desktop and run `docker compose up --build mlflow trainer api`.
- [ ] Confirm `curl http://localhost:8000/health` returns HTTP 200 and model identity.
- [ ] Confirm a valid `/predict` request returns the requested monthly forecasts.
- [ ] Confirm an invalid request such as `{"horizon_months":0}` returns HTTP 422.
- [ ] Open MLflow at <http://localhost:5000> and verify the served version and run.
- [ ] Open `reports/evidently_report.html` and verify the request-input result.
- [ ] Open `reports/evidently_prediction_drift.html` and verify the monthly-output result.
- [ ] Stop the stack with `docker compose down`.

## Git and submission checks

Review changes before staging; do not use a blanket stage command without
checking the file list.

```bash
git status --short
git diff --check
git diff --stat
git diff
```

- [ ] Confirm there are no secrets, real customer data, local MLflow databases,
      `.env`, `.venv`, or generated pipeline outputs in the change set.
- [ ] Commit the reviewed files on `final-project`.
- [ ] Push the branch and open a pull request into `main`.
- [ ] Wait for the GitHub Actions check to turn green.
- [ ] Obtain teammate approval and merge the pull request.
- [ ] Tag the agreed final commit only after the merge, if the professor requires a tag.
- [ ] Submit the repository/PR URL, required screenshots or recording, and any
      separate LMS files named exactly as the professor requested.
