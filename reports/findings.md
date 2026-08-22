# Drift-monitoring findings

## Executive summary

This reproducible demonstration detected both input drift and prediction drift. The simulated
current period contains far more long-horizon forecast requests than the reference period. Because
longer requests return more future monthly forecasts, the per-month output distribution shifted too.
This is a monitoring signal to investigate, **not proof that the model has become inaccurate**.

## Results

| Signal | PSI | Threshold | Result |
|---|---:|---:|---|
| API input: `horizon_months` | 2.723 | 0.2 | Drift detected |
| Model output: monthly `prediction` | 0.676 | 0.2 | Drift detected |

The report used 400 reference requests and 400
current requests, producing 2448 reference forecast rows and
6915 current forecast rows. It evaluated registered model
`revenue-forecast-bundle` Version `2` (run `c404df276c8946cd886502763d264815`).

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
