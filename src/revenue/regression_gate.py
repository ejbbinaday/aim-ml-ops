"""Forecast-quality regression gate.

Compares each series' new MASE against the prior manifest's MASE for the
same series. Fires (non-zero exit) when **both** of these are true for the
same series:

  - new_MASE > ratio × prior_MASE
  - new_MASE > seasonal-naive baseline MASE (from `04_evaluation_results.csv`)

A series that worsens but still beats the dumb benchmark gets a WARN log,
not a hard fail. A first run with no prior manifest skips the gate entirely.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_RATIO = 1.30
ENV_VAR = "MPD_REVENUE_MASE_REGRESSION_RATIO"


class RegressionError(RuntimeError):
    """Raised when one or more series fail the regression gate."""


@dataclass
class Regression:
    series: str
    prior_mase: float
    current_mase: float
    baseline_mase: float
    ratio: float


def _ratio_from_env() -> float:
    raw = os.getenv(ENV_VAR)
    if not raw:
        return DEFAULT_RATIO
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_RATIO


def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def check_regression(
    eval_df: pd.DataFrame,
    prior_manifest: dict[str, Any] | None,
    ratio: float | None = None,
) -> list[Regression]:
    """Evaluate the gate. Raises ``RegressionError`` on hard regressions.

    Returns the list of regressions identified (hard fails go through the
    raise; soft regressions are logged-and-returned for the caller's
    awareness if it wants to summarize).
    """
    if prior_manifest is None or not prior_manifest.get("metrics"):
        print("  Regression gate skipped: no prior manifest")
        return []

    effective_ratio = ratio if ratio is not None else _ratio_from_env()
    prior_metrics = prior_manifest["metrics"]

    hard_regressions: list[Regression] = []
    soft_regressions: list[Regression] = []

    for _, row in eval_df.iterrows():
        series = row["series"]
        current = float(row["MASE"]) if not _is_nan(row["MASE"]) else math.nan
        baseline = float(row.get("Baseline_MASE", math.nan))
        prior = prior_metrics.get(series, {}).get("MASE")
        if _is_nan(current) or prior is None or _is_nan(prior) or prior <= 0:
            continue

        observed_ratio = current / prior
        if observed_ratio <= effective_ratio:
            continue  # within tolerance

        reg = Regression(
            series=series,
            prior_mase=prior,
            current_mase=current,
            baseline_mase=baseline,
            ratio=observed_ratio,
        )

        # Double-condition: only block publication when the new MASE is also
        # worse than the seasonal-naive baseline. A model that worsens but
        # still beats the dumb benchmark is a quality drift to log, not a
        # publication-blocker.
        if not _is_nan(baseline) and current > baseline:
            hard_regressions.append(reg)
        else:
            soft_regressions.append(reg)

    for reg in soft_regressions:
        print(
            f"  WARN: {reg.series} MASE worsened "
            f"({reg.prior_mase:.3f} → {reg.current_mase:.3f}, "
            f"ratio {reg.ratio:.2f}× ≥ {effective_ratio:.2f}×) "
            f"but still beats baseline ({reg.baseline_mase:.3f}); proceeding"
        )

    if hard_regressions:
        detail = "; ".join(
            f"{r.series}: prior {r.prior_mase:.3f} → current {r.current_mase:.3f} "
            f"(ratio {r.ratio:.2f}×, baseline {r.baseline_mase:.3f})"
            for r in hard_regressions
        )
        raise RegressionError(
            f"forecast regression in {len(hard_regressions)} series "
            f"(threshold {effective_ratio:.2f}×): {detail}"
        )

    return soft_regressions
