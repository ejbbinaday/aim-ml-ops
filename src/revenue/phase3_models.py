"""
Phase 3 – Model Training
========================
Fits the model suite prescribed by the methodology for each revenue series.

  IV_League         → EAT Ensemble  (AutoARIMA + AutoETS + Theta, equal weights)
                       Phase 4 compares EAT vs Croston and may downgrade to Croston
                       if the sparse-demand method wins on MASE.
  MPD_Core_MRR      → EAT Ensemble  (AutoARIMA + AutoETS + Theta, equal weights)
                       Switched from BSTS local-level: backtest showed MASE 2.38 vs
                       baseline 2.06; Ljung-Box p=0 confirmed residual autocorrelation.
  MPD_Core_OneTime  → Level Model   (mean-reverting flat forecast; Croston fallback if sparse)

Baseline models (Mean, Seasonal Naïve, Drift) are fit for all series.

Returns a ModelBundle NamedTuple consumed by Phases 4 and 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Theta
from statsmodels.tsa.statespace.structural import UnobservedComponents

from src import storage

from . import phase1_data
from .config import SEASON

BUNDLE_KEY = "tables/03_bundle.joblib"


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class SeriesModels:
    """All fitted models for one revenue series."""

    name: str
    train: pd.Series  # raw (AUD) training series
    train_log: pd.Series | None  # log1p-transformed series (legacy, unused)
    sf: StatsForecast | None  # fitted statsforecast object (EAT)
    sf_df: pd.DataFrame | None  # training DataFrame in sf format
    bsts_result: Any | None  # legacy field, always None
    exog_train: np.ndarray | None  # legacy field, always None
    outlier_dummies: pd.DataFrame  # binary intervention columns (one per outlier)
    model_type: str  # 'EAT' | 'Croston' | 'Level'
    nonzero_frac: float = 1.0
    croston_rate: float | None = None  # pre-computed Croston rate for Phase 4 comparison
    break_date: pd.Timestamp | None = None  # structural break — step dummy injected at this date
    model_info: dict = field(default_factory=dict)  # AICc, selected order, etc.


@dataclass
class ModelBundle:
    iv_league: SeriesModels
    mpd_core_mrr: SeriesModels
    mpd_core_onetime: SeriesModels
    # SHA-256 of the input the bundle was fitted against, used by
    # `--phase 4` / `--phase 5` to reject a stale bundle (see design D6).
    input_sha256: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_sf_df(series: pd.Series, uid: str) -> pd.DataFrame:
    """Convert a monthly pd.Series to the (unique_id, ds, y) format statsforecast needs."""
    return pd.DataFrame(
        {
            "unique_id": uid,
            "ds": series.index,
            "y": series.values,
        }
    )


def _build_outlier_dummies(series: pd.Series, outlier_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """
    Create binary dummy columns (one per outlier month) aligned to the series index.
    These become exogenous regressors in the BSTS model.
    """
    dummies = pd.DataFrame(index=series.index)
    for dt in outlier_dates:
        col = f"outlier_{dt.strftime('%Y%m')}"
        dummies[col] = (dummies.index == dt).astype(float)
    return dummies


def _covid_dummy(series: pd.Series) -> np.ndarray:
    """Return the is_covid_startup flag as a 2-D numpy array for statsmodels exog."""
    from .config import COVID_END

    flag = (series.index < pd.Timestamp(COVID_END)).astype(float)
    return np.asarray(flag).reshape(-1, 1)


# ── Baseline forecasts (reference benchmarks) ─────────────────────────────────


def baseline_mean(train: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, train.mean())


def baseline_seasonal_naive(train: np.ndarray, h: int, season: int = SEASON) -> np.ndarray:
    out = np.empty(h)
    for i in range(h):
        out[i] = train[-(season - (i % season))]
    return out


def baseline_drift(train: np.ndarray, h: int) -> np.ndarray:
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


# ── Step dummy for structural breaks ─────────────────────────────────────────


def _add_break_dummy(df: pd.DataFrame, break_date: pd.Timestamp | None) -> pd.DataFrame:
    """Add a post_break step column (0 before break, 1 from break onwards)."""
    if break_date is None:
        return df
    df = df.copy()
    df["post_break"] = (df["ds"] >= pd.Timestamp(break_date)).astype(float)
    return df


def _extract_model_info(sf: StatsForecast) -> dict:
    """Extract AICc and selected model order from fitted statsforecast EAT ensemble."""
    info = {}
    try:
        fitted = sf.fitted_[0]  # shape: (n_models,) for one series
        for model_obj, model_cls in zip(fitted, sf.models, strict=False):
            name = type(model_cls).__name__
            m = getattr(model_obj, "model_", {})
            if not isinstance(m, dict):
                continue
            aicc = m.get("aicc")
            if aicc is not None:
                info[f"{name}_aicc"] = round(float(aicc), 1)
            # ARIMA order: arma vector is (p, q, P, Q, period, d, D)
            arma = m.get("arma")
            if arma is not None and len(arma) >= 7:
                p, q, P, Q, _, d, D = arma[:7]
                info["AutoARIMA_order"] = f"ARIMA({p},{d},{q})({P},{D},{Q})[12]"
            # ETS method string
            method = m.get("method")
            if method:
                info["AutoETS_method"] = method
    except Exception:
        pass
    return info


# ── EAT Ensemble ──────────────────────────────────────────────────────────────


def fit_eat_ensemble(
    series: pd.Series,
    uid: str,
    break_date: pd.Timestamp | None = None,
) -> tuple[StatsForecast, pd.DataFrame]:
    """
    Fit AutoARIMA + AutoETS + Theta ensemble via statsforecast.
    Uses AICc-selected model orders; equal-weight averaging at forecast time.
    If break_date is set, injects a post_break step dummy as exogenous regressor.
    """
    sf_df = _to_sf_df(series, uid)
    sf_df = _add_break_dummy(sf_df, break_date)
    sf = StatsForecast(
        models=[
            AutoARIMA(season_length=SEASON, approximation=False, nmodels=20),
            AutoETS(season_length=SEASON),
            Theta(season_length=SEASON),
        ],
        freq="MS",
        n_jobs=1,
    )
    sf.fit(sf_df)
    return sf, sf_df


# ── Croston fallback for sparse one-time series ───────────────────────────────


def _croston_fit(series: np.ndarray, alpha: float = 0.15) -> float:
    """
    Croston's method: separate smoothing of demand size and inter-demand interval.
    Returns the estimated demand rate (AUD/month).
    """
    nonzero_idx = np.where(series > 0)[0]
    if len(nonzero_idx) < 2:
        return float(series.mean())

    demand_sizes = series[nonzero_idx].astype(float)
    intervals = np.diff(np.concatenate([[0], nonzero_idx])).astype(float)

    a = demand_sizes[0]
    q = intervals[0]
    for d, iv in zip(demand_sizes[1:], intervals[1:], strict=False):
        a = alpha * d + (1 - alpha) * a
        q = alpha * iv + (1 - alpha) * q

    return a / q if q > 0 else a


def croston_forecast(series: pd.Series, h: int, alpha: float = 0.15) -> np.ndarray:
    return np.full(h, _croston_fit(series.values, alpha))


# ── BSTS (UnobservedComponents) ───────────────────────────────────────────────


def fit_bsts(
    series_log: pd.Series,
    exog: np.ndarray | None = None,
) -> Any:
    """
    Fit a Bayesian Structural Time Series via statsmodels UnobservedComponents:
      • Local linear trend  (time-varying level + slope)
      • Monthly seasonal    (12-period dummy seasonal)
      • Exogenous regressors (COVID dummy + outlier intervention dummies)

    Returns the fitted result object.
    """
    # "local level" (level-only, no time-varying slope) is more appropriate than
    # "local linear trend" for subscription revenue: the level can drift with new
    # patients but we don't assume a consistently compounding growth rate.
    # A free slope would extrapolate recent volatile months forward explosively.
    mod = UnobservedComponents(
        endog=series_log.values,
        level="local level",
        seasonal=SEASON,
        exog=exog,
    )

    try:
        result = mod.fit(
            disp=False,
            method="lbfgs",
            maxiter=500,
            optim_score="approx",
        )
    except Exception:
        mod = UnobservedComponents(
            endog=series_log.values,
            level="local level",
            exog=exog,
        )
        result = mod.fit(disp=False, method="lbfgs", maxiter=300)

    return result


# ── Per-series fitting logic ──────────────────────────────────────────────────


def _fit_iv_league(
    monthly: pd.DataFrame,
    outlier_dates: list[pd.Timestamp],
) -> SeriesModels:
    series = monthly["IV_League"]
    nonzero_frac = float((series > 0).mean())
    # Fit EAT; Phase 4 will compare against Croston and may switch model_type.
    print(
        f"  [IV_League] EAT Ensemble (auto-select vs Croston in Phase 4)  "
        f"non-zero={nonzero_frac:.0%}  n={len(series)}"
    )
    sf, sf_df = fit_eat_ensemble(series, "IV_League")
    croston_rate = _croston_fit(series.values)
    model_info = _extract_model_info(sf)
    return SeriesModels(
        name="IV_League",
        train=series,
        train_log=None,
        sf=sf,
        sf_df=sf_df,
        bsts_result=None,
        exog_train=None,
        outlier_dummies=_build_outlier_dummies(series, outlier_dates),
        model_type="EAT",
        nonzero_frac=nonzero_frac,
        croston_rate=croston_rate,
        model_info=model_info,
    )


def _fit_mpd_core_mrr(
    monthly: pd.DataFrame,
    outlier_dates: list[pd.Timestamp],
    break_date: pd.Timestamp | None = None,
) -> SeriesModels:
    series = monthly["MPD_Core_MRR"]
    break_note = f"  break dummy from {break_date.strftime('%b %Y')}" if break_date else ""
    print(
        f"  [MPD_Core_MRR] EAT Ensemble (AutoARIMA + AutoETS + Theta)  "
        f"non-zero=100%  n={len(series)}{break_note}"
    )
    sf, sf_df = fit_eat_ensemble(series, "MPD_Core_MRR", break_date=break_date)
    model_info = _extract_model_info(sf)
    return SeriesModels(
        name="MPD_Core_MRR",
        train=series,
        train_log=None,
        sf=sf,
        sf_df=sf_df,
        bsts_result=None,
        exog_train=None,
        outlier_dummies=_build_outlier_dummies(series, outlier_dates),
        model_type="EAT",
        nonzero_frac=float((series > 0).mean()),
        break_date=break_date,
        model_info=model_info,
    )


def _fit_mpd_core_onetime(
    monthly: pd.DataFrame,
    outlier_dates: list[pd.Timestamp],
) -> SeriesModels:
    series = monthly["MPD_Core_OneTime"]
    nonzero_frac = float((series > 0).mean())

    if nonzero_frac >= 0.60:
        # Level model: One-Time revenue is mean-reverting and lumpy — no structural trend.
        # Projecting a stable smoothed level is more honest than trend extrapolation.
        print(
            f"  [MPD_Core_OneTime] Level Model (mean-reverting, no trend)  "
            f"non-zero={nonzero_frac:.0%}  n={len(series)}"
        )
        sf, sf_df = None, None
        bsts_result = None
        exog_train = None
        model_type = "Level"
    else:
        # Sparse → Croston (no statsforecast object needed; forecast in Phase 5)
        print(
            f"  [MPD_Core_OneTime] Croston (sparse demand)  "
            f"non-zero={nonzero_frac:.0%}  n={len(series)}"
        )
        sf, sf_df = None, None
        bsts_result = None
        exog_train = None
        model_type = "Croston"

    return SeriesModels(
        name="MPD_Core_OneTime",
        train=series,
        train_log=None,
        sf=sf,
        sf_df=sf_df,
        bsts_result=bsts_result,
        exog_train=exog_train,
        outlier_dummies=_build_outlier_dummies(series, outlier_dates),
        model_type=model_type,
        nonzero_frac=nonzero_frac,
    )


# ── Public entry point ────────────────────────────────────────────────────────


def run(
    monthly: pd.DataFrame,
    outlier_dates: dict[str, list[pd.Timestamp]],
    break_dates: dict[str, list[pd.Timestamp]] | None = None,
) -> ModelBundle:
    """
    Fit all models. Returns a ModelBundle for use by Phases 4 and 5.

    Parameters
    ----------
    monthly       : Monthly DataFrame from Phase 1.
    outlier_dates : Dict of {series_name: [outlier timestamps]} from Phase 2.
    break_dates   : Dict of {series_name: [break timestamps]} from Phase 2.
    """
    break_dates = break_dates or {}
    print("\n── Phase 3: Model Training ──────────────────────────────────────")

    mrr_breaks = break_dates.get("MPD_Core_MRR", [])
    mrr_break_dt = mrr_breaks[0] if mrr_breaks else None

    iv = _fit_iv_league(monthly, outlier_dates.get("IV_League", []))
    mrr = _fit_mpd_core_mrr(monthly, outlier_dates.get("MPD_Core_MRR", []), break_date=mrr_break_dt)
    ot = _fit_mpd_core_onetime(monthly, outlier_dates.get("MPD_Core_OneTime", []))

    # Save a brief model spec summary
    summary = pd.DataFrame(
        [
            {
                "series": iv.name,
                "model": iv.model_type,
                "n_train": len(iv.train),
                "nonzero_pct": f"{iv.nonzero_frac:.0%}",
            },
            {
                "series": mrr.name,
                "model": mrr.model_type,
                "n_train": len(mrr.train),
                "nonzero_pct": f"{mrr.nonzero_frac:.0%}",
            },
            {
                "series": ot.name,
                "model": ot.model_type,
                "n_train": len(ot.train),
                "nonzero_pct": f"{ot.nonzero_frac:.0%}",
            },
        ]
    )
    storage.save_csv("tables/03_model_spec.csv", summary, index=False)

    # Save AICc and model order info
    info_rows = []
    for sm in [iv, mrr, ot]:
        row = {"series": sm.name, "model": sm.model_type}
        row.update(sm.model_info)
        if sm.break_date:
            row["break_date"] = sm.break_date.strftime("%Y-%m")
        info_rows.append(row)
    storage.save_csv("tables/03_model_info.csv", pd.DataFrame(info_rows), index=False)
    print("  Model spec + AICc saved → 03_model_spec.csv, 03_model_info.csv")

    bundle = ModelBundle(
        iv_league=iv,
        mpd_core_mrr=mrr,
        mpd_core_onetime=ot,
        input_sha256=phase1_data.current_input_sha256(),
    )
    save_bundle(bundle)
    print(f"  Bundle persisted → {BUNDLE_KEY} (input_sha256={bundle.input_sha256[:12]}…)")
    return bundle


# ── Bundle persistence ────────────────────────────────────────────────────────


def save_bundle(bundle: ModelBundle, key: str = BUNDLE_KEY) -> None:
    """Serialize the fitted bundle to the outputs store via joblib."""
    buf = BytesIO()
    joblib.dump(bundle, buf)
    storage.save_bytes(key, buf.getvalue(), content_type="application/octet-stream")


def load_bundle(key: str = BUNDLE_KEY) -> ModelBundle:
    """Load a previously-persisted ModelBundle.

    The bundle's `input_sha256` should be compared against the current input's
    SHA before consuming — `--phase 4` / `--phase 5` reject stale bundles.
    """
    return joblib.load(BytesIO(storage.load_bytes(key)))
