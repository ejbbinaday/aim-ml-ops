"""
Phase 4 – Model Evaluation & Validation
=========================================
Evaluates each fitted model using rolling-origin (expanding-window) cross-validation
and residual diagnostics. Selects the winning model per series.

Metrics
-------
MASE   – Mean Absolute Scaled Error vs. seasonal naïve baseline (primary)
RMSE   – Root Mean Squared Error (secondary; penalises large misses)
Winkler– Average Winkler score at 80% level (interval quality)
Ljung-Box – Portmanteau test on model residuals (white-noise check)

Outputs
-------
outputs/tables/04_evaluation_results.csv
outputs/figures/04_cv_performance.png
outputs/figures/04_residuals.png
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

from src import storage

from . import manifest, phase1_data, phase3_models, regression_gate
from .config import (
    COLOURS,
    CV_STEP,
    CV_WINDOWS,
    MIN_TRAIN_MONTHS,
    SEASON,
)
from .phase3_models import (
    ModelBundle,
    SeriesModels,
    baseline_seasonal_naive,
    croston_forecast,
)

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ── Metric functions ──────────────────────────────────────────────────────────


def mase(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray) -> float:
    """
    MASE = MAE(forecast) / MAE(seasonal naïve on training set).
    Scale is the mean absolute lag-SEASON difference on the training series,
    i.e., the in-sample error of the seasonal naïve method.
    MASE < 1.0  →  model beats seasonal naïve.
    """
    if len(train) <= SEASON:
        return np.nan
    scale = np.mean(np.abs(train[SEASON:] - train[:-SEASON]))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(actual - forecast)) / scale)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def winkler_score(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.20) -> float:
    """
    Winkler score at (1-alpha) coverage level.
    Penalises: interval width + 2/alpha × shortfall if actual is outside the interval.
    Lower is better.
    """
    width = hi - lo
    below = actual < lo
    above = actual > hi
    penalty = np.where(
        below, (2 / alpha) * (lo - actual), np.where(above, (2 / alpha) * (actual - hi), 0.0)
    )
    return float(np.mean(width + penalty))


# ── Cross-validation helpers ──────────────────────────────────────────────────


def _cv_windows(
    n: int, h: int = SEASON, n_windows: int = CV_WINDOWS, step: int = CV_STEP
) -> list[tuple[int, int]]:
    """
    Return (train_end, test_end) index pairs for rolling-origin CV.
    The last window's test set ends at n.
    """
    last_train_end = n - h
    pairs = []
    for i in range(n_windows - 1, -1, -1):
        train_end = last_train_end - i * step
        test_end = train_end + h
        if train_end >= MIN_TRAIN_MONTHS and test_end <= n:
            pairs.append((train_end, test_end))
    return pairs


def _eat_cv(sm: SeriesModels, h: int = SEASON) -> dict:
    """Rolling-origin CV for the EAT Ensemble using statsforecast."""
    series = sm.train
    n = len(series)
    windows = _cv_windows(n, h)

    all_actual, all_fc_arima, all_fc_ets, all_fc_theta = [], [], [], []
    all_lo80, all_hi80 = [], []

    for train_end, test_end in windows:
        train_s = series.iloc[:train_end]
        actual = series.iloc[train_end:test_end].values

        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, Theta

        from .phase3_models import _add_break_dummy

        sf_tmp = StatsForecast(
            models=[
                AutoARIMA(season_length=SEASON, approximation=True),
                AutoETS(season_length=SEASON),
                Theta(season_length=SEASON),
            ],
            freq="MS",
            n_jobs=1,
        )
        train_df = pd.DataFrame(
            {
                "unique_id": sm.name,
                "ds": train_s.index,
                "y": train_s.values,
            }
        )
        train_df = _add_break_dummy(train_df, sm.break_date)
        sf_tmp.fit(train_df)

        if sm.break_date is not None:
            future_dates = pd.date_range(
                train_s.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS"
            )
            X_df = pd.DataFrame(
                {
                    "unique_id": sm.name,
                    "ds": future_dates,
                    "post_break": np.ones(h),
                }
            )
            fc = sf_tmp.predict(h=h, level=[80], X_df=X_df)
        else:
            fc = sf_tmp.predict(h=h, level=[80])

        fc_arima = fc["AutoARIMA"].values
        fc_ets = fc["AutoETS"].values
        fc_theta = fc["Theta"].values

        lo80 = (
            fc["AutoARIMA-lo-80"].values + fc["AutoETS-lo-80"].values + fc["Theta-lo-80"].values
        ) / 3.0
        hi80 = (
            fc["AutoARIMA-hi-80"].values + fc["AutoETS-hi-80"].values + fc["Theta-hi-80"].values
        ) / 3.0

        all_actual.append(actual)
        all_fc_arima.append(fc_arima)
        all_fc_ets.append(fc_ets)
        all_fc_theta.append(fc_theta)
        all_lo80.append(lo80)
        all_hi80.append(hi80)

    actual_all = np.concatenate(all_actual)
    ensemble_all = np.concatenate(
        [(a + b + c) / 3 for a, b, c in zip(all_fc_arima, all_fc_ets, all_fc_theta, strict=False)]
    )
    lo80_all = np.concatenate(all_lo80)
    hi80_all = np.concatenate(all_hi80)

    return {
        "MASE": mase(actual_all, ensemble_all, series.values),
        "RMSE": rmse(actual_all, ensemble_all),
        "Winkler": winkler_score(actual_all, lo80_all, hi80_all, alpha=0.20),
        "actual": actual_all,
        "forecast": ensemble_all,
        "residuals": actual_all - ensemble_all,
    }


def _bsts_cv(sm: SeriesModels, h: int = SEASON) -> dict:
    """Manual rolling-origin CV for the BSTS (UnobservedComponents) model."""
    series = sm.train
    series_log = sm.train_log
    exog_full = sm.exog_train
    n = len(series)
    windows = _cv_windows(n, h)

    all_actual, all_fc, all_lo80, all_hi80 = [], [], [], []

    for train_end, test_end in windows:
        train_log = series_log.iloc[:train_end]
        actual_raw = series.iloc[train_end:test_end].values
        exog_tr = exog_full[:train_end] if exog_full is not None else None
        exog_te = exog_full[train_end:test_end] if exog_full is not None else None

        res = phase3_models.fit_bsts(train_log, exog=exog_tr)
        fc_obj = res.get_forecast(steps=h, exog=exog_te)

        fc_log = np.asarray(fc_obj.predicted_mean)
        ci_log = fc_obj.conf_int(alpha=0.20)  # 80 % interval in log space

        fc_raw = np.expm1(fc_log)
        lo_raw = np.expm1(np.asarray(ci_log)[:, 0])
        hi_raw = np.expm1(np.asarray(ci_log)[:, 1])

        all_actual.append(actual_raw)
        all_fc.append(fc_raw)
        all_lo80.append(lo_raw)
        all_hi80.append(hi_raw)

    actual_all = np.concatenate(all_actual)
    forecast_all = np.concatenate(all_fc)
    lo80_all = np.concatenate(all_lo80)
    hi80_all = np.concatenate(all_hi80)

    return {
        "MASE": mase(actual_all, forecast_all, series.values),
        "RMSE": rmse(actual_all, forecast_all),
        "Winkler": winkler_score(actual_all, lo80_all, hi80_all, alpha=0.20),
        "actual": actual_all,
        "forecast": forecast_all,
    }


def _croston_cv(sm: SeriesModels, h: int = SEASON) -> dict:
    """Rolling-origin CV for the Croston model (constant-rate forecast)."""
    series = sm.train
    n = len(series)
    windows = _cv_windows(n, h)

    all_actual, all_fc = [], []
    for train_end, test_end in windows:
        train_s = series.iloc[:train_end]
        actual = series.iloc[train_end:test_end].values
        fc = croston_forecast(train_s, h)
        all_actual.append(actual)
        all_fc.append(fc)

    actual_all = np.concatenate(all_actual)
    forecast_all = np.concatenate(all_fc)

    return {
        "MASE": mase(actual_all, forecast_all, series.values),
        "RMSE": rmse(actual_all, forecast_all),
        "Winkler": np.nan,
        "actual": actual_all,
        "forecast": forecast_all,
        "residuals": actual_all - forecast_all,
    }


def _level_cv(sm: SeriesModels, h: int = SEASON) -> dict:
    """Rolling-origin CV for the Level model."""
    from .config import COVID_END

    series = sm.train
    n = len(series)
    windows = _cv_windows(n, h)
    alpha = 0.25

    all_actual, all_fc, all_lo80, all_hi80 = [], [], [], []

    for train_end, test_end in windows:
        train_s = series.iloc[:train_end]
        actual = series.iloc[train_end:test_end].values

        covid_mask = train_s.index < pd.Timestamp(COVID_END)
        clean = train_s[~covid_mask & (train_s > 0)]
        if len(clean) < 2:
            clean = train_s[train_s > 0]
        if len(clean) < 1:
            clean = train_s

        weights = np.array([(1 - alpha) ** i for i in range(len(clean) - 1, -1, -1)])
        weights /= weights.sum()
        level = float(max(np.dot(weights, clean.values), 0.0))
        sigma = float(np.std(clean.values - level)) if len(clean) > 1 else level * 0.5

        fc = np.full(h, level)
        lo80 = np.maximum(fc - 1.28 * sigma, 0.0)
        hi80 = fc + 1.28 * sigma

        all_actual.append(actual)
        all_fc.append(fc)
        all_lo80.append(lo80)
        all_hi80.append(hi80)

    actual_all = np.concatenate(all_actual)
    fc_all = np.concatenate(all_fc)
    lo80_all = np.concatenate(all_lo80)
    hi80_all = np.concatenate(all_hi80)

    return {
        "MASE": mase(actual_all, fc_all, series.values),
        "RMSE": rmse(actual_all, fc_all),
        "Winkler": winkler_score(actual_all, lo80_all, hi80_all, alpha=0.20),
        "actual": actual_all,
        "forecast": fc_all,
        "residuals": actual_all - fc_all,
    }


# ── Baseline CV (for comparison) ──────────────────────────────────────────────


def _baseline_cv(sm: SeriesModels, h: int = SEASON) -> dict:
    """Evaluate seasonal naïve baseline via rolling-origin CV."""
    series = sm.train
    n = len(series)
    windows = _cv_windows(n, h)

    all_actual, all_fc = [], []
    for train_end, test_end in windows:
        train_s = series.iloc[:train_end].values
        actual = series.iloc[train_end:test_end].values
        fc = baseline_seasonal_naive(train_s, h)
        all_actual.append(actual)
        all_fc.append(fc)

    actual_all = np.concatenate(all_actual)
    forecast_all = np.concatenate(all_fc)
    return {
        "MASE": mase(actual_all, forecast_all, series.values),
        "RMSE": rmse(actual_all, forecast_all),
    }


# ── Residual diagnostics ──────────────────────────────────────────────────────


def _ljung_box(residuals: np.ndarray, lags: int = 12) -> pd.DataFrame:
    return acorr_ljungbox(residuals, lags=lags, return_df=True)


def plot_residuals(bundle: ModelBundle, results: dict | None = None) -> None:
    """3-column diagnostic panel: residuals over time | ACF | histogram."""
    _ATTR = {
        "IV_League": "iv_league",
        "MPD_Core_MRR": "mpd_core_mrr",
        "MPD_Core_OneTime": "mpd_core_onetime",
    }
    _TITLES = {
        "IV_League": "IV League",
        "MPD_Core_MRR": "MPD Core MRR",
        "MPD_Core_OneTime": "MPD Core OneTime",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle("Residual Diagnostics  (CV out-of-sample errors)", fontsize=13)

    for row, col in enumerate(["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]):
        title = _TITLES[col]
        color = COLOURS.get(col, "#607D8B")
        r = (results or {}).get(col, {})
        resid = r.get("residuals", np.array([]))
        lb_p = r.get("LjungBox_p", np.nan)
        sw_p = r.get("ShapiroWilk_p", np.nan)

        if len(resid) == 0:
            sm = getattr(bundle, _ATTR[col])
            resid = np.zeros(len(sm.train))

        # ── Col 0: residuals over time ────────────────────────────────────────
        axes[row, 0].plot(range(len(resid)), resid, color=color, linewidth=0.9)
        axes[row, 0].axhline(0, color="black", linewidth=0.7, linestyle="--")
        axes[row, 0].set_title(f"{title} — Residuals over Time", fontsize=9)
        axes[row, 0].set_ylabel("Error (AUD)")
        axes[row, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # ── Col 1: ACF ────────────────────────────────────────────────────────
        n_lags = min(20, max(2, len(resid) // 4 - 1))
        plot_acf(resid, lags=n_lags, ax=axes[row, 1], alpha=0.05, color=color, zero=False, title="")
        lb_label = f"Ljung-Box min-p = {lb_p:.3f}" if not np.isnan(lb_p) else "Ljung-Box: n/a"
        wn_flag = (
            "✓ white noise"
            if (not np.isnan(lb_p) and lb_p > 0.05)
            else "✗ autocorrelation detected"
        )
        axes[row, 1].set_title(f"{title} — ACF\n{lb_label}  {wn_flag}", fontsize=9)

        # ── Col 2: histogram ──────────────────────────────────────────────────
        axes[row, 2].hist(resid, bins=20, color=color, edgecolor="white", alpha=0.8)
        sw_label = f"Shapiro-Wilk p = {sw_p:.3f}" if not np.isnan(sw_p) else "Shapiro-Wilk: n/a"
        norm_flag = "✓ normal" if (not np.isnan(sw_p) and sw_p > 0.05) else "✗ non-normal"
        axes[row, 2].set_title(f"{title} — Distribution\n{sw_label}  {norm_flag}", fontsize=9)
        axes[row, 2].set_xlabel("Error (AUD)")

    fig.tight_layout()
    storage.save_figure("figures/04_residuals.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/04_residuals.png")


# ── CV performance plot ───────────────────────────────────────────────────────


def plot_cv_performance(
    results: dict[str, dict],
    monthly: pd.DataFrame,
) -> None:
    series_names = ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]
    fig, axes = plt.subplots(len(series_names), 1, figsize=(14, 12))
    fig.suptitle("Cross-Validation: Actual vs. Forecast (all CV windows pooled)", fontsize=13)

    for ax, col in zip(axes, series_names, strict=False):
        res = results.get(col, {})
        actual = res.get("actual", np.array([]))
        forecast = res.get("forecast", np.array([]))
        n_pts = len(actual)

        ax.scatter(range(n_pts), actual, color="#333", s=20, label="Actual", zorder=3)
        ax.scatter(
            range(n_pts),
            forecast,
            color=COLOURS.get(col, "#607D8B"),
            s=20,
            alpha=0.7,
            label="Forecast",
            zorder=2,
        )
        ax.set_title(
            f"{col.replace('_', ' ')}  |  "
            f"MASE={res.get('MASE', np.nan):.3f}  "
            f"RMSE={res.get('RMSE', np.nan):,.0f}  "
            f"Winkler={res.get('Winkler', np.nan):.1f}",
            fontsize=9,
        )
        ax.set_ylabel("AUD")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8)

    fig.tight_layout()
    storage.save_figure("figures/04_cv_performance.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/04_cv_performance.png")


# ── Public entry point ────────────────────────────────────────────────────────


def run(bundle: ModelBundle | None, monthly: pd.DataFrame) -> dict[str, dict]:
    """
    Run cross-validation and diagnostics for all three series.

    `bundle=None` triggers a load from the configured outputs store; the loaded
    bundle's `input_sha256` is checked against the current input — a mismatch
    raises so the caller can re-run Phase 3 against the fresh input.

    Returns
    -------
    dict
        ``{series_name: {...}}`` where each per-series dict carries the CV
        metrics (``MASE``, ``RMSE``, ``Winkler``), the pooled ``actual`` /
        ``forecast`` arrays, residual diagnostics (``LjungBox_p``,
        ``ShapiroWilk_p`` — present once enough residuals exist), and the
        baseline comparison (``Baseline_MASE``, ``beats_baseline``). The
        manifest writer reads ``LjungBox_p`` from this dict.
    """
    if bundle is None:
        bundle = phase3_models.load_bundle()
        current_sha = phase1_data.current_input_sha256()
        if bundle.input_sha256 and bundle.input_sha256 != current_sha:
            raise RuntimeError(
                f"Bundle is stale: bundle.input_sha256={bundle.input_sha256[:12]}…, "
                f"current input SHA={current_sha[:12]}…. "
                f"Re-run `python run_revenue_pipeline.py --phase 3` against the current input."
            )

    print("\n── Phase 4: Model Evaluation & Validation ───────────────────────")
    print("  Running rolling-origin cross-validation …")

    results = {}

    # ── IV League: compare EAT vs Croston, pick winner ───────────────────────
    print("    IV_League (EAT vs Croston auto-select) …")
    eat_iv = _eat_cv(bundle.iv_league)
    crost_iv = _croston_cv(bundle.iv_league)
    if not np.isnan(crost_iv["MASE"]) and crost_iv["MASE"] < eat_iv["MASE"]:
        print(f"    → Croston wins  MASE {crost_iv['MASE']:.3f} < EAT {eat_iv['MASE']:.3f}")
        results["IV_League"] = crost_iv
        bundle.iv_league.model_type = "Croston"  # Phase 5 will use Croston
    else:
        print(f"    → EAT wins  MASE {eat_iv['MASE']:.3f} ≤ Croston {crost_iv['MASE']:.3f}")
        results["IV_League"] = eat_iv
        # model_type stays 'EAT'

    # ── MPD Core MRR: compare EAT vs Level, pick winner ─────────────────────
    print("    MPD_Core_MRR (EAT vs Level auto-select) …")
    eat_mrr = _eat_cv(bundle.mpd_core_mrr)
    level_mrr = _level_cv(bundle.mpd_core_mrr)
    if not np.isnan(level_mrr["MASE"]) and level_mrr["MASE"] < eat_mrr["MASE"]:
        print(f"    → Level wins  MASE {level_mrr['MASE']:.3f} < EAT {eat_mrr['MASE']:.3f}")
        results["MPD_Core_MRR"] = level_mrr
        bundle.mpd_core_mrr.model_type = "Level"
    else:
        print(f"    → EAT wins  MASE {eat_mrr['MASE']:.3f} ≤ Level {level_mrr['MASE']:.3f}")
        results["MPD_Core_MRR"] = eat_mrr
        # model_type stays 'EAT'

    # ── MPD Core OneTime: compare Level vs EAT, pick winner ──────────────────
    print("    MPD_Core_OneTime (Level vs EAT auto-select) …")
    level_ot = _level_cv(bundle.mpd_core_onetime)
    eat_ot = _eat_cv(bundle.mpd_core_onetime)
    if not np.isnan(eat_ot["MASE"]) and eat_ot["MASE"] < level_ot["MASE"]:
        print(f"    → EAT wins  MASE {eat_ot['MASE']:.3f} < Level {level_ot['MASE']:.3f}")
        results["MPD_Core_OneTime"] = eat_ot
        bundle.mpd_core_onetime.model_type = "EAT"
        # Phase 5 needs a fitted sf object — fit it now
        from .phase3_models import fit_eat_ensemble

        sf, sf_df = fit_eat_ensemble(bundle.mpd_core_onetime.train, "MPD_Core_OneTime")
        bundle.mpd_core_onetime.sf = sf
        bundle.mpd_core_onetime.sf_df = sf_df
    else:
        print(f"    → Level wins  MASE {level_ot['MASE']:.3f} ≤ EAT {eat_ot['MASE']:.3f}")
        results["MPD_Core_OneTime"] = level_ot
        # model_type stays 'Level'

    # ── Residual diagnostics: Ljung-Box + Shapiro-Wilk ───────────────────────
    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
        resid = results[col].get("residuals", np.array([]))
        if len(resid) >= 10:
            lags = min(12, max(1, len(resid) // 5))
            lb = _ljung_box(resid, lags=lags)
            results[col]["LjungBox_p"] = float(lb["lb_pvalue"].min())
        if len(resid) >= 3:
            _, sw_p = scipy_stats.shapiro(resid[: min(len(resid), 5000)])
            results[col]["ShapiroWilk_p"] = float(sw_p)

    # ── Seasonal naïve baselines ──────────────────────────────────────────────
    baselines = {}
    baselines["IV_League"] = _baseline_cv(bundle.iv_league, SEASON)
    baselines["MPD_Core_MRR"] = _baseline_cv(bundle.mpd_core_mrr, SEASON)
    baselines["MPD_Core_OneTime"] = _baseline_cv(bundle.mpd_core_onetime, SEASON)

    # Print and save evaluation table
    rows = []
    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
        r = results[col]
        b = baselines.get(col, {})
        lb_p = r.get("LjungBox_p", np.nan)
        sw_p = r.get("ShapiroWilk_p", np.nan)
        rows.append(
            {
                "series": col,
                "model": getattr(
                    bundle,
                    {
                        "IV_League": "iv_league",
                        "MPD_Core_MRR": "mpd_core_mrr",
                        "MPD_Core_OneTime": "mpd_core_onetime",
                    }[col],
                ).model_type,
                "MASE": round(r.get("MASE", np.nan), 4),
                "RMSE": round(r.get("RMSE", np.nan), 2),
                "Winkler_80": round(r.get("Winkler", np.nan), 2),
                "LjungBox_p": round(lb_p, 4) if not np.isnan(lb_p) else np.nan,
                "residuals_white_noise": lb_p > 0.05 if not np.isnan(lb_p) else None,
                "ShapiroWilk_p": round(sw_p, 4) if not np.isnan(sw_p) else np.nan,
                "residuals_normal": sw_p > 0.05 if not np.isnan(sw_p) else None,
                "Baseline_MASE": round(b.get("MASE", np.nan), 4),
                "beats_baseline": r.get("MASE", np.nan) < b.get("MASE", np.nan),
            }
        )

    eval_df = pd.DataFrame(rows)

    # ── Regression gate ───────────────────────────────────────────────────────
    # Run BEFORE persisting the eval table or re-persisting the bundle. A hard
    # regression raises here, so a blocked run leaves the outputs store
    # untouched — the prior eval table, bundle, forecast CSVs, and manifest all
    # stay mutually consistent. (Previously the gate ran in the orchestrator
    # *after* these writes, which could leave a new eval table + bundle beside
    # stale forecasts and a stale manifest.)
    prior = manifest.load_prior_manifest()
    soft_regressions = regression_gate.check_regression(eval_df, prior)
    if soft_regressions:
        print(
            f"  Regression gate: {len(soft_regressions)} soft regression(s) "
            "logged above — model worsened but still beats baseline."
        )

    storage.save_csv("tables/04_evaluation_results.csv", eval_df, index=False)

    # Propagate `beats_baseline` and `Baseline_MASE` back into the per-series
    # results dict so the manifest writer can read them without re-loading the CSV.
    for row in rows:
        s = row["series"]
        if s in results:
            results[s]["beats_baseline"] = row["beats_baseline"]
            results[s]["Baseline_MASE"] = row["Baseline_MASE"]

    # Phase 4 may have rewritten `model_type` per the CV winner (IV_League may
    # flip to Croston; MRR may flip to Level; OneTime may flip to EAT and gain
    # a freshly-fitted `sf` object). Re-persist so `--phase 5` standalone reads
    # the post-eval bundle, not the pristine post-fit bundle from Phase 3.
    phase3_models.save_bundle(bundle)

    print("\n  Evaluation results (MASE < 1.0 beats seasonal naïve baseline):")
    print(
        eval_df[
            [
                "series",
                "model",
                "MASE",
                "RMSE",
                "Winkler_80",
                "beats_baseline",
                "LjungBox_p",
                "residuals_white_noise",
                "ShapiroWilk_p",
                "residuals_normal",
            ]
        ].to_string(index=False)
    )

    # Plots
    plot_cv_performance(results, monthly)
    try:
        plot_residuals(bundle, results)
    except Exception as e:
        print(f"  [warn] Residual plot skipped: {e}")

    return results
