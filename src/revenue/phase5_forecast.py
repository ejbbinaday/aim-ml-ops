"""
Phase 5 – Forecasting & Output
================================
Generates 12-month and 24-month forecasts with prediction intervals,
aggregates to Total Revenue via bottom-up summation, adds Bear/Base/Bull
scenario analysis, and produces all stakeholder-facing outputs.

Outputs
-------
outputs/tables/05_forecast_12m.csv     – 12-month detailed forecast table
outputs/tables/05_forecast_24m.csv     – 24-month detailed forecast table
outputs/tables/05_scenarios.csv        – quarterly Bear / Base / Bull summary
outputs/figures/05_forecast_iv.png
outputs/figures/05_forecast_mrr.png
outputs/figures/05_forecast_onetime.png
outputs/figures/05_forecast_total.png
outputs/figures/05_scenario_chart.png
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import storage

from . import phase1_data, phase3_models
from .config import (
    COLOURS,
    COVID_END,
    FORECAST_LONG,
    FORECAST_SHORT,
)
from .phase3_models import ModelBundle, SeriesModels, croston_forecast

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

LABELS = {
    "IV_League": "IV League",
    "MPD_Core_MRR": "MPD Core – MRR",
    "MPD_Core_OneTime": "MPD Core – One-Time",
    "Total": "Total Revenue",
}


# ── Future exog builder ───────────────────────────────────────────────────────


def _future_exog(sm: SeriesModels, h: int) -> np.ndarray | None:
    """Build the exogenous array for the forecast horizon (all post-COVID → zeros)."""
    if sm.exog_train is None:
        return None
    n_cols = sm.exog_train.shape[1]
    return np.zeros((h, n_cols))


# ── EAT ensemble forecast ─────────────────────────────────────────────────────


def _eat_forecast(
    sm: SeriesModels, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point, lo_80, hi_80, lo_95, hi_95) – all in raw AUD.
    Point = equal-weight average of AutoARIMA, AutoETS, Theta.
    Intervals = median across models (robust to explosive ETS bounds).
    If a structural break_date was injected at training time, a post_break=1
    exogenous column is passed for the entire forecast horizon.
    """
    if sm.break_date is not None:
        future_dates = pd.date_range(
            sm.train.index[-1] + pd.DateOffset(months=1), periods=h, freq="MS"
        )
        X_df = pd.DataFrame(
            {
                "unique_id": sm.name,
                "ds": future_dates,
                "post_break": np.ones(h),
            }
        )
        fc = sm.sf.predict(h=h, level=[80, 95], X_df=X_df)
    else:
        fc = sm.sf.predict(h=h, level=[80, 95])

    # Point forecast: mean of all three models (all well-behaved for point estimates)
    point = (fc["AutoARIMA"] + fc["AutoETS"] + fc["Theta"]).values / 3.0

    # Prediction intervals: median across models, not mean.
    # AutoETS can select a multiplicative-error model whose intervals grow
    # exponentially at long horizons (observed: $42M upper bound at month 24 for MRR).
    # Median is robust — one explosive model cannot pull the ensemble bound up.
    lo_80 = np.median(
        [fc["AutoARIMA-lo-80"].values, fc["AutoETS-lo-80"].values, fc["Theta-lo-80"].values], axis=0
    )
    hi_80 = np.median(
        [fc["AutoARIMA-hi-80"].values, fc["AutoETS-hi-80"].values, fc["Theta-hi-80"].values], axis=0
    )
    lo_95 = np.median(
        [fc["AutoARIMA-lo-95"].values, fc["AutoETS-lo-95"].values, fc["Theta-lo-95"].values], axis=0
    )
    hi_95 = np.median(
        [fc["AutoARIMA-hi-95"].values, fc["AutoETS-hi-95"].values, fc["Theta-hi-95"].values], axis=0
    )

    # Clip to zero (revenue cannot be negative)
    return (
        np.maximum(point, 0),
        np.maximum(lo_80, 0),
        np.maximum(hi_80, 0),
        np.maximum(lo_95, 0),
        np.maximum(hi_95, 0),
    )


# ── BSTS forecast ─────────────────────────────────────────────────────────────


def _bsts_forecast(
    sm: SeriesModels, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point, lo_80, hi_80, lo_95, hi_95) – all in raw AUD (back-transformed).
    """
    exog_f = _future_exog(sm, h)
    fc_obj = sm.bsts_result.get_forecast(steps=h, exog=exog_f)

    log_point = np.asarray(fc_obj.predicted_mean)
    ci_80 = np.asarray(fc_obj.conf_int(alpha=0.20))  # 80 % interval
    ci_95 = np.asarray(fc_obj.conf_int(alpha=0.05))  # 95 % interval

    point = np.expm1(log_point)
    lo_80 = np.expm1(ci_80[:, 0])
    hi_80 = np.expm1(ci_80[:, 1])
    lo_95 = np.expm1(ci_95[:, 0])
    hi_95 = np.expm1(ci_95[:, 1])

    return (
        np.maximum(point, 0),
        np.maximum(lo_80, 0),
        np.maximum(hi_80, 0),
        np.maximum(lo_95, 0),
        np.maximum(hi_95, 0),
    )


# ── Level model forecast ─────────────────────────────────────────────────────


def _level_forecast(
    sm: SeriesModels, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flat level forecast for mean-reverting series (e.g. One-Time revenue).
    Projects the exponentially smoothed recent level as a constant.
    Intervals are fixed-width (empirical sigma) — no horizon compounding,
    because each month is independently drawn around a stable level, not a random walk.
    """

    covid_mask = sm.train.index < pd.Timestamp(COVID_END)
    if not sm.outlier_dummies.empty:
        outlier_mask = sm.outlier_dummies.any(axis=1)
    else:
        outlier_mask = pd.Series(False, index=sm.train.index)

    clean = sm.train[~covid_mask & ~outlier_mask & (sm.train > 0)]
    if len(clean) < 3:
        clean = sm.train[~covid_mask & (sm.train > 0)]
    if len(clean) < 2:
        clean = sm.train[sm.train > 0]

    # Exponentially weighted level — alpha=0.25 weights recent months without
    # over-reacting to a single spike
    alpha = 0.25
    weights = np.array([(1 - alpha) ** i for i in range(len(clean) - 1, -1, -1)])
    weights /= weights.sum()
    level = float(max(np.dot(weights, clean.values), 0.0))

    sigma = float(np.std(clean.values - level)) if len(clean) > 1 else level * 0.5

    point = np.full(h, level)
    lo_80 = np.maximum(point - 1.28 * sigma, 0.0)
    hi_80 = point + 1.28 * sigma
    lo_95 = np.maximum(point - 1.96 * sigma, 0.0)
    hi_95 = point + 1.96 * sigma

    return point, lo_80, hi_80, lo_95, hi_95


# ── Per-series dispatch ───────────────────────────────────────────────────────


def _forecast_series(
    sm: SeriesModels, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sm.model_type == "BSTS":
        return _bsts_forecast(sm, h)
    elif sm.model_type == "Level":
        return _level_forecast(sm, h)
    elif sm.model_type == "EAT":
        pt, lo80, hi80, lo95, hi95 = _eat_forecast(sm, h)
        recent = sm.train.iloc[-18:]
        floor = float(recent[recent > 0].median()) * 0.25
        pt = np.maximum(pt, floor)
        return pt, lo80, hi80, lo95, hi95
    else:  # Croston – point only; intervals set wide as data is sparse
        point = croston_forecast(sm.train, h)
        spread = np.std(sm.train[sm.train > 0]) if (sm.train > 0).sum() > 1 else point * 0.5
        lo_80 = np.maximum(point - 1.28 * spread, 0)
        hi_80 = point + 1.28 * spread
        lo_95 = np.maximum(point - 1.96 * spread, 0)
        hi_95 = point + 1.96 * spread
        return point, lo_80, hi_80, lo_95, hi_95


# ── Forecast date index ───────────────────────────────────────────────────────


def _forecast_index(last_train_date: pd.Timestamp, h: int) -> pd.DatetimeIndex:
    return pd.date_range(last_train_date + pd.DateOffset(months=1), periods=h, freq="MS")


# ── Fan chart ─────────────────────────────────────────────────────────────────


def _plot_fan_chart(
    train: pd.Series,
    fc_dates: pd.DatetimeIndex,
    point: np.ndarray,
    lo_80: np.ndarray,
    hi_80: np.ndarray,
    lo_95: np.ndarray,
    hi_95: np.ndarray,
    title: str,
    colour: str,
    filename: str,
    show_months: int = 36,  # how many historical months to show
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(title, fontsize=13, y=0.98)

    # Historical (last N months for clarity)
    hist = train.iloc[-show_months:]
    ax.plot(hist.index, hist.values, color="#424242", linewidth=1.8, label="Historical")

    # 95 % PI band
    ax.fill_between(fc_dates, lo_95, hi_95, color=colour, alpha=0.15, label="95% PI")

    # 80 % PI band
    ax.fill_between(fc_dates, lo_80, hi_80, color=colour, alpha=0.35, label="80% PI")

    # Point forecast
    ax.plot(
        fc_dates, point, color=colour, linewidth=2.0, linestyle="--", label="Point Forecast (Base)"
    )

    # Vertical divider
    ax.axvline(fc_dates[0], color="#9E9E9E", linewidth=0.8, linestyle=":")

    # Bear / Base / Bull labels at final forecast month
    ax.annotate(
        f"Bull  ${hi_80[-1]:,.0f}",
        xy=(fc_dates[-1], hi_80[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color=colour,
        va="center",
    )
    ax.annotate(
        f"Base  ${point[-1]:,.0f}",
        xy=(fc_dates[-1], point[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color="#424242",
        va="center",
    )
    ax.annotate(
        f"Bear  ${lo_80[-1]:,.0f}",
        xy=(fc_dates[-1], lo_80[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=8,
        color="#9E9E9E",
        va="center",
    )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("AUD")
    ax.legend(fontsize=9, loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    storage.save_figure(f"figures/{filename}", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{filename}")


# ── Scenario chart ────────────────────────────────────────────────────────────


def _plot_scenario_chart(scenario_df: pd.DataFrame) -> None:
    """Bar chart of quarterly total revenue scenarios (Bear / Base / Bull)."""
    df = scenario_df.copy()
    df["quarter"] = df["month"].dt.to_period("Q").astype(str)
    quarterly = df.groupby("quarter")[["Bear", "Base", "Bull"]].sum() / 1_000  # in AUD thousands

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(quarterly))
    width = 0.25

    ax.bar(x - width, quarterly["Bear"], width, label="Bear", color="#9E9E9E", alpha=0.85)
    ax.bar(x, quarterly["Base"], width, label="Base", color=COLOURS["forecast"], alpha=0.85)
    ax.bar(
        x + width, quarterly["Bull"], width, label="Bull", color=COLOURS["IV_League"], alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(quarterly.index, rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}K"))
    ax.set_ylabel("Total Revenue (AUD thousands)")
    ax.set_title("Quarterly Total Revenue – Bear / Base / Bull Scenarios", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()

    storage.save_figure("figures/05_scenario_chart.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/05_scenario_chart.png")


# ── Assemble forecast DataFrame ───────────────────────────────────────────────


def _build_forecast_df(
    fc_dates: pd.DatetimeIndex,
    iv: tuple,
    mrr: tuple,
    ot: tuple,
) -> pd.DataFrame:
    iv_pt, iv_lo80, iv_hi80, iv_lo95, iv_hi95 = iv
    mrr_pt, mrr_lo80, mrr_hi80, mrr_lo95, mrr_hi95 = mrr
    ot_pt, ot_lo80, ot_hi80, ot_lo95, ot_hi95 = ot

    total_pt = iv_pt + mrr_pt + ot_pt
    total_lo80 = iv_lo80 + mrr_lo80 + ot_lo80
    total_hi80 = iv_hi80 + mrr_hi80 + ot_hi80
    total_lo95 = iv_lo95 + mrr_lo95 + ot_lo95
    total_hi95 = iv_hi95 + mrr_hi95 + ot_hi95

    df = pd.DataFrame(
        {
            "month": fc_dates,
            "IV_League": iv_pt.round(2),
            "MPD_Core_MRR": mrr_pt.round(2),
            "MPD_Core_OneTime": ot_pt.round(2),
            "Total_Base": total_pt.round(2),
            "Total_lo_80": total_lo80.round(2),
            "Total_hi_80": total_hi80.round(2),
            "Total_lo_95": total_lo95.round(2),
            "Total_hi_95": total_hi95.round(2),
            # Scenarios (80 % bounds)
            "Bear": total_lo80.round(2),
            "Base": total_pt.round(2),
            "Bull": total_hi80.round(2),
        }
    )
    return df


# ── Public entry point ────────────────────────────────────────────────────────


def run(bundle: ModelBundle | None, monthly: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Generate all forecasts, charts, and stakeholder tables.

    `bundle=None` triggers a load from the configured outputs store; the loaded
    bundle's `input_sha256` is checked against the current input, and a mismatch
    raises so the caller can re-run Phase 3 against the fresh input.

    Returns
    -------
    dict with keys '12m' and '24m', each a forecast DataFrame.
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

    print("\n── Phase 5: Forecasting & Output ────────────────────────────────")

    last_date = monthly.index[-1]
    forecasts = {}

    for h, label in [(FORECAST_SHORT, "12m"), (FORECAST_LONG, "24m")]:
        print(f"\n  Generating {h}-month ({label}) forecasts …")

        fc_dates = _forecast_index(last_date, h)

        iv_fc = _forecast_series(bundle.iv_league, h)
        mrr_fc = _forecast_series(bundle.mpd_core_mrr, h)
        ot_fc = _forecast_series(bundle.mpd_core_onetime, h)

        df = _build_forecast_df(fc_dates, iv_fc, mrr_fc, ot_fc)
        storage.save_csv(f"tables/05_forecast_{label}.csv", df, index=False)
        forecasts[label] = df

        print(f"    Saved: tables/05_forecast_{label}.csv")

        # Per-series fan charts
        _plot_fan_chart(
            train=monthly["IV_League"],
            fc_dates=fc_dates,
            point=iv_fc[0],
            lo_80=iv_fc[1],
            hi_80=iv_fc[2],
            lo_95=iv_fc[3],
            hi_95=iv_fc[4],
            title=f"IV League Revenue Forecast – {h}-Month Horizon",
            colour=COLOURS["IV_League"],
            filename=f"05_forecast_iv_{label}.png",
        )
        _plot_fan_chart(
            train=monthly["MPD_Core_MRR"],
            fc_dates=fc_dates,
            point=mrr_fc[0],
            lo_80=mrr_fc[1],
            hi_80=mrr_fc[2],
            lo_95=mrr_fc[3],
            hi_95=mrr_fc[4],
            title=f"MPD Core MRR Forecast – {h}-Month Horizon",
            colour=COLOURS["MPD_Core_MRR"],
            filename=f"05_forecast_mrr_{label}.png",
        )
        _plot_fan_chart(
            train=monthly["MPD_Core_OneTime"],
            fc_dates=fc_dates,
            point=ot_fc[0],
            lo_80=ot_fc[1],
            hi_80=ot_fc[2],
            lo_95=ot_fc[3],
            hi_95=ot_fc[4],
            title=f"MPD Core One-Time Revenue Forecast – {h}-Month Horizon",
            colour=COLOURS["MPD_Core_OneTime"],
            filename=f"05_forecast_onetime_{label}.png",
        )

        # Total Revenue fan chart
        _plot_fan_chart(
            train=monthly["Total"],
            fc_dates=fc_dates,
            point=df["Total_Base"].values,
            lo_80=df["Total_lo_80"].values,
            hi_80=df["Total_hi_80"].values,
            lo_95=df["Total_lo_95"].values,
            hi_95=df["Total_hi_95"].values,
            title=f"Total Revenue Forecast – {h}-Month Horizon (Bottom-Up Aggregation)",
            colour=COLOURS["forecast"],
            filename=f"05_forecast_total_{label}.png",
        )

    # Scenario chart (24-month)
    _plot_scenario_chart(forecasts["24m"])
    storage.save_csv("tables/05_scenarios.csv", forecasts["24m"], index=False)

    # Print 12-month summary
    df12 = forecasts["12m"]
    print("\n  12-Month Forecast Summary (Total Revenue):")
    print(f"    {'Month':<12}  {'Bear':>12}  {'Base':>12}  {'Bull':>12}")
    print(f"    {'-'*52}")
    for _, row in df12.iterrows():
        print(
            f"    {row['month'].strftime('%b %Y'):<12}"
            f"  ${row['Bear']:>11,.0f}"
            f"  ${row['Base']:>11,.0f}"
            f"  ${row['Bull']:>11,.0f}"
        )

    annual = df12[["Bear", "Base", "Bull"]].sum()
    print("\n  12-Month Totals:")
    print(f"    Bear (conservative): AUD {annual['Bear']:>12,.0f}")
    print(f"    Base (expected):     AUD {annual['Base']:>12,.0f}")
    print(f"    Bull (optimistic):   AUD {annual['Bull']:>12,.0f}")

    return forecasts
