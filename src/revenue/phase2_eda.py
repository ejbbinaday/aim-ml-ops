"""
Phase 2 – Exploratory Data Analysis
=====================================
Produces four analytical outputs before any model is fit:

1. Time-series plots  – raw revenue for all three series + COVID shading
2. STL decomposition  – trend / seasonal / remainder for each series (robust)
3. Structural breaks  – CUSUM-based changepoint detection (ruptures library)
4. ACF / PACF plots   – autocorrelation structure of each series
5. Hampel outlier detection on STL remainder → outlier timestamps returned
   to the pipeline so Phase 3 can use them as intervention dummy variables.

Outputs
-------
outputs/figures/02_time_series.png
outputs/figures/02_stl_decomposition.png
outputs/figures/02_structural_breaks.png
outputs/figures/02_acf_plots.png
outputs/tables/02_outliers.csv
outputs/tables/02_break_dates.csv
"""

from __future__ import annotations

from typing import NamedTuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from src import storage

from .config import COLOURS, COVID_END

SERIES = ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]
LABELS = {
    "IV_League": "IV League",
    "MPD_Core_MRR": "MPD Core – MRR",
    "MPD_Core_OneTime": "MPD Core – One-Time",
}

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


class EDAResult(NamedTuple):
    outlier_dates: dict[str, list[pd.Timestamp]]  # series → list of outlier months
    break_dates: dict[str, list[pd.Timestamp]]  # series → list of detected breaks


# ── Helper: shade COVID period ────────────────────────────────────────────────


def _shade_covid(ax: plt.Axes, series: pd.Series) -> None:
    start = series.index[0]
    end = pd.Timestamp(COVID_END)
    ax.axvspan(start, end, color=COLOURS["covid"], alpha=0.5, label="COVID/startup")


# ── 1. Time-series plots ──────────────────────────────────────────────────────


def plot_time_series(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("MPD Revenue – Historical Monthly Time Series", fontsize=14, y=0.98)

    series_to_plot = SERIES + ["Total"]
    for ax, col in zip(axes, series_to_plot, strict=False):
        s = monthly[col]
        colour = COLOURS.get(col, "#607D8B")
        ax.fill_between(s.index, s.values, alpha=0.15, color=colour)
        ax.plot(s.index, s.values, color=colour, linewidth=1.6, label=LABELS.get(col, col))
        _shade_covid(ax, s)
        ax.set_ylabel("AUD", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Month")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    storage.save_figure("figures/02_time_series.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/02_time_series.png")


# ── 2. STL decomposition + Hampel outlier detection ──────────────────────────


def _hampel_detect(residual: np.ndarray, window: int = 5, n_sigma: float = 3.0) -> np.ndarray:
    """
    Hampel filter on the STL remainder component.
    Returns a boolean mask where True = outlier.
    Window uses local median ± n_sigma * 1.4826 * MAD.
    """
    s = pd.Series(residual)
    k = 1.4826  # consistency factor: MAD → σ for Gaussian
    rolling_med = s.rolling(window, center=True, min_periods=1).median()
    rolling_mad = (s - rolling_med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * k * rolling_mad
    return ((s - rolling_med).abs() > threshold).values


def run_stl_and_outliers(
    monthly: pd.DataFrame,
    n_sigma: float = 4.5,
    max_outliers: int = 5,
) -> dict[str, list[pd.Timestamp]]:
    """
    Fit STL (robust=True) to each series. Plot decomposition.
    Detect the most extreme Remainder spikes via the Hampel criterion.
    Caps at max_outliers per series to avoid over-parameterising models.
    Returns {series_name: [outlier_timestamps]}.
    """
    fig, axes = plt.subplots(len(SERIES), 4, figsize=(18, 4 * len(SERIES)))
    fig.suptitle("STL Decomposition (robust=True) per Revenue Series", fontsize=13)

    outlier_dates: dict[str, list[pd.Timestamp]] = {}

    for row, col in enumerate(SERIES):
        s = monthly[col]
        stl = STL(s, period=12, robust=True)
        res = stl.fit()

        components = {
            "Observed": s.values,
            "Trend": res.trend,
            "Seasonal": res.seasonal,
            "Remainder": res.resid,
        }

        outlier_mask = _hampel_detect(res.resid, n_sigma=n_sigma)
        # Rank by magnitude and keep only the most extreme to avoid over-parameterisation
        if outlier_mask.sum() > max_outliers:
            magnitudes = np.abs(res.resid)
            ranked_idx = np.argsort(magnitudes)[::-1]
            keep = np.zeros(len(outlier_mask), dtype=bool)
            keep[ranked_idx[:max_outliers]] = True
            outlier_mask = outlier_mask & keep
        outlier_timestamps = list(s.index[outlier_mask])
        outlier_dates[col] = outlier_timestamps

        colour = COLOURS[col]
        for j, (label, data) in enumerate(components.items()):
            ax = axes[row, j]
            ax.plot(s.index, data, color=colour, linewidth=1.2)
            if label == "Remainder" and outlier_mask.any():
                ax.scatter(
                    s.index[outlier_mask],
                    data[outlier_mask],
                    color=COLOURS["outlier"],
                    zorder=5,
                    s=40,
                    label="Outlier",
                )
                ax.legend(fontsize=7)
            ax.set_title(f"{LABELS[col]} – {label}", fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    storage.save_figure("figures/02_stl_decomposition.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/02_stl_decomposition.png")

    return outlier_dates


# ── 3. Structural break detection ─────────────────────────────────────────────


def detect_structural_breaks(
    monthly: pd.DataFrame, penalty: float = 15.0
) -> dict[str, list[pd.Timestamp]]:
    """
    Use ruptures Pelt (rbf kernel) to detect structural breaks in each series.
    Returns {series_name: [break_timestamps]}.
    """
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Structural Break Detection (PELT, RBF kernel)", fontsize=13)

    break_dates: dict[str, list[pd.Timestamp]] = {}

    for ax, col in zip(axes, SERIES, strict=False):
        s = monthly[col]
        signal = s.values.reshape(-1, 1)

        algo = rpt.Pelt(model="rbf").fit(signal)
        result = algo.predict(pen=penalty)
        breakpoints = result[:-1]  # last element is len(signal), not a real break

        break_timestamps = [s.index[bp - 1] for bp in breakpoints if bp < len(s)]
        break_dates[col] = break_timestamps

        ax.plot(s.index, s.values, color=COLOURS[col], linewidth=1.4, label=LABELS[col])
        _shade_covid(ax, s)
        for bt in break_timestamps:
            ax.axvline(
                bt,
                color=COLOURS["break"],
                linestyle="--",
                linewidth=1.2,
                label=f"Break: {bt.strftime('%b %Y')}",
            )
        ax.set_ylabel("AUD", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    axes[-1].set_xlabel("Month")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    storage.save_figure("figures/02_structural_breaks.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/02_structural_breaks.png")

    return break_dates


# ── 4. ACF / PACF plots ───────────────────────────────────────────────────────


def plot_acf_pacf(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(SERIES), 2, figsize=(14, 4 * len(SERIES)))
    fig.suptitle("Autocorrelation (ACF) and Partial Autocorrelation (PACF)", fontsize=13)

    for row, col in enumerate(SERIES):
        s = monthly[col]
        plot_acf(s, lags=24, ax=axes[row, 0], title=f"{LABELS[col]} – ACF", color=COLOURS[col])
        plot_pacf(
            s,
            lags=24,
            ax=axes[row, 1],
            title=f"{LABELS[col]} – PACF",
            color=COLOURS[col],
            method="ywm",
        )
        for ax in axes[row]:
            ax.set_xlabel("Lag (months)")
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    storage.save_figure("figures/02_acf_plots.png", fig=fig, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figures/02_acf_plots.png")


# ── Public entry point ────────────────────────────────────────────────────────


def run(monthly: pd.DataFrame) -> EDAResult:
    print("\n── Phase 2: Exploratory Data Analysis ───────────────────────────")

    plot_time_series(monthly)
    outlier_dates = run_stl_and_outliers(monthly)
    break_dates = detect_structural_breaks(monthly)
    plot_acf_pacf(monthly)

    # Report outliers
    print("\n  Hampel outliers detected (STL remainder ±3 MAD):")
    all_outliers = []
    for col, dates in outlier_dates.items():
        print(f"    {LABELS[col]}: {len(dates)} month(s) → {[d.strftime('%b %Y') for d in dates]}")
        for d in dates:
            all_outliers.append({"series": col, "month": d})
    storage.save_csv("tables/02_outliers.csv", pd.DataFrame(all_outliers), index=False)

    # Report structural breaks
    print("\n  Structural breaks detected:")
    all_breaks = []
    for col, dates in break_dates.items():
        print(f"    {LABELS[col]}: {len(dates)} break(s) → {[d.strftime('%b %Y') for d in dates]}")
        for d in dates:
            all_breaks.append({"series": col, "month": d})
    storage.save_csv("tables/02_break_dates.csv", pd.DataFrame(all_breaks), index=False)

    return EDAResult(outlier_dates=outlier_dates, break_dates=break_dates)
