"""
Phase 1 – Data Engineering & Preprocessing
===========================================
Transforms the raw Stripe export into three clean monthly time series:
  • IV_League          – Calendly IV League bookings
  • MPD_Core_MRR       – Subscription creation and update charges (true recurring revenue)
  • MPD_Core_OneTime   – All other MPD Core transactions (package initiations,
                         Keyman/corporate, invoices, consultations, etc.)

Outputs
-------
outputs/tables/01_monthly_series.csv   – wide-format monthly DataFrame
outputs/tables/01_category_summary.csv – per-category transaction audit
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from src import storage

from .config import (
    COVID_END,
    DATA_FILE,
    IV_LEAGUE_PATTERN,
    MRR_DESCRIPTIONS,
    SERIES_END,
    SERIES_START,
)

SNAPSHOT_KEY = "tables/00_stripe_charges.csv"
_REQUIRED_COLUMNS = ("Status", "Amount", "Description", "Created date (UTC)")


def current_input_sha256() -> str:
    """SHA-256 of whichever CSV `load_and_clean()` would currently read.

    Centralizes source selection so the idempotency gate, the manifest writer,
    and the Phase 3 bundle's `input_sha256` all see the same bytes.
    """
    if storage.exists(SNAPSHOT_KEY):
        return storage.compute_sha256(SNAPSHOT_KEY)
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


# ── Categorisation ────────────────────────────────────────────────────────────


def _categorize(desc: str) -> str:
    if pd.isna(desc) or str(desc).strip() == "":
        return "MPD_Core_OneTime"
    if re.search(IV_LEAGUE_PATTERN, desc, re.IGNORECASE):
        return "IV_League"
    if str(desc).strip() in MRR_DESCRIPTIONS:
        return "MPD_Core_MRR"
    return "MPD_Core_OneTime"


# ── Loading ───────────────────────────────────────────────────────────────────


def _validate_columns(df: pd.DataFrame, source_label: str) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)} (source: {source_label})")


def read_source() -> tuple[pd.DataFrame, dict]:
    """Read the raw charges from the source (no cleaning) + provenance meta.

    Source selection: prefer the Stripe API snapshot at `tables/00_stripe_charges.csv`
    (via `src.storage`, so it works against either local disk or S3). Fall back
    to the repo-root historical CSV when the snapshot is absent — the historical
    CSV lives outside the outputs store and is read directly via `pd.read_csv`.
    Validates the required columns are present (raises `ValueError` if not).

    This is the single source-selection entry point shared by `load_and_clean`
    (full pipeline) and `pipeline_m1.extract` (Milestone 1 DAG), so both always
    read the same source with the same precedence.
    """
    if storage.exists(SNAPSHOT_KEY):
        df = storage.load_csv(SNAPSHOT_KEY, parse_dates=["Created date (UTC)"])
        meta = {"input_source": "stripe_api", "input_path": SNAPSHOT_KEY}
    else:
        df = pd.read_csv(DATA_FILE, parse_dates=["Created date (UTC)"])
        meta = {"input_source": "historical_csv", "input_path": str(DATA_FILE)}

    _validate_columns(df, meta["input_path"])
    return df, meta


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Read the raw charges and clean them (Paid-only, AUD-only, categorised).

    Returns a tuple of the cleaned DataFrame and a small metadata dict the
    pipeline driver propagates into the manifest's `input_source` / `input_path`.
    """
    df, meta = read_source()
    return clean_raw(df), meta


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw charges frame: AUD-only, Paid-only, dated, categorised.

    Split out of `load_and_clean` so the Milestone 1 extract→validate→load
    pipeline (`pipeline_m1`) and the full offline pipeline share exactly one
    cleaning path. Returns the 4-column frame downstream aggregation expects:
    ``date``, ``amount``, ``category``, ``Description``.
    """
    # AUD-only guard. The monthly totals are summed without FX conversion, so a
    # stray USD/NZD charge would silently inflate the series. Drop non-AUD rows
    # and warn loudly; the historical CSV is all-AUD today, so the expected
    # count is zero.
    if "Currency" in df.columns:
        currency = df["Currency"].astype("string").str.lower()
        non_aud_mask = currency.notna() & (currency != "aud")
        if non_aud_mask.any():
            dropped = sorted(currency[non_aud_mask].unique().tolist())
            print(
                f"  WARN: dropping {int(non_aud_mask.sum()):,} non-AUD rows (currencies: {dropped})"
            )
            df = df.loc[~non_aud_mask].copy()

    df = df[df["Status"] == "Paid"].copy()
    # `Created date (UTC)` may parse as tz-aware or tz-naive depending on the
    # source; normalise to naive wall-clock either way. If the column didn't
    # parse as datetime at all (an unparseable value leaves it object-dtype in
    # the ungated full-pipeline path), fail with a clear data-quality message
    # rather than an opaque `.dt` AttributeError.
    created = df["Created date (UTC)"]
    if not pd.api.types.is_datetime64_any_dtype(created):
        raise ValueError(
            "'Created date (UTC)' did not parse as datetime — check the source "
            "for malformed date values."
        )
    if getattr(created.dt, "tz", None) is not None:
        created = created.dt.tz_localize(None)
    df["date"] = created
    df["amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["category"] = df["Description"].apply(_categorize)

    return df[["date", "amount", "category", "Description"]].reset_index(drop=True)


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily transactions to monthly buckets aligned to month-start.
    Missing months are imputed with zero (no sales that month).
    """
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    pivot = df.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0.0)

    # Cap to actual data: if data hasn't reached SERIES_END (historical CSV
    # fallback or a timing race on the 1st), reindex would pad trailing months
    # with 0.0 — silently injecting a fake revenue collapse that anchors the
    # forecast off garbage. Use the lesser of SERIES_END and the last month the
    # data actually covers.
    data_end = pivot.index.max() if not pivot.empty else pd.Timestamp(SERIES_START)
    effective_end = min(pd.Timestamp(SERIES_END), data_end)
    if effective_end < pd.Timestamp(SERIES_END):
        print(
            f"  WARN: data ends at {effective_end.strftime('%b %Y')}, "
            f"behind SERIES_END={SERIES_END} — trailing months not padded. "
            f"Run phase 0 (Stripe ingest) to close the gap."
        )
    full_idx = pd.date_range(SERIES_START, effective_end, freq="MS")
    pivot = pivot.reindex(full_idx, fill_value=0.0)
    pivot.index.name = "month"

    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    return pivot[["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]]


# ── Feature engineering ───────────────────────────────────────────────────────


def add_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add COVID flag, log-transformed columns, and total revenue."""
    monthly = monthly.copy()

    # Structural break flag (1 = COVID/startup period)
    monthly["is_covid_startup"] = (monthly.index < pd.Timestamp(COVID_END)).astype(int)

    # Log-transformed series (log1p handles zero months gracefully)
    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
        monthly[f"{col}_log"] = np.log1p(monthly[col])

    # Total revenue (raw AUD)
    monthly["Total"] = monthly["IV_League"] + monthly["MPD_Core_MRR"] + monthly["MPD_Core_OneTime"]

    return monthly


# ── Audit summary ─────────────────────────────────────────────────────────────


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("category")["amount"]
        .agg(transactions="count", total_aud="sum", avg_aud="mean", min_aud="min", max_aud="max")
        .round(2)
        .sort_values("total_aud", ascending=False)
    )
    summary["pct_of_total"] = (summary["total_aud"] / summary["total_aud"].sum() * 100).round(1)
    return summary


# ── Public entry point ────────────────────────────────────────────────────────


def run() -> tuple[pd.DataFrame, dict]:
    """
    Execute Phase 1 end-to-end.

    Returns
    -------
    (monthly, meta)
        monthly: Monthly series indexed by month-start date.
        meta: ``{"input_source": ..., "input_path": ...}`` for the manifest writer.
    """
    df_raw, meta = load_and_clean()
    monthly = aggregate_monthly(df_raw)
    monthly = add_features(monthly)

    # Save outputs via the storage switch (local outputs/ or S3, per OUTPUTS_BUCKET).
    storage.save_csv("tables/01_monthly_series.csv", monthly)
    storage.save_csv("tables/01_category_summary.csv", category_summary(df_raw))

    # Console summary
    print("\n── Phase 1: Data Engineering ────────────────────────────────────")
    print(f"  Raw transactions (Paid): {len(df_raw):,}")
    print(f"  Date range : {monthly.index[0].date()} → {monthly.index[-1].date()}")
    print(f"  Months     : {len(monthly)}")
    print(f"  COVID flag : months 1–{monthly['is_covid_startup'].sum()} flagged")
    print()
    for col in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime", "Total"]:
        nonzero = (monthly[col] > 0).sum()
        print(
            f"  {col:<22} total AUD {monthly[col].sum():>12,.0f} "
            f"| non-zero months {nonzero}/{len(monthly)}"
        )
    print(f"  Source     : {meta['input_source']} ({meta['input_path']})")

    return monthly, meta
