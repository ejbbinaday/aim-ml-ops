"""Generate a synthetic, PII-free ``historical_unified_payments.csv``.

The real pipeline reads a Stripe charges export, which is confidential customer
financial data and must never be committed. This script fabricates a
structurally faithful stand-in: ~5 years of transaction-level rows across the
three revenue streams the pipeline categorises, with realistic seasonality,
growth, sparsity and noise — plus a handful of non-Paid and non-AUD rows so the
cleaning guards have something to exercise. It is fully deterministic (fixed
seed), so re-running reproduces the same file byte-for-byte.

Run:  uv run python scripts/generate_synthetic_data.py
Out:  data/historical_unified_payments.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "historical_unified_payments.csv"

RNG = np.random.default_rng(20260716)

START = pd.Timestamp("2020-08-01")
# 71 whole months → Aug 2020 … Jun 2026, matching the pipeline's dynamic
# SERIES_END (the last complete month) as of this deliverable, so a fresh run
# has no trailing-month gap. Fixed count → the committed sample is stable and
# reproducible byte-for-byte regardless of the wall clock.
N_MONTHS = 71

# MRR subscription tiers (monthly AUD) and their relative popularity.
TIERS = {"Silver": 199.0, "Gold": 349.0, "Platinum": 599.0}
TIER_WEIGHTS = np.array([0.55, 0.30, 0.15])

# One-time revenue product lines (description → (mean, sd) AUD).
ONE_TIME = {
    "Package initiation": (2200.0, 600.0),
    "Corporate wellness - Keyman": (7500.0, 2500.0),
    "Invoice payment": (1400.0, 500.0),
    "Initial consultation": (350.0, 60.0),
    "Longevity panel": (900.0, 200.0),
}


def _month_starts() -> list[pd.Timestamp]:
    return [START + pd.DateOffset(months=i) for i in range(N_MONTHS)]


def _random_datetime(month: pd.Timestamp) -> str:
    """A tz-naive 'Created date (UTC)' string at a random instant in `month`."""
    days_in_month = (month + pd.DateOffset(months=1) - month).days
    day = int(RNG.integers(1, days_in_month + 1))
    hour = int(RNG.integers(6, 21))
    minute = int(RNG.integers(0, 60))
    ts = month.replace(day=day, hour=hour, minute=minute, second=int(RNG.integers(0, 60)))
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _seasonal(month_idx: int, amplitude: float, phase: float = 0.0) -> float:
    return 1.0 + amplitude * np.sin(2 * np.pi * (month_idx / 12.0) + phase)


def _build() -> pd.DataFrame:
    rows: list[dict] = []
    months = _month_starts()

    # Subscriber base grows from ~40 to ~300 over the window (logistic-ish).
    base = np.linspace(40, 300, N_MONTHS) * np.array([_seasonal(i, 0.08) for i in range(N_MONTHS)])

    for i, month in enumerate(months):
        # COVID/startup drag on the first ~15 months.
        covid = 0.55 if i < 15 else 1.0
        n_subs = max(1, int(base[i] * covid))

        # ── MRR: renewals (Subscription update) + a slice of new (creation) ──
        n_new = int(RNG.binomial(n_subs, 0.12))
        n_renew = n_subs - n_new
        for desc, count in (("Subscription creation", n_new), ("Subscription update", n_renew)):
            tiers = RNG.choice(list(TIERS), size=count, p=TIER_WEIGHTS)
            for tier in tiers:
                amt = TIERS[tier] * (1 + RNG.normal(0, 0.02))
                rows.append(_row(month, desc, round(amt, 2)))

        # ── IV League: intermittent Calendly bookings (some months zero) ──
        iv_lambda = max(0.0, 8 * covid * _seasonal(i, 0.4, phase=1.2) - 2)
        n_iv = int(RNG.poisson(iv_lambda))
        for _ in range(n_iv):
            amt = float(RNG.choice([180, 250, 320, 450], p=[0.3, 0.35, 0.25, 0.1]))
            rows.append(
                _row(month, "IV League - Booking", round(amt * (1 + RNG.normal(0, 0.05)), 2))
            )

        # ── One-time: lumpy packages / corporate / invoices / consults ──
        n_ot = int(RNG.poisson(max(1.0, 6 * covid * _seasonal(i, 0.25, phase=0.5))))
        for _ in range(n_ot):
            desc = RNG.choice(list(ONE_TIME), p=[0.35, 0.08, 0.22, 0.25, 0.10])
            mean, sd = ONE_TIME[desc]
            amt = max(50.0, RNG.normal(mean, sd))
            rows.append(_row(month, desc, round(amt, 2)))

    # A few non-Paid rows (filtered out by the Paid guard).
    for _ in range(40):
        month = RNG.choice(months)
        r = _row(month, "Subscription update", 349.0)
        r["Status"] = RNG.choice(["Refunded", "Failed"])
        rows.append(r)

    # A few non-AUD rows (dropped by the AUD guard with a loud warning).
    for _ in range(6):
        month = RNG.choice(months)
        r = _row(month, "Initial consultation", 250.0)
        r["Currency"] = "usd"
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("Created date (UTC)").reset_index(drop=True)
    df.insert(0, "id", [f"ch_{n:07d}" for n in range(len(df))])
    return df


def _row(month: pd.Timestamp, desc: str, amount: float) -> dict:
    return {
        "Status": "Paid",
        "Created date (UTC)": _random_datetime(month),
        "Amount": amount,
        "Currency": "aud",
        "Description": desc,
    }


def main() -> None:
    df = _build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} rows → {OUT.relative_to(ROOT)}")
    print(df["Description"].value_counts().to_string())


if __name__ == "__main__":
    main()
