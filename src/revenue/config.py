from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
# In this isolated repo the raw payments extract lives under data/ as a
# synthetic, PII-free sample (the real Stripe export is never committed).
DATA_FILE = ROOT / "data" / "historical_unified_payments.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# ── Series date bounds ────────────────────────────────────────────────────────
SERIES_START = "2020-08-01"  # first month with data
# Last fully completed month — always the month before the current one.
# Computed at import time so the pipeline never needs a manual bump.
_today = date.today()
SERIES_END = (_today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d")

# Months flagged as COVID / startup disruption (inclusive start → exclusive end)
# First ~15 months: Aug 2020 – Oct 2021
COVID_END = "2021-11-01"

# ── Modelling ─────────────────────────────────────────────────────────────────
SEASON = 12  # monthly seasonality period
FORECAST_SHORT = 12  # near-term planning horizon (months)
FORECAST_LONG = 24  # strategic planning horizon (months)
MIN_TRAIN_MONTHS = 24  # minimum training window for cross-validation
CV_WINDOWS = 8  # number of rolling-origin CV windows
CV_STEP = 3  # months between CV window origins

# ── Transaction categorisation ────────────────────────────────────────────────
# Regex matched against the Description column (case-insensitive)
IV_LEAGUE_PATTERN = r"IV League"

# Exact description strings that represent ongoing subscription charges (true MRR)
MRR_DESCRIPTIONS = {"Subscription update", "Subscription creation"}

# Everything else → MPD_Core_OneTime

# ── Plot colours ──────────────────────────────────────────────────────────────
COLOURS = {
    "IV_League": "#2196F3",  # blue
    "MPD_Core_MRR": "#4CAF50",  # green
    "MPD_Core_OneTime": "#FF9800",  # orange
    "Total": "#424242",  # dark grey
    "forecast": "#9C27B0",  # purple
    "pi_80": "#CE93D8",  # light purple
    "pi_95": "#EDE7F6",  # very light purple
    "covid": "#FFCDD2",  # light red
    "outlier": "#F44336",  # red
    "break": "#795548",  # brown
}
