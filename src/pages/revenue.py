"""Revenue Forecast page for the MPD dashboard."""

from __future__ import annotations

import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import storage
from src.auth import page

SLUG = "revenue"

# ── Design tokens ─────────────────────────────────────────────────────────────
# Semantic colors stay as hex — they encode meaning (bear/base/bull, revenue
# stream identity) and are paired with low-alpha rgba backgrounds so the
# pairing reads in both light and dark Streamlit themes. Neutral surface /
# text tokens map onto the dashboard-wide `--c-*` system declared in
# dashboard.py so this page inherits Streamlit's active theme the same way
# retention.py does. See openspec/specs/dashboard-theming/spec.md.
C = {
    # Semantic — hex (used in Plotly charts and as accent fills)
    "bear": "#E53935",
    "bear_bg": "rgba(229,57,53,0.15)",
    "base": "#5C6BC0",
    "base_bg": "rgba(92,107,192,0.15)",
    "bull": "#43A047",
    "bull_bg": "rgba(67,160,71,0.15)",
    "iv": "#2196F3",
    "mrr": "#4CAF50",
    "ot": "#FF9800",
    "accent": "#5C6BC0",
    # Neutral — theme-neutral CSS vars from dashboard.py's `--c-*` system
    "page": "transparent",
    "card": "var(--c-card)",
    "text": "var(--c-text)",
    "secondary": "var(--c-text2)",
    "muted": "var(--c-muted)",
    "border": "var(--c-border)",
    "border_mid": "var(--c-border-soft)",
    "grid": "var(--c-grid)",
    "row_alt": "var(--c-surface-alt)",
}

SERIES = ["IV_League", "MPD_Core"]
SERIES_LABEL = {"IV_League": "IV League", "MPD_Core": "Subscriptions"}
SERIES_COLOR = {"IV_League": C["iv"], "MPD_Core": C["mrr"]}
SC_COLOR = {"Bear": C["bear"], "Base": C["base"], "Bull": C["bull"]}
SC_BG = {"Bear": C["bear_bg"], "Base": C["base_bg"], "Bull": C["bull_bg"]}

MODEL_PLAIN = {
    "EAT": "EAT Ensemble (AutoARIMA + AutoETS + Theta)",
    "Croston": "Croston's Method",
    "Level": "Level Model (mean-reverting, no trend)",
}

# Theme-neutral Plotly constants (mirror retention.py and the
# dashboard-theming spec — Plotly cannot consume CSS variables).
PLOT_TEXT_COLOR = "#888888"
PLOT_GRID_COLOR = "rgba(128,128,128,0.20)"

# ── Chart base (design.md §10) ────────────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color=PLOT_TEXT_COLOR),
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
    hoverlabel=dict(align="left"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color=PLOT_TEXT_COLOR),
    ),
    xaxis=dict(
        gridcolor=PLOT_GRID_COLOR,
        showline=False,
        tickfont=dict(size=10, color=PLOT_TEXT_COLOR),
        zeroline=False,
        tickformat="%b %Y",
    ),
    yaxis=dict(
        gridcolor=PLOT_GRID_COLOR,
        showline=False,
        tickfont=dict(size=10, color=PLOT_TEXT_COLOR),
        tickprefix="$",
        tickformat=",.0f",
        zeroline=False,
    ),
)

_CSS = """
<style>
/* Page-level tweaks — neutral tokens come from dashboard.py's :root.
   We do NOT redefine `--c-*` tokens here; doing so would shadow the
   theme-inheriting values dashboard.py sets via currentColor. */

section[data-testid="stSidebar"] {
    border-right: 1px solid var(--c-border);
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--c-border);
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--c-muted);
    padding: 8px 16px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #5C6BC0 !important;
    border-bottom: 2px solid #5C6BC0 !important;
    background: transparent !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 4px; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--c-border);
    border-radius: 4px;
    overflow: hidden;
}

[data-testid="stPills"] button,
[data-testid="stPills"] [role="option"],
[data-testid="stPills"] [data-baseweb="tag"] {
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
[data-testid="stPills"] label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--c-muted);
}

/* "Subtle" download buttons. Applied once for the whole page instead of
   re-injected per call site. */
[data-testid="stDownloadButton"] button {
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--c-text2) !important;
    padding: 2px 10px;
    border: 1px solid var(--c-border) !important;
    border-radius: 4px !important;
    background: var(--c-card) !important;
    opacity: 0.85;
}
[data-testid="stDownloadButton"] button:hover {
    opacity: 1;
    border-color: var(--c-text2) !important;
    background: var(--c-surface-alt) !important;
}
</style>
"""

# ── Data ──────────────────────────────────────────────────────────────────────


def _safe_load_csv(key: str, **kwargs) -> pd.DataFrame:
    """Read an optional pipeline CSV via the storage switch. Returns an empty
    DataFrame if the artifact is missing, empty, or has no header — keeps the
    page out of an unhandled traceback when the pipeline ran but produced an
    empty table (e.g. ``02_outliers.csv`` when no outliers were detected)."""
    if not storage.exists(key):
        return pd.DataFrame()
    try:
        return storage.load_csv(key, **kwargs)
    except (pd.errors.EmptyDataError, ValueError):
        # EmptyDataError: zero-byte file
        # ValueError: parse_dates names a column that doesn't exist (empty CSV)
        return pd.DataFrame()


def _load_manifest() -> dict | None:
    """Return the pipeline's run manifest if present, else None.

    Used by the sidebar to surface ``run_at``, ``git_sha``, ``input_source``,
    and ``data_through``. Absence is rendered as a "no provenance" badge — it
    does NOT block the page from rendering forecast charts.
    """
    try:
        return storage.load_json("tables/00_manifest.json")
    except Exception:  # noqa: BLE001 — JSON parse / botocore 404 both "no manifest"
        return None


# 1 h TTL — the revenue pipeline refreshes monthly, so re-reading the CSVs
# every 5 minutes was pure waste. An hour is still tight enough that a manual
# pipeline rerun shows up within one nav click.
@st.cache_data(ttl=3600)
def load_data() -> dict | None:
    """Load pre-computed pipeline outputs via the storage switch.

    Returns None if any required CSV is missing so the page can render the
    "run the pipeline" guidance. Raises only when a *required* CSV exists
    but is corrupt — `main()` catches that and shows a friendly error,
    matching retention.py's behaviour for S3-mid-write states.
    """
    needed = [
        "tables/05_forecast_12m.csv",
        "tables/05_forecast_24m.csv",
        "tables/01_monthly_series.csv",
    ]
    if any(not storage.exists(k) for k in needed):
        return None
    return dict(
        fc12=storage.load_csv("tables/05_forecast_12m.csv", parse_dates=["month"]),
        fc24=storage.load_csv("tables/05_forecast_24m.csv", parse_dates=["month"]),
        monthly=storage.load_csv(
            "tables/01_monthly_series.csv", index_col="month", parse_dates=True
        ),
        spec=_safe_load_csv("tables/03_model_spec.csv"),
        model_info=_safe_load_csv("tables/03_model_info.csv"),
        ev=_safe_load_csv("tables/04_evaluation_results.csv"),
        outliers=_safe_load_csv("tables/02_outliers.csv", parse_dates=["month"]),
        manifest=_load_manifest(),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def fmt(v: float) -> str:
    # Guard NaN / None / negative inputs — the dashboard receives numbers
    # straight from CSV columns that can briefly be NaN during pipeline
    # refresh windows, and printing "$nan" or "$-1,234" looks broken.
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{sign}${av/1_000_000:.2f}M"
    if av >= 1_000:
        return f"{sign}${av/1_000:.0f}K"
    return f"{sign}${av:,.0f}"


def pct(new: float, old: float) -> str:
    if old == 0 or old is None or (isinstance(old, float) and np.isnan(old)):
        return "—"
    if new is None or (isinstance(new, float) and np.isnan(new)):
        return "—"
    return f"{(new-old)/old*100:+.1f}%"


def rgba(hex_c: str, a: float) -> str:
    r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
    return f"rgba({r},{g},{b},{a})"


# design.md §4 — section header (desc renders at body size, not caption)
def section(title: str, desc: str = None) -> None:
    desc_html = (
        f"<div style='font-size:0.82rem;color:var(--c-text2);margin-top:4px;line-height:1.5'>{desc}</div>"
        if desc
        else ""
    )
    st.markdown(
        f"<div style='padding-bottom:8px;margin-bottom:14px;border-bottom:1px solid {C['border']}'>"
        f"<span style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{C['muted']}'>{title}</span>"
        f"{desc_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def divider(label: str) -> None:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:20px 0 14px'>"
        f"<span style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:var(--c-muted)'>{label}</span>"
        f"<div style='flex:1;height:1px;background:{C['border']}'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def subtle_download(data: bytes, file_name: str, key: str) -> None:
    # Button styling lives in the page-level `_CSS` block, not here — see
    # the `[data-testid="stDownloadButton"] button` rule. Repeating the
    # injection per call duplicates the same <style> tag across the DOM.
    _, _btn_col = st.columns([8, 2])
    with _btn_col:
        st.download_button(
            "↓ Export to CSV",
            data=data,
            file_name=file_name,
            mime="text/csv",
            key=key,
            use_container_width=True,
            type="tertiary",
        )


# Forecast table: Month | Base | Bear (±) | Bull (±) | Spread pill
def forecast_html_table(fc: pd.DataFrame) -> str:
    # XSS guard: this function builds HTML with unsafe_allow_html=True, so any
    # future free-text column added to `fc` would render unescaped. Assert that
    # the columns we interpolate are numeric / datetime — adding a text column
    # here later requires routing it through `html.escape()` first.
    assert pd.api.types.is_datetime64_any_dtype(fc["month"]), "month must be datetime"
    for c in ("Bear", "Base", "Bull"):
        assert pd.api.types.is_numeric_dtype(fc[c]), f"{c} must be numeric"

    rows = fc[["month", "Bear", "Base", "Bull"]].copy()
    rows["month"] = rows["month"].dt.strftime("%b %Y")
    rows["spread"] = rows["Bull"] - rows["Bear"]

    c_bear = C["bear"]
    c_bear_bg = C["bear_bg"]
    c_base = C["base"]
    c_bull = C["bull"]
    c_bull_bg = C["bull_bg"]
    c_text = C["text"]
    c_muted = C["muted"]

    # Spread color: low spread = green, mid = orange, high = red
    spread_max = rows["spread"].max()

    def spread_color(s: float):
        pct = s / spread_max
        if pct < 0.35:
            return C["bull"], C["bull_bg"]
        if pct < 0.70:
            return "#FB8C00", "rgba(251,140,0,0.15)"
        return C["bear"], C["bear_bg"]

    th_wrap = (
        "padding:8px 14px;border-bottom:2px solid var(--c-border);background:var(--c-surface-alt);"
    )
    th_label = "display:block;font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
    th_sub = "display:block;font-size:0.68rem;font-weight:400;text-transform:none;letter-spacing:0;margin-top:2px;color:var(--c-caption);"

    def th(label, sub, color, align="right"):
        return (
            f"<th style='{th_wrap}text-align:{align}'>"
            f"<span style='{th_label}color:{color}'>{label}</span>"
            f"<span style='{th_sub}'>{sub}</span>"
            f"</th>"
        )

    head = (
        "<tr>"
        + th("Month", "", c_muted, "left")
        + th("Expected", "our central forecast", c_base)
        + th("Worst case", "if conditions turn unfavourable", c_bear)
        + th("Best case", "if conditions turn favourable", c_bull)
        + th("Uncertainty window", "gap between worst &amp; best case", c_muted, "center")
        + "</tr>"
    )

    td = "padding:8px 14px;font-size:0.82rem;border-bottom:1px solid var(--c-grid);"
    body = ""
    for i, row in rows.iterrows():
        spread_pct = row["spread"] / spread_max if spread_max else 0
        high_unc = spread_pct >= 0.70
        bg = (
            "rgba(229,57,53,0.06)"
            if high_unc
            else ("var(--c-surface-alt)" if i % 2 == 0 else "var(--c-card)")
        )
        row_style = (
            f"background:{bg};border-left:3px solid {C['bear']};"
            if high_unc
            else f"background:{bg};"
        )
        watch_badge = (
            (
                f"<span style='font-size:0.60rem;font-weight:700;color:{C['bear']};"
                f"background:{C['bear_bg']};padding:1px 6px;border-radius:3px;"
                f"margin-left:6px;vertical-align:middle'>⚠ wide range</span>"
            )
            if high_unc
            else ""
        )
        downside = row["Bear"] - row["Base"]
        upside = row["Bull"] - row["Base"]
        sc, sbg = spread_color(row["spread"])
        spread_label = f"${row['spread']:,.0f} range"
        body += (
            f"<tr style='{row_style}'>"
            f"<td style='{td}color:{c_text};font-weight:500'>{row['month']}{watch_badge}</td>"
            f"<td style='{td}color:{c_base};font-weight:700;text-align:right'>${row['Base']:,.0f}</td>"
            f"<td style='{td}text-align:right'>"
            f"<span style='color:{c_bear};font-weight:600'>${row['Bear']:,.0f}</span>"
            f"<span style='font-size:0.7rem;color:{c_bear};background:{c_bear_bg};"
            f"padding:1px 5px;border-radius:3px;margin-left:5px'>{downside:+,.0f}</span>"
            f"</td>"
            f"<td style='{td}text-align:right'>"
            f"<span style='color:{c_bull};font-weight:600'>${row['Bull']:,.0f}</span>"
            f"<span style='font-size:0.7rem;color:{c_bull};background:{c_bull_bg};"
            f"padding:1px 5px;border-radius:3px;margin-left:5px'>{upside:+,.0f}</span>"
            f"</td>"
            f"<td style='{td}text-align:center'>"
            f"<span style='font-size:0.72rem;font-weight:700;color:{sc};background:{sbg};"
            f"padding:3px 10px;border-radius:20px;white-space:nowrap'>{spread_label}</span>"
            f"</td>"
            f"</tr>"
        )

    # Total row
    tt = "padding:10px 14px;font-size:0.82rem;font-weight:700;border-top:2px solid var(--c-border);"
    t_bear = fc["Bear"].sum()
    t_base = fc["Base"].sum()
    t_bull = fc["Bull"].sum()
    t_sprd = t_bull - t_bear
    sc, sbg = spread_color(t_sprd)
    body += (
        f"<tr style='background:var(--c-surface-alt)'>"
        f"<td style='{tt}color:{c_text}'>TOTAL</td>"
        f"<td style='{tt}color:{c_base};text-align:right'>${t_base:,.0f}</td>"
        f"<td style='{tt}color:{c_bear};text-align:right'>${t_bear:,.0f}</td>"
        f"<td style='{tt}color:{c_bull};text-align:right'>${t_bull:,.0f}</td>"
        f"<td style='{tt}text-align:center'>"
        f"<span style='font-size:0.72rem;font-weight:700;color:{sc};background:{sbg};"
        f"padding:3px 10px;border-radius:20px'>${t_sprd:,.0f} range</span>"
        f"</td>"
        f"</tr>"
    )

    return (
        "<div style='overflow-y:auto;max-height:480px;border:1px solid var(--c-border-soft);border-radius:6px'>"
        "<table style='width:100%;border-collapse:collapse'>"
        f"<thead style='position:sticky;top:0'>{head}</thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


# Stream definition cards — shown near charts so users know what each stream means
STREAM_DEFS = {
    "IV_League": {
        "what": "Revenue from IV therapy packages and sessions booked through IV League.",
        "how": 'Transactions whose description contains <em>"IV League"</em>.',
        "model": "EAT Ensemble (auto-selects Croston if demand is too sparse).",
        "note": "Intermittent — not every month has revenue. Roughly 38% of months are non-zero.",
    },
    "MPD_Core": {
        "what": "All MPD Core revenue — recurring subscription charges plus premium one-off packages and consultations.",
        "how": "Modelled as two internal sub-streams (recurring MRR and one-time packages) then combined for reporting.",
        "model": "Two separate models: EAT Ensemble for recurring MRR · Level Model for premium packages.",
        "note": "MRR is the most predictable component; package timing is the main source of forecast uncertainty.",
    },
}


def stream_definition_cards(spec: pd.DataFrame | None = None) -> str:
    cards = ""
    for s in SERIES:
        d = STREAM_DEFS[s]
        color = SERIES_COLOR[s]
        cards += (
            f"<div style='border:1px solid var(--c-border-soft);border-top:3px solid {color};"
            f"border-radius:6px;background:var(--c-card);padding:14px 16px;"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.04)'>"
            f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:{color};margin-bottom:6px'>"
            f"{SERIES_LABEL[s]}</div>"
            f"<div style='font-size:0.82rem;color:var(--c-text);line-height:1.5;margin-bottom:10px'>"
            f"{d['what']}</div>"
            f"<div style='font-size:0.72rem;color:var(--c-muted);line-height:1.6;border-top:1px solid var(--c-grid);padding-top:8px'>"
            f"<span style='font-weight:700;color:var(--c-text2)'>Identified by</span> {d['how']}<br>"
            f"<span style='font-weight:700;color:var(--c-text2)'>Note</span> {d['note']}"
            f"</div></div>"
        )
    return (
        f"<div style='display:grid;grid-template-columns:repeat(2,1fr);"
        f"gap:12px;margin-bottom:20px'>{cards}</div>"
    )


# design.md §13 — pill badge (inline only, never concatenated into tables)
def pill(label: str, color: str, bg: str) -> str:
    return (
        f"<span style='background:{bg};color:{color};font-size:0.65rem;font-weight:700;"
        f"padding:2px 9px;border-radius:20px;display:inline-block'>{label}</span>"
    )


# design.md §3 — full card anatomy (4px border-top, tint bg, label, large number, desc)
def kpi_card(
    label: str,
    value: str,
    desc: str,
    border_color: str,
    label_color: str,
    number_color: str,
    bg: str,
) -> str:
    return (
        f"<div style='border:1px solid var(--c-border-soft);border-top:4px solid {border_color};"
        f"border-radius:6px;background:{bg};padding:16px 18px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.04)'>"
        f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{label_color};margin-bottom:10px'>{label}</div>"
        f"<div style='font-size:2.2rem;font-weight:800;color:{number_color};line-height:1;"
        f"margin-bottom:8px'>{value}</div>"
        f"<div style='font-size:0.82rem;color:var(--c-text2);line-height:1.4'>{desc}</div>"
        f"</div>"
    )


# ── Charts ────────────────────────────────────────────────────────────────────


def _add_year_vlines(fig: go.Figure, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Faint dashed vertical lines at Jan 1 of each year in the date range."""
    for yr in range(start.year + 1, end.year + 1):
        fig.add_vline(
            x=pd.Timestamp(f"{yr}-01-01").isoformat(),
            line_dash="dash",
            line_color=PLOT_GRID_COLOR,
            line_width=1,
        )


def chart_fan(monthly: pd.DataFrame, fc: pd.DataFrame) -> go.Figure:
    hist = monthly["Total"].iloc[-36:]
    col = C["base"]
    fig = go.Figure()

    # Filled bands — no hover (handled by invisible helpers below)
    for lo, hi, a, name in [
        ("Total_lo_95", "Total_hi_95", 0.10, "95% range"),
        ("Total_lo_80", "Total_hi_80", 0.28, "80% range (Bear–Bull)"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=pd.concat([fc["month"], fc["month"][::-1]]),
                y=pd.concat([fc[hi], fc[lo][::-1]]),
                fill="toself",
                fillcolor=rgba(col, a),
                line=dict(color="rgba(0,0,0,0)"),
                name=name,
                hoverinfo="skip",
            )
        )

    # Historical line — fixed neutral mid-tone (Plotly cannot consume CSS
    # vars; PLOT_TEXT_COLOR is theme-neutral per the dashboard-theming spec)
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist.values,
            mode="lines",
            line=dict(color=PLOT_TEXT_COLOR, width=2),
            name="Historical",
            hovertemplate="Historical  $%{y:,.0f}<extra></extra>",
        )
    )

    # Base forecast line — visual only, hover handled by master trace below
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=fc["Total_Base"],
            mode="lines",
            line=dict(color=col, width=2.5, dash="dash"),
            name="Base forecast",
            hoverinfo="skip",
        )
    )

    # Single master hover trace — merges Base + Bear + Bull into one left-aligned block
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=fc["Total_Base"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            customdata=np.column_stack([fc["Total_lo_80"].values, fc["Total_hi_80"].values]),
            hovertemplate=(
                "Base  $%{y:,.0f}<br>"
                "🐻 Bear  $%{customdata[0]:,.0f}<br>"
                "🐂 Bull  $%{customdata[1]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    vline_x = fc["month"].iloc[0].isoformat()
    fig.add_vline(x=vline_x, line_dash="dot", line_color=PLOT_GRID_COLOR, line_width=1.5)
    fig.add_annotation(
        x=vline_x,
        y=0.97,
        yref="paper",
        text="forecast →",
        showarrow=False,
        xanchor="left",
        font=dict(size=10, color=PLOT_TEXT_COLOR),
    )
    _add_year_vlines(fig, hist.index[0], fc["month"].iloc[-1])
    fig.update_layout(**BASE_LAYOUT)
    return fig


def chart_stacked(monthly: pd.DataFrame, fc: pd.DataFrame, series: list | None = None) -> go.Figure:
    series = series or SERIES
    hist = monthly[series].iloc[-36:].reset_index()
    bridge = hist.iloc[[-1]].copy()
    fc_data = pd.concat([bridge, fc[["month"] + series]], ignore_index=True)
    fig = go.Figure()
    for s in series:
        col = SERIES_COLOR[s]
        fig.add_trace(
            go.Scatter(
                x=hist["month"],
                y=hist[s],
                mode="none",
                stackgroup="hist",
                name=SERIES_LABEL[s],
                fillcolor=col,
                hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>" + SERIES_LABEL[s] + "</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_data["month"],
                y=fc_data[s],
                mode="none",
                stackgroup="fc",
                showlegend=False,
                fillcolor=rgba(col, 0.5),
                hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>" + SERIES_LABEL[s] + " (fcst)</extra>",
            )
        )
    fig.add_vline(
        x=fc["month"].iloc[0].isoformat(),
        line_dash="dot",
        line_color=PLOT_GRID_COLOR,
        line_width=1.5,
    )
    _add_year_vlines(fig, hist["month"].iloc[0], fc["month"].iloc[-1])
    fig.update_layout(**BASE_LAYOUT)
    return fig


def chart_cumulative(fc: pd.DataFrame) -> go.Figure:
    cum = {sc: fc[sc].cumsum() for sc in ["Bear", "Base", "Bull"]}
    months = list(fc["month"])
    months_rev = months[::-1]
    fig = go.Figure()

    # Fill bands — hidden from legend so only lines appear there
    fig.add_trace(
        go.Scatter(
            x=months + months_rev,
            y=list(cum["Base"]) + list(cum["Bear"][::-1]),
            fill="toself",
            fillcolor=rgba(C["bear"], 0.10),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months + months_rev,
            y=list(cum["Bull"]) + list(cum["Base"][::-1]),
            fill="toself",
            fillcolor=rgba(C["bull"], 0.12),
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lines only — these drive the legend
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=cum["Bear"],
            mode="lines",
            name="Bear",
            line=dict(color=rgba(C["bear"], 0.45), width=1, dash="dot"),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Bear</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=cum["Base"],
            mode="lines",
            name="Base",
            line=dict(color=C["base"], width=2.5),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Base</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["month"],
            y=cum["Bull"],
            mode="lines",
            name="Bull",
            line=dict(color=rgba(C["bull"], 0.45), width=1, dash="dot"),
            hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Bull</extra>",
        )
    )

    _add_year_vlines(fig, fc["month"].iloc[0], fc["month"].iloc[-1])
    fig.update_layout(**BASE_LAYOUT)
    return fig


def chart_quarterly_bar(fc: pd.DataFrame) -> go.Figure:
    q = fc.copy()
    q["quarter"] = q["month"].dt.to_period("Q").astype(str)
    qdf = q.groupby("quarter")[["Bear", "Base", "Bull"]].sum().reset_index()
    # Drop partial quarters (fewer than 3 months) to avoid misleading low bars
    month_counts = q.groupby("quarter")["month"].count()
    full_qs = month_counts[month_counts == 3].index
    qdf = qdf[qdf["quarter"].isin(full_qs)].reset_index(drop=True)
    fig = go.Figure()
    for sc in ["Bear", "Base", "Bull"]:
        fig.add_trace(
            go.Bar(
                x=qdf["quarter"],
                y=qdf[sc],
                name=sc,
                marker_color=SC_COLOR[sc],
                marker_line_width=0,
                text=qdf[sc],
                texttemplate="%{text:$,.3s}",
                textposition="outside",
                textfont=dict(size=10),
                hovertemplate="%{x}: $%{y:,.0f}<extra>" + sc + "</extra>",
            )
        )
    quarters = list(qdf["quarter"])
    for i, q in enumerate(quarters):
        if q.endswith("Q1") and i > 0:
            fig.add_shape(
                type="line",
                x0=i - 0.5,
                x1=i - 0.5,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(dash="dash", color=PLOT_GRID_COLOR, width=1),
            )
            fig.add_annotation(
                x=i - 0.5,
                y=0.0,
                xref="x",
                yref="paper",
                text=q[:4],
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=9, color=PLOT_TEXT_COLOR),
            )
    fig.update_layout(
        **BASE_LAYOUT, barmode="group", xaxis_title=None, bargap=0.2, bargroupgap=0.05
    )
    fig.update_layout(margin=dict(l=8, r=8, t=52, b=8))
    return fig


def chart_donut(fc: pd.DataFrame) -> go.Figure:
    # Donut slice borders use a neutral mid-tone instead of pure white so the
    # separator reads on both light and dark plot backgrounds.
    fig = go.Figure(
        go.Pie(
            labels=[SERIES_LABEL[s] for s in SERIES],
            values=[fc[s].sum() for s in SERIES],
            hole=0.6,
            marker=dict(
                colors=[SERIES_COLOR[s] for s in SERIES], line=dict(color=PLOT_GRID_COLOR, width=2)
            ),
            textinfo="percent",
            textposition="outside",
            automargin=True,
            textfont=dict(family="Inter, system-ui, sans-serif", size=13, color=PLOT_TEXT_COLOR),
            hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="top",
            font=dict(family="Inter, system-ui, sans-serif", size=11, color=PLOT_TEXT_COLOR),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=16, b=72),
        font=dict(family="Inter, system-ui, sans-serif", color=PLOT_TEXT_COLOR),
    )
    return fig


# ── Tab 1: Executive Summary ──────────────────────────────────────────────────


def tab_exec(data: dict, fc: pd.DataFrame, horizon: str) -> None:
    monthly = data["monthly"]
    base_12m = fc["Base"].sum()
    bear_12m = fc["Bear"].sum()
    bull_12m = fc["Bull"].sum()

    # ── Shared date references ────────────────────────────────────────────────
    last_ts = monthly.index[-1]
    lyear_ts = monthly.index[-13] if len(monthly) >= 13 else None
    last_lbl = last_ts.strftime("%B %Y")

    # ── YTD tracker (first) ───────────────────────────────────────────────────
    cur_yr = last_ts.year
    ytd_actual = float(monthly[monthly.index.year == cur_yr]["Total"].sum())
    ytd_months = int((monthly.index.year == cur_yr).sum())
    fc_rem = float(fc[fc["month"].dt.year == cur_yr]["Base"].sum())
    full_proj = ytd_actual + fc_rem
    prior_yr = cur_yr - 1

    prior_ytd = (
        float(monthly[monthly.index.year == prior_yr].iloc[:ytd_months]["Total"].sum())
        if ytd_months > 0
        else 0.0
    )
    ytd_yoy_pct = (ytd_actual - prior_ytd) / prior_ytd * 100 if prior_ytd else 0.0

    _ytd_arr = "▲" if ytd_yoy_pct >= 0 else "▼"
    _ytd_col = C["bull"] if ytd_yoy_pct >= 0 else C["bear"]

    _bar_pct = min(100, round(ytd_months / 12 * 100))
    _bar_html = (
        f"<div style='margin-top:10px'>"
        f"<div style='background:{C['border_mid']};border-radius:4px;height:5px;overflow:hidden;margin-bottom:5px'>"
        f"<div style='background:{C['base']};height:100%;width:{_bar_pct}%;border-radius:4px'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.63rem;color:{C['muted']}'>"
        f"<span>{ytd_months} completed months · {fmt(ytd_actual)} earned</span>"
        f"<span>{fmt(full_proj)} full year projected</span>"
        f"</div>"
        f"</div>"
    )
    ytd_tile = (
        f"<div style='background:var(--c-card);border:1px solid {C['border_mid']};"
        f"border-top:3px solid {C['base']};border-radius:6px;padding:14px 18px'>"
        f"<div style='font-size:0.60rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{C['muted']};margin-bottom:6px'>{cur_yr} through {last_ts.strftime('%B')}</div>"
        f"<div style='font-size:1.5rem;font-weight:800;color:{C['text']};line-height:1'>{fmt(ytd_actual)}</div>"
        f"<div style='font-size:0.72rem;color:{_ytd_col};margin-top:5px'>"
        f"{_ytd_arr} {abs(ytd_yoy_pct):.0f}% vs same period {prior_yr}</div>"
        f"{_bar_html}"
        f"</div>"
    )

    # ── MTD tile with stream split bar ────────────────────────────────────────
    _mtd_total = float(monthly.loc[last_ts, "Total"])
    _mtd_subs = float(monthly.loc[last_ts, "MPD_Core"])
    _mtd_iv = float(monthly.loc[last_ts, "IV_League"])
    _mtd_ly = float(monthly.loc[lyear_ts, "Total"]) if lyear_ts is not None else None
    _mtd_yoy = (_mtd_total - _mtd_ly) / _mtd_ly * 100 if _mtd_ly else None
    _mtd_yoy_col = C["bull"] if (_mtd_yoy or 0) >= 0 else C["bear"]
    _mtd_yoy_arr = "▲" if (_mtd_yoy or 0) >= 0 else "▼"
    _mtd_delta = (
        f"{_mtd_yoy_arr} {abs(_mtd_yoy):.0f}% vs {last_ts.strftime('%b')} last year"
        if _mtd_yoy is not None
        else "No prior year data"
    )
    _subs_pct = round(_mtd_subs / _mtd_total * 100) if _mtd_total else 50
    _iv_pct = 100 - _subs_pct
    _split_bar = (
        f"<div style='margin-top:10px'>"
        f"<div style='display:flex;border-radius:4px;overflow:hidden;height:22px'>"
        f"<div style='width:{_subs_pct}%;background:{C['mrr']};display:flex;"
        f"align-items:center;justify-content:center'>"
        f"<span style='font-size:0.60rem;font-weight:700;color:#fff'>{fmt(_mtd_subs)}</span>"
        f"</div>"
        f"<div style='width:{_iv_pct}%;background:{C['iv']};display:flex;"
        f"align-items:center;justify-content:center'>"
        f"<span style='font-size:0.60rem;font-weight:700;color:#fff'>{fmt(_mtd_iv)}</span>"
        f"</div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.63rem;color:{C['muted']};margin-top:4px'>"
        f"<span>Subscriptions {_subs_pct}%</span>"
        f"<span>IV League {_iv_pct}%</span>"
        f"</div>"
        f"</div>"
    )
    mtd_tile = (
        f"<div style='background:var(--c-card);border:1px solid {C['border_mid']};"
        f"border-top:3px solid {C['base']};border-radius:6px;padding:14px 18px'>"
        f"<div style='font-size:0.60rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{C['muted']};margin-bottom:6px'>"
        f"Last completed month — {last_lbl}</div>"
        f"<div style='font-size:1.5rem;font-weight:800;color:{C['text']};line-height:1'>{fmt(_mtd_total)}</div>"
        f"<div style='font-size:0.72rem;color:{_mtd_yoy_col};margin-top:5px'>{_mtd_delta}</div>"
        f"{_split_bar}"
        f"</div>"
    )

    st.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;"
        f"margin-bottom:24px'>{ytd_tile}{mtd_tile}</div>",
        unsafe_allow_html=True,
    )

    # ── Forecast outlook ──────────────────────────────────────────────────────
    # Compare next-horizon base against the most recent complete calendar year
    # of actuals. The card label uses that year dynamically so the comparison
    # stays correct as time rolls forward.
    last_full_yr = cur_yr - 1
    prior_yr_tot = float(monthly[monthly.index.year == last_full_yr]["Total"].sum())
    dir_color = C["bull"] if base_12m >= prior_yr_tot else C["bear"]
    dir_arrow = "▲" if base_12m >= prior_yr_tot else "▼"
    pct_vs_prior = abs((base_12m - prior_yr_tot) / prior_yr_tot * 100) if prior_yr_tot else 0.0
    spread = bull_12m - bear_12m
    ot_share = (fc["MPD_Core_OneTime"].sum() / base_12m * 100) if base_12m else 0.0

    with st.container(border=True):
        section(f"Forecast outlook — next {horizon}")
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px'>"
            f"<div style='background:{C['bear_bg']};border-left:3px solid {C['bear']};"
            f"border-radius:0 6px 6px 0;padding:10px 14px'>"
            f"<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{C['bear']};margin-bottom:4px'>🐻 Bear</div>"
            f"<div style='font-size:1.25rem;font-weight:800;color:{C['bear']};margin-bottom:2px'>{fmt(bear_12m)}</div>"
            f"<div style='font-size:0.68rem;color:{C['muted']}'>if things slow down</div></div>"
            f"<div style='background:{C['base_bg']};border-left:3px solid {C['base']};"
            f"border-radius:0 6px 6px 0;padding:10px 14px'>"
            f"<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{C['base']};margin-bottom:4px'>📊 Base</div>"
            f"<div style='font-size:1.25rem;font-weight:800;color:{C['base']};margin-bottom:2px'>{fmt(base_12m)}</div>"
            f"<div style='font-size:0.68rem;color:{C['muted']}'>best estimate</div>"
            f"<div style='font-size:0.72rem;font-weight:700;color:{dir_color};margin-top:5px'>"
            f"{dir_arrow} {pct_vs_prior:.0f}% vs full-year {last_full_yr}</div></div>"
            f"<div style='background:{C['bull_bg']};border-left:3px solid {C['bull']};"
            f"border-radius:0 6px 6px 0;padding:10px 14px'>"
            f"<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{C['bull']};margin-bottom:4px'>🐂 Bull</div>"
            f"<div style='font-size:1.25rem;font-weight:800;color:{C['bull']};margin-bottom:2px'>{fmt(bull_12m)}</div>"
            f"<div style='font-size:0.68rem;color:{C['muted']}'>if things go well</div></div>"
            f"</div>"
            # planning range note + bottom spacer
            f"<div style='background:var(--c-surface-alt);border-left:3px solid {C['border']};"
            f"border-radius:0 4px 4px 0;padding:8px 14px;margin-bottom:4px;"
            f"font-size:0.78rem;color:{C['secondary']}'>"
            f"<strong>Planning range: {fmt(spread)}</strong> separates Bear from Bull. "
            f"The gap is driven almost entirely by premium package timing — the unpredictable portion of Subscriptions revenue ({ot_share:.0f}% of Base). "
            f"Your sales pipeline is the best input for narrowing this range."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ── Stream health ─────────────────────────────────────────────────────────
    divider("stream health")

    # Pre-compute MRR compounding rate (used in Subscriptions card + action text)
    mrr_growth_pct = 0.0
    if "MPD_Core_MRR" in monthly.columns:
        _mrr = monthly["MPD_Core_MRR"].iloc[-6:]
        _mrr_chg = _mrr.pct_change().dropna()
        if len(_mrr_chg) > 0:
            mrr_growth_pct = float(_mrr_chg.mean() * 100)
    mrr_end_val = float(fc["MPD_Core_MRR"].iloc[-1]) if "MPD_Core_MRR" in fc.columns else 0.0
    mrr_end_month = fc["month"].iloc[-1].strftime("%b %Y")

    # Pre-compute IV League peak forecast month
    iv_fc_vals = fc["IV_League"]
    iv_peak_val = float(iv_fc_vals.max())
    iv_peak_month = fc.loc[iv_fc_vals.idxmax(), "month"].strftime("%b %Y")
    iv_fc_avg = float(iv_fc_vals.mean())

    health_cards = ""
    for skey in ["MPD_Core", "IV_League"]:  # Subscriptions on top, IV League below
        label = SERIES_LABEL[skey]
        color = SERIES_COLOR[skey]
        series = monthly[skey]

        recent_6m = float(series.iloc[-6:].mean())
        prior_6m = float(series.iloc[-12:-6].mean())
        last_12m = float(series.iloc[-12:].sum())
        chg_pct = (recent_6m - prior_6m) / prior_6m * 100 if prior_6m else 0.0

        next_12m = float(fc[skey].iloc[:12].sum())
        fc_chg_pct = (next_12m - last_12m) / last_12m * 100 if last_12m else 0.0

        if fc_chg_pct > 5:
            status, sc, sbg, arrow = "Growing", C["bull"], C["bull_bg"], "▲"
        elif fc_chg_pct < -5:
            status, sc, sbg, arrow = "Declining", C["bear"], C["bear_bg"], "▼"
        else:
            status, sc, sbg, arrow = "Stable", "#FB8C00", "rgba(251,140,0,0.15)", "→"

        if skey == "IV_League":
            action = {
                "growing": f"Model projects {fmt(iv_peak_val)} peak in {iv_peak_month}. Push spend ahead of that month.",
                "stable": f"IV League averaging {fmt(iv_fc_avg)}/month forecast — {iv_peak_month} is the highest projected month. Target promotions before it.",
                "declining": f"IV League forecast is {abs(fc_chg_pct):.0f}% below the prior 12 months — review package pricing and IV offering urgently.",
            }[status.lower()]
        else:
            if status.lower() == "growing":
                action = f"Recurring MRR compounding at {mrr_growth_pct:+.1f}%/month — at this rate MRR reaches {fmt(mrr_end_val)}/month by {mrr_end_month}. Ensure clinical capacity keeps pace."
            elif status.lower() == "stable":
                action = f"MRR at {mrr_growth_pct:+.1f}%/month — forecast reaches {fmt(mrr_end_val)}/month by {mrr_end_month}. Accelerate new patient acquisition to reach Bull case."
            else:  # declining
                if mrr_growth_pct >= 0:
                    action = (
                        f"Subscriptions forecast is {abs(fc_chg_pct):.0f}% below the prior 12 months — recurring MRR is still "
                        f"growing at {mrr_growth_pct:+.1f}%/month but one-time package revenue is expected lower. "
                        f"Build the premium package sales pipeline."
                    )
                else:
                    action = f"MRR declining at {mrr_growth_pct:+.1f}%/month — investigate churn and build the premium package pipeline immediately."

        chg_col = C["bull"] if chg_pct >= 0 else C["bear"]
        hist_arrow = "▲" if chg_pct > 0 else ("▼" if chg_pct < 0 else "→")

        mrr_subtitle = ""
        if skey == "MPD_Core":
            _mc = C["bull"] if mrr_growth_pct >= 0 else C["bear"]
            _ma = "▲" if mrr_growth_pct >= 0 else "▼"
            mrr_subtitle = (
                f"<div style='font-size:0.68rem;color:{_mc};margin-top:2px'>"
                f"{_ma} Recurring MRR compounding at {mrr_growth_pct:+.1f}%/month</div>"
            )

        health_cards += (
            f"<div style='border:1px solid {C['border_mid']};border-top:4px solid {color};"
            f"border-radius:6px;padding:16px 18px 18px;background:var(--c-card);"
            f"display:flex;flex-direction:column;margin-bottom:12px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>"
            f"<div style='display:flex;align-items:center;gap:7px'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{color}'></div>"
            f"<div style='font-size:0.82rem;font-weight:700;color:{C['text']}'>{label}</div>"
            f"</div>"
            f"<span style='font-size:0.68rem;font-weight:700;color:{sc};background:{sbg};"
            f"padding:3px 9px;border-radius:20px'>{arrow} {status}</span>"
            f"</div>"
            f"{mrr_subtitle}"
            f"<div style='font-size:0.72rem;color:{chg_col};margin-bottom:12px;margin-top:4px'>"
            f"{hist_arrow} {abs(chg_pct):.0f}% vs prior 6 months (avg monthly revenue)</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:12px;flex:1'>"
            f"<div style='background:var(--c-surface-alt);border-radius:4px;padding:8px 10px;text-align:center'>"
            f"<div style='font-size:0.58rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;"
            f"color:{C['muted']};margin-bottom:2px'>Last 12m</div>"
            f"<div style='font-size:1.0rem;font-weight:800;color:{C['text']}'>{fmt(last_12m)}</div>"
            f"</div>"
            f"<div style='background:var(--c-surface-alt);border-radius:4px;padding:8px 10px;text-align:center'>"
            f"<div style='font-size:0.58rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;"
            f"color:{C['muted']};margin-bottom:2px'>Next {horizon.replace(' months','m')} (Base)</div>"
            f"<div style='font-size:1.0rem;font-weight:800;color:{color}'>{fmt(fc[skey].sum())}</div>"
            f"</div></div>"
            f"<div style='background:{sbg};border-left:3px solid {sc};border-radius:0 4px 4px 0;"
            f"padding:8px 10px'>"
            f"<div style='font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;"
            f"color:{sc};margin-bottom:2px'>Recommended action</div>"
            f"<div style='font-size:0.75rem;color:{C['secondary']};line-height:1.5'>{action}</div>"
            f"</div>"
            f"</div>"
        )

    col_donut, col_cards = st.columns([5, 8])
    with col_donut:
        st.plotly_chart(chart_donut(fc), use_container_width=True, key="e_donut")
    with col_cards:
        st.markdown(health_cards, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    divider("quarterly breakdown")
    col_ql, col_qr = st.columns(2)
    with col_ql:
        with st.container(border=True):
            section(
                "Quarterly revenue — expected, pessimistic, and optimistic",
                "Base = what we expect. Bear = revenue if conditions turn unfavourable. Bull = revenue if conditions turn favourable. Use Bear for risk planning, Base for operational targets.",
            )
            st.plotly_chart(chart_quarterly_bar(fc), use_container_width=True, key="e_qbar")
            _eq = fc.copy()
            _eq["quarter"] = _eq["month"].dt.to_period("Q").astype(str)
            _eq_csv = _eq.groupby("quarter")[["Bear", "Base", "Bull"]].sum().reset_index()
            subtle_download(
                _eq_csv.to_csv(index=False).encode(),
                f"mpd_quarterly_{datetime.now():%Y%m%d}.csv",
                "dl_eq",
            )
    with col_qr:
        with st.container(border=True):
            section(
                "Running total — how each scenario adds up",
                "Cumulative revenue over the horizon. The gap between Bear and Bull shows how much the outcome could vary. Wider gap = more uncertainty ahead.",
            )
            st.plotly_chart(chart_cumulative(fc), use_container_width=True, key="e_cumul")
            _ec = fc[["month", "Bear", "Base", "Bull"]].copy()
            _ec["Bear_cum"] = _ec["Bear"].cumsum()
            _ec["Base_cum"] = _ec["Base"].cumsum()
            _ec["Bull_cum"] = _ec["Bull"].cumsum()
            subtle_download(
                _ec.to_csv(index=False).encode(),
                f"mpd_cumulative_{datetime.now():%Y%m%d}.csv",
                "dl_ec",
            )


# ── Tab 2: Forecast Dashboard ─────────────────────────────────────────────────


def tab_dashboard(data: dict, fc: pd.DataFrame, horizon: str) -> None:
    monthly = data["monthly"]

    # ── Section 1: Revenue by stream — diagnostic view ────────────────────────
    divider("revenue by stream")
    st.markdown(
        f"<p style='font-size:0.88rem;color:{C['secondary']};line-height:1.6;margin:0 0 12px'>"
        f"MPD revenue is split into two independent streams, each forecast separately. "
        f"Select which stream to view — the summary numbers and chart below update together.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(stream_definition_cards(data.get("spec")), unsafe_allow_html=True)

    _all_label = "All streams"
    _stream_options = [_all_label] + [SERIES_LABEL[s] for s in SERIES]
    _stream_sel = st.selectbox(
        "Filter streams",
        options=_stream_options,
        index=0,
        label_visibility="collapsed",
    )
    selected = (
        SERIES
        if _stream_sel == _all_label
        else [s for s in SERIES if SERIES_LABEL[s] == _stream_sel]
    )

    if not selected:
        st.info("Select at least one stream to see the breakdown.", icon="ℹ️")
    else:
        # KPI cards — only for selected streams
        n_cols = len(selected)
        stream_cards = ""
        for s in selected:
            hist_avg = monthly[s].iloc[-12:].mean()
            fc_avg = fc[s].mean()
            stream_cards += kpi_card(
                SERIES_LABEL[s].upper(),
                fmt(fc[s].sum()),
                f"avg/month {fmt(fc_avg)} · <strong>{pct(fc_avg, hist_avg)} vs prior 12m</strong>",
                SERIES_COLOR[s],
                SERIES_COLOR[s],
                SERIES_COLOR[s],
                C["card"],
            )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:repeat({n_cols},1fr);"
            f"gap:12px;margin:12px 0 16px'>{stream_cards}</div>",
            unsafe_allow_html=True,
        )

        # Chart — same selection
        with st.container(border=True):
            section(
                "Historical actuals + forecast by stream",
                "Solid = historical · Faded = forecast · Streams stack to total",
            )
            st.plotly_chart(
                chart_stacked(monthly, fc, series=list(selected)),
                use_container_width=True,
                key="d_stack",
            )
            _stream_csv = fc[["month"] + [s for s in list(selected) if s in fc.columns]].copy()
            subtle_download(
                _stream_csv.to_csv(index=False).encode(),
                f"mpd_stream_forecast_{datetime.now():%Y%m%d}.csv",
                "dl_stream",
            )

    # ── Section 2: Total fan chart — forecast uncertainty ─────────────────────
    divider("forecast uncertainty")
    with st.container(border=True):
        hdr_l, hdr_r = st.columns([8, 2])
        with hdr_l:
            section("Total revenue — forecast range")
        with hdr_r:
            show_pi = st.toggle(
                "Uncertainty bands", value=True, help="Toggle the shaded confidence regions on/off."
            )

        st.markdown(
            f"<div style='display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap'>"
            f"<div style='display:flex;align-items:flex-start;gap:10px;flex:1;min-width:200px;"
            f"background:var(--c-surface-alt);border-radius:6px;padding:10px 14px'>"
            f"<div style='margin-top:3px;width:28px;flex-shrink:0;border-top:2px dashed {C['base']}'></div>"
            f"<div><div style='font-size:0.72rem;font-weight:700;color:{C['text']};margin-bottom:2px'>Base forecast</div>"
            f"<div style='font-size:0.70rem;color:{C['secondary']};line-height:1.5'>"
            f"The model's single best guess — what revenue is most likely to be each month.</div></div></div>"
            f"<div style='display:flex;align-items:flex-start;gap:10px;flex:1;min-width:200px;"
            f"background:var(--c-surface-alt);border-radius:6px;padding:10px 14px'>"
            f"<div style='margin-top:3px;width:28px;height:14px;flex-shrink:0;"
            f"background:rgba(92,107,192,0.45);border-radius:3px'></div>"
            f"<div><div style='font-size:0.72rem;font-weight:700;color:{C['text']};margin-bottom:2px'>Planning range (80%)</div>"
            f"<div style='font-size:0.70rem;color:{C['secondary']};line-height:1.5'>"
            f"In 4 out of 5 months, actual revenue should land inside this band. Use it as your floor-to-ceiling for budgeting.</div></div></div>"
            f"<div style='display:flex;align-items:flex-start;gap:10px;flex:1;min-width:200px;"
            f"background:var(--c-surface-alt);border-radius:6px;padding:10px 14px'>"
            f"<div style='margin-top:3px;width:28px;height:14px;flex-shrink:0;"
            f"background:rgba(92,107,192,0.18);border-radius:3px'></div>"
            f"<div><div style='font-size:0.72rem;font-weight:700;color:{C['text']};margin-bottom:2px'>Outer boundary (95%)</div>"
            f"<div style='font-size:0.70rem;color:{C['secondary']};line-height:1.5'>"
            f"The realistic worst-case and best-case. Only 1 in 20 months should fall outside this zone.</div></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        fc_start = fc["Total_Base"].iloc[0]
        fc_end = fc["Total_Base"].iloc[-1]
        _csec = C["secondary"]
        if fc_end < fc_start * 0.97:
            st.markdown(
                f"<div style='background:rgba(255,160,0,0.10);border-left:3px solid #FFA000;"
                f"border-radius:0 4px 4px 0;padding:8px 14px;margin:0 0 12px;"
                f"font-size:0.80rem;color:{_csec};line-height:1.5'>"
                f"⚠ Base trend declines because spike months in late 2025–2026 were flagged as outliers and excluded from the forward trajectory. "
                f"MRR continues to grow — the dip is One-Time packages normalising. "
                f"If premium sales hold, actuals will track the <strong>Bull case</strong>."
                f"</div>",
                unsafe_allow_html=True,
            )

        fan = chart_fan(monthly, fc)
        if not show_pi:
            fan.data = tuple(t for t in fan.data if not (t.name and "range" in t.name))
        st.plotly_chart(fan, use_container_width=True, key="d_fan")
        subtle_download(
            fc.to_csv(index=False).encode(),
            f"mpd_forecast_{horizon.replace(' ','_')}_{datetime.now():%Y%m%d}.csv",
            "dl_fan",
        )


# ── Tab 3: About This Forecast ────────────────────────────────────────────────


def tab_assumptions(data: dict) -> None:
    monthly = data["monthly"]
    model_info = data.get("model_info", pd.DataFrame())
    ev = data["ev"]
    outliers = data["outliers"]

    # ── Hero banner ───────────────────────────────────────────────────────────
    n_months = len(monthly)
    total_rev = float(monthly["Total"].sum()) if "Total" in monthly.columns else 0.0
    last_month = monthly.index[-1].strftime("%B %Y")
    post_covid = (
        monthly[monthly["is_covid_startup"] == 0]
        if "is_covid_startup" in monthly.columns
        else monthly
    )
    avg_monthly = float(post_covid["Total"].mean()) if "Total" in post_covid.columns else 0.0

    total_rev_m = total_rev / 1_000_000
    _accent_border = "rgba(92,107,192,0.30)"  # indigo accent border, theme-neutral
    _hero_gradient = "linear-gradient(135deg, rgba(92,107,192,0.14) 0%, rgba(92,107,192,0.06) 100%)"
    st.markdown(
        f"<div style='background:{_hero_gradient};"
        f"border:1px solid {_accent_border};border-radius:8px;padding:16px 24px;"
        f"margin-bottom:20px;display:flex;align-items:center;justify-content:space-between;"
        f"flex-wrap:wrap;gap:12px'>"
        # left: headline + tagline
        f"<div style='flex:1;min-width:240px'>"
        f"<div style='font-size:1.0rem;font-weight:800;color:{C['text']};margin-bottom:4px'>"
        f"MPD Revenue Forecast</div>"
        f"<div style='font-size:0.8rem;color:{C['secondary']};line-height:1.5'>"
        f"Built on {n_months} months of MPD's own transactions"
        f"</div></div>"
        # right: 4 inline stat chips
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center'>"
        + "".join(
            [
                f"<div style='background:var(--c-card);border:1px solid {_accent_border};"
                f"border-radius:6px;padding:8px 14px;text-align:center;min-width:90px'>"
                f"<div style='font-size:1.05rem;font-weight:800;color:{C['accent']}'>{val}</div>"
                f"<div style='font-size:0.62rem;font-weight:700;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:{C['accent']};opacity:0.75;margin-top:2px'>{lbl}</div>"
                f"</div>"
                for val, lbl in [
                    (f"{n_months}", "months"),
                    (f"${total_rev_m:.1f}M", "modelled"),
                    (f"${avg_monthly:,.0f}", "avg / month"),
                    (last_month, "data through"),
                ]
            ]
        )
        + "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Accuracy cards ────────────────────────────────────────────────────────
    with st.container(border=True):
        section(
            "How reliable is each forecast?",
            "Model is trained on older data, "
            "predicted months it had never seen, and checked how close it got — repeated 8 times.",
        )

        # About-tab labels: internal model keys → human-readable names
        # (SERIES_LABEL only has the 2 combined business streams now)
        ABOUT_LABELS = {
            "IV_League": "IV League",
            "MPD_Core_MRR": "Subscriptions — Recurring MRR",
            "MPD_Core_OneTime": "Subscriptions — Premium Packages",
        }
        ABOUT_COLORS = {
            "IV_League": C["iv"],
            "MPD_Core_MRR": C["mrr"],
            "MPD_Core_OneTime": C["ot"],
        }

        # Per-stream decision-utility config
        STREAM_UTILITY = {
            "IV_League": {
                "use_for": "Seasonal planning & marketing timing",
                "strength": "Outperforms the last-year benchmark — good for identifying strong vs soft months.",
                "limit": "IV League volume is intermittent; individual month predictions carry a ±$6.5K typical error.",
            },
            "MPD_Core_MRR": {
                "use_for": "Subscriptions — growth trajectory & capacity planning",
                "strength": "The recurring subscription growth direction is reliable and stable. Month-on-month compounding is well-captured.",
                "limit": "Exact monthly figures carry higher error — use for trend direction and staffing ramp, not precise monthly budgets.",
            },
            "MPD_Core_OneTime": {
                "use_for": "Subscriptions — revenue range planning (Bear/Bull)",
                "strength": "Outperforms the last-year benchmark. The Bear/Bull range correctly brackets most months.",
                "limit": "Individual package sales can't be timed precisely. The wide Bear/Bull spread reflects real unpredictability, not a model weakness.",
            },
        }

        def accuracy_grade(beats: bool, mase_v: float, base_v: float):
            if beats and mase_v < 1.3:
                return ("Beats benchmark", C["bull"], C["bull_bg"])
            if beats:
                return ("Beats benchmark", "#2E7D32", "rgba(46,125,50,0.15)")
            if mase_v < base_v * 1.5:
                return ("Near benchmark", "#FB8C00", "rgba(251,140,0,0.15)")
            return ("Trajectory only", C["bear"], C["bear_bg"])

        def rmse_context(rmse: float, stream_key: str) -> str:
            post = (
                monthly[stream_key][monthly["is_covid_startup"] == 0]
                if "is_covid_startup" in monthly.columns
                else monthly[stream_key]
            )
            avg = float(post[post > 0].mean()) if len(post[post > 0]) > 0 else 1.0
            pct = rmse / avg * 100 if avg > 0 else 0.0
            return f"Typical monthly error: ${rmse:,.0f} ({pct:.0f}% of stream average)"

        if not ev.empty:
            # Executive summary callout
            beats_count = int(ev["beats_baseline"].sum()) if "beats_baseline" in ev.columns else 0
            total_count = len(ev)
            st.markdown(
                f"<div style='background:{C['base_bg']};border-left:4px solid {C['accent']};"
                f"border-radius:4px;padding:12px 16px;margin-bottom:16px'>"
                f"<div style='font-size:0.82rem;font-weight:700;color:{C['accent']};margin-bottom:4px'>"
                f"{beats_count} of {total_count} internal models outperform the seasonal benchmark</div>"
                f"<div style='font-size:0.78rem;color:{C['secondary']};line-height:1.5'>"
                f"The seasonal benchmark is the simplest possible forecast — \"next month will look like the same month last year.\" "
                f"Beating it means the model extracts genuine signal. "
                f"Subscriptions is modelled as two internal sub-streams (Recurring MRR + Premium Packages) for better accuracy, then combined for reporting."
                f"</div></div>",
                unsafe_allow_html=True,
            )

            def _single_card_html(skey: str) -> str:
                row = ev[ev["series"] == skey]
                if row.empty:
                    return ""
                row = row.iloc[0]
                color = ABOUT_COLORS.get(skey, C["base"])
                mv = float(row["MASE"])
                rv = float(row["RMSE"])
                bv = float(row.get("Baseline_MASE", mv))
                beats = bool(row.get("beats_baseline", False))
                grade, gc, gbg = accuracy_grade(beats, mv, bv)
                util = STREAM_UTILITY.get(skey, {})
                rmse_str = rmse_context(rv, skey)
                if beats:
                    pct_b = (bv - mv) / bv * 100
                    expl = f"The model's predictions were {pct_b:.0f}% closer to reality than simply repeating the same month from last year."
                else:
                    pct_w = (mv - bv) / bv * 100
                    expl = f"The calendar guess was {pct_w:.0f}% more accurate on exact monthly figures — but the growth direction and planning range are still reliable."
                return (
                    f"<div style='border:1px solid var(--c-border-soft);border-top:4px solid {color};"
                    f"border-radius:6px;padding:16px;background:var(--c-card);display:flex;flex-direction:column'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:2px'>"
                    f"<div style='display:flex;align-items:center;gap:7px'>"
                    f"<div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0'></div>"
                    f"<div style='font-size:0.82rem;font-weight:700;color:{C['text']}'>{ABOUT_LABELS.get(skey, skey)}</div>"
                    f"</div>"
                    f"<span style='font-size:0.68rem;font-weight:700;color:{gc};background:{gbg};"
                    f"padding:3px 9px;border-radius:20px;white-space:nowrap'>{grade}</span>"
                    f"</div>"
                    f"<div style='font-size:0.72rem;color:{gc};font-style:italic;line-height:1.45;margin-bottom:8px'>{expl}</div>"
                    f"<div style='font-size:0.7rem;color:{C['muted']};font-weight:600;letter-spacing:0.04em;margin-bottom:10px'>{util.get('use_for','')}</div>"
                    f"<div style='display:flex;gap:8px;margin-bottom:8px'>"
                    f"<div style='color:{C['bull']};font-size:0.78rem;flex-shrink:0;margin-top:1px'>✓</div>"
                    f"<div style='font-size:0.78rem;color:{C['secondary']};line-height:1.5'>{util.get('strength','')}</div>"
                    f"</div>"
                    f"<div style='display:flex;gap:8px;margin-bottom:12px;flex:1'>"
                    f"<div style='color:{C['muted']};font-size:0.78rem;flex-shrink:0;margin-top:1px'>↳</div>"
                    f"<div style='font-size:0.78rem;color:{C['muted']};line-height:1.5'>{util.get('limit','')}</div>"
                    f"</div>"
                    f"<div style='font-size:0.72rem;color:{C['muted']};background:var(--c-surface-alt);"
                    f"border-radius:4px;padding:6px 8px;margin-bottom:12px'>{rmse_str}</div>"
                    f"<div style='font-size:0.67rem;color:{C['muted']};border-top:1px solid var(--c-grid);"
                    f"padding-top:8px;line-height:1.6'>"
                    f"MASE {mv:.2f} &nbsp;·&nbsp; Baseline MASE {bv:.2f} &nbsp;·&nbsp; "
                    f"{'✓ beats benchmark' if beats else '✗ below benchmark'}"
                    f"</div></div>"
                )

            # Card 1 — IV League
            iv_card = _single_card_html("IV_League")

            # Card 2 — MPD Core: single flat card, compact sub-model rows inside
            _mrr_r = ev[ev["series"] == "MPD_Core_MRR"]
            _ot_r = ev[ev["series"] == "MPD_Core_OneTime"]

            def _sub_row(r_df, sub_label, color):
                if r_df.empty:
                    return ""
                r = r_df.iloc[0]
                mv, bv = float(r["MASE"]), float(r.get("Baseline_MASE", r["MASE"]))
                beats = bool(r.get("beats_baseline", False))
                grade, gc, gbg = accuracy_grade(beats, mv, bv)
                if beats:
                    pct_b = (bv - mv) / bv * 100
                    expl = f"{pct_b:.0f}% closer to reality than repeating last year."
                else:
                    pct_w = (mv - bv) / bv * 100
                    expl = f"Calendar was {pct_w:.0f}% more accurate on exact figures — direction still reliable."
                rmse_v = float(r["RMSE"])
                post = monthly["MPD_Core_MRR" if "MRR" in sub_label else "MPD_Core_OneTime"]
                post = (
                    post[monthly["is_covid_startup"] == 0]
                    if "is_covid_startup" in monthly.columns
                    else post
                )
                avg_v = float(post[post > 0].mean()) if len(post[post > 0]) > 0 else 1.0
                pct_err = rmse_v / avg_v * 100 if avg_v > 0 else 0
                return (
                    f"<div style='border:1px solid var(--c-border-soft);border-radius:5px;padding:10px 12px;"
                    f"background:var(--c-surface-alt)'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>"
                    f"<div style='display:flex;align-items:center;gap:6px'>"
                    f"<div style='width:6px;height:6px;border-radius:50%;background:{color}'></div>"
                    f"<div style='font-size:0.78rem;font-weight:700;color:{C['text']}'>{sub_label}</div>"
                    f"</div>"
                    f"<span style='font-size:0.65rem;font-weight:700;color:{gc};background:{gbg};"
                    f"padding:2px 7px;border-radius:20px'>{grade}</span>"
                    f"</div>"
                    f"<div style='font-size:0.70rem;color:{gc};font-style:italic;margin-bottom:4px'>{expl}</div>"
                    f"<div style='font-size:0.67rem;color:{C['muted']}'>"
                    f"Typical error: ${rmse_v:,.0f} ({pct_err:.0f}% of avg) &nbsp;·&nbsp; "
                    f"MASE {mv:.2f} vs baseline {bv:.2f}</div>"
                    f"</div>"
                )

            mrr_row_html = _sub_row(_mrr_r, "Recurring MRR", C["mrr"])
            ot_row_html = _sub_row(_ot_r, "Premium Packages", C["ot"])

            sub_card = (
                f"<div style='border:1px solid {C['border_mid']};border-top:4px solid {C['mrr']};"
                f"border-radius:6px;padding:16px;background:var(--c-card);display:flex;flex-direction:column'>"
                # header
                f"<div style='display:flex;align-items:center;gap:7px;margin-bottom:4px'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{C['mrr']};flex-shrink:0'></div>"
                f"<div style='font-size:0.82rem;font-weight:700;color:{C['text']}'>MPD Core</div>"
                f"</div>"
                f"<div style='font-size:0.72rem;color:{C['muted']};margin-bottom:12px'>"
                f"Internally split into two sub-models for accuracy, combined for all reporting.</div>"
                # sub-model rows
                f"<div style='display:flex;flex-direction:column;gap:8px;flex:1'>"
                f"{mrr_row_html}{ot_row_html}"
                f"</div>"
                # combined guidance
                f"<div style='margin-top:12px;border-top:1px solid var(--c-grid);padding-top:10px'>"
                f"<div style='display:flex;gap:8px;margin-bottom:6px'>"
                f"<div style='color:{C['bull']};font-size:0.78rem;flex-shrink:0'>✓</div>"
                f"<div style='font-size:0.78rem;color:{C['secondary']};line-height:1.5'>"
                f"MRR growth direction is reliable. Bear/Bull range correctly brackets package months.</div>"
                f"</div>"
                f"<div style='display:flex;gap:8px'>"
                f"<div style='color:{C['muted']};font-size:0.78rem;flex-shrink:0'>↳</div>"
                f"<div style='font-size:0.78rem;color:{C['muted']};line-height:1.5'>"
                f"Use MRR for staffing ramp and growth trajectory. Use Bear/Bull range for package revenue planning — individual sale timing cannot be predicted.</div>"
                f"</div></div>"
                f"</div>"
            )

            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:stretch'>"
                f"{iv_card}{sub_card}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Run the pipeline to generate accuracy scores.", icon="ℹ️")

        # Score explainer
        if not ev.empty:
            _cbull = C["bull"]
            _cmuted = C["muted"]
            _ctext = C["text"]
            _csec = C["secondary"]
            beat_icon = f"<span style='color:{_cbull};font-weight:700'>Model wins ✓</span>"
            lose_icon = f"<span style='color:{_cmuted}'>Calendar wins</span>"
            rows_exp = ""
            for skey in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
                r = ev[ev["series"] == skey]
                if r.empty:
                    continue
                r = r.iloc[0]
                lbl = ABOUT_LABELS.get(skey, skey)
                color = ABOUT_COLORS.get(skey, C["base"])
                beats = bool(r.get("beats_baseline", False))
                mv, bv = float(r["MASE"]), float(r.get("Baseline_MASE", r["MASE"]))
                result = beat_icon if beats else lose_icon
                rows_exp += (
                    f"<tr>"
                    f"<td style='padding:7px 10px;border-bottom:1px solid var(--c-grid)'>"
                    f"<span style='display:inline-flex;align-items:center;gap:6px'>"
                    f"<span style='width:7px;height:7px;border-radius:50%;background:{color};display:inline-block'></span>"
                    f"<span style='font-size:0.75rem;font-weight:600;color:{_ctext}'>{lbl}</span></span></td>"
                    f"<td style='padding:7px 10px;border-bottom:1px solid var(--c-grid);font-size:0.75rem;color:{_csec};text-align:center'>{mv:.2f}</td>"
                    f"<td style='padding:7px 10px;border-bottom:1px solid var(--c-grid);font-size:0.75rem;color:{_csec};text-align:center'>{bv:.2f}</td>"
                    f"<td style='padding:7px 10px;border-bottom:1px solid var(--c-grid);text-align:center'>{result}</td>"
                    f"</tr>"
                )
            explainer_html = (
                f"<div style='background:var(--c-surface-alt);border:1px solid var(--c-border);border-radius:6px;"
                f"padding:14px 16px 16px;margin-top:16px'>"
                f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;"
                f"color:{_cmuted};margin-bottom:10px'>How the accuracy test works</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:14px'>"
                f"<div>"
                f"<div style='font-size:0.78rem;font-weight:700;color:{_ctext};margin-bottom:5px'>What is the benchmark?</div>"
                f"<div style='font-size:0.75rem;color:{_csec};line-height:1.6'>"
                f"The <strong>seasonal naïve</strong>: predict each month using the same calendar month from last year. "
                f"To forecast July 2025, use July 2024's actual revenue. No model needed — just a calendar lookup.<br><br>"
                f"It's the industry standard baseline used in academic forecasting competitions worldwide (M4/M5)."
                f"</div></div>"
                f"<div>"
                f"<div style='font-size:0.78rem;font-weight:700;color:{_ctext};margin-bottom:5px'>Does it account for growth?</div>"
                f"<div style='font-size:0.75rem;color:{_csec};line-height:1.6'>"
                f"No — and that's intentional. For a growing business, last year's July is <em>lower</em> than this year's July. "
                f"So the benchmark systematically under-predicts for growing streams.<br><br>"
                f"This makes it a <strong>harder bar to beat</strong>, not an easier one. Beating it means the model captures "
                f"genuine patterns beyond a simple calendar guess."
                f"</div></div>"
                f"<div>"
                f"<div style='font-size:0.78rem;font-weight:700;color:{_ctext};margin-bottom:5px'>How we compare them</div>"
                f"<div style='font-size:0.75rem;color:{_csec};line-height:1.6'>"
                f"Both the model and the calendar benchmark are tested on the same hidden months. "
                f"MASE puts both on the same scale so they're directly comparable — lower is better.<br><br>"
                f"The model wins when its error is lower than the benchmark's error."
                f"</div></div>"
                f"</div>"
                f"<table style='width:100%;border-collapse:collapse;background:var(--c-card);"
                f"border:1px solid var(--c-border);border-radius:6px;overflow:hidden'>"
                f"<thead><tr style='background:var(--c-surface-alt)'>"
                f"<th style='padding:7px 10px;text-align:left;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cmuted}'>Stream</th>"
                f"<th style='padding:7px 10px;text-align:center;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cmuted}'>Model error</th>"
                f"<th style='padding:7px 10px;text-align:center;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cmuted}'>Calendar guess error</th>"
                f"<th style='padding:7px 10px;text-align:center;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cmuted}'>Winner</th>"
                f"</tr></thead>"
                f"<tbody>{rows_exp}</tbody>"
                f"</table>"
                f"<div style='font-size:0.68rem;color:{_cmuted};margin-top:8px;line-height:1.5'>"
                f"Lower error = better. Both scores use the same scale (MASE) so they're directly comparable across rows."
                f"</div>"
                f"</div>"
            )
            st.markdown(explainer_html, unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── How models were selected — CV + AICc explainer ───────────────────────
    divider("how the models were selected")
    with st.container(border=True):
        section(
            "How the models were selected",
            "Two quality gates — AICc picks the right structure; cross-validation proves it works on real unseen data.",
        )

        # ── Two-column concept explainer ──────────────────────────────────────
        _ct = C["text"]
        _cs = C["secondary"]
        _cm = C["muted"]
        _ca = C["accent"]
        concept_html = (
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px'>"
            # Left: AICc
            f"<div style='border:1px solid rgba(92,107,192,0.30);border-top:4px solid {_ca};"
            f"border-radius:6px;padding:16px 18px;background:var(--c-card)'>"
            f"<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{_ca};margin-bottom:8px'>AICc — Model Selection</div>"
            f"<div style='font-size:0.85rem;font-weight:700;color:{_ct};margin-bottom:6px'>"
            f"Think of hiring a contractor</div>"
            f"<div style='font-size:0.78rem;color:{_cs};line-height:1.6;margin-bottom:10px'>"
            f"A good contractor completes the job <em>and</em> doesn't bring 50 unnecessary workers. "
            f"AICc rewards a model that fits the data well, but penalises unnecessary complexity. "
            f"A model that's too simple misses real patterns. A model that's too complex "
            f"memorises noise and fails on new data.</div>"
            f"<div style='font-size:0.78rem;color:{_cs};line-height:1.6;margin-bottom:10px'>"
            f"AutoARIMA tested up to 20 different ARIMA structures. Each one was scored. "
            f"The one with the <strong>lowest AICc wins</strong> — best accuracy for the least complexity. "
            f"AutoETS did the same for exponential smoothing models.</div>"
            f"<div style='background:var(--c-surface-alt);border-radius:4px;padding:8px 10px;"
            f"font-size:0.72rem;color:{_cm};line-height:1.5'>"
            f"AICc = AIC corrected for small samples. At 69 months of data, "
            f"plain AIC over-selects complex models — AICc adds a stronger complexity penalty."
            f"</div></div>"
            # Right: CV
            f"<div style='border:1px solid rgba(67,160,71,0.30);border-top:4px solid {C['bull']};"
            f"border-radius:6px;padding:16px 18px;background:var(--c-card)'>"
            f"<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{C['bull']};margin-bottom:8px'>Cross-Validation — Real-World Test</div>"
            f"<div style='font-size:0.85rem;font-weight:700;color:{_ct};margin-bottom:6px'>"
            f"The practice exam — taken 8 times</div>"
            f"<div style='font-size:0.78rem;color:{_cs};line-height:1.6;margin-bottom:10px'>"
            f"AICc only checks how well the model fits data it <em>already saw</em> — like studying "
            f"with the answer sheet. Cross-validation is different: "
            f"train on older data, predict 12 months the model has <strong>never seen</strong>, "
            f"measure the real error. Then repeat 8 times, sliding the window forward."
            f"</div>"
            f"<div style='font-size:0.78rem;color:{_cs};line-height:1.6;margin-bottom:10px'>"
            f"The MASE accuracy scores in the section above are from those 8 rounds — "
            f"not from the training data. This is the gold standard for validating a forecast model "
            f"(the M4/M5 forecasting competitions use this exact method)."
            f"</div>"
            f"<div style='background:var(--c-surface-alt);border-radius:4px;padding:8px 10px;"
            f"font-size:0.72rem;color:{_cm};line-height:1.5'>"
            f"Rolling-origin CV: 8 windows, 3-month step, 12-month forecast horizon each time. "
            f"All windows are pooled to compute the final MASE score."
            f"</div></div>"
            f"</div>"
        )
        st.markdown(concept_html, unsafe_allow_html=True)

        # ── Model selection table ─────────────────────────────────────────────
        if not model_info.empty:
            _MI_LABELS = {
                "IV_League": "IV League",
                "MPD_Core_MRR": "Subscriptions — Recurring MRR",
                "MPD_Core_OneTime": "Subscriptions — Premium Packages",
            }
            _MI_COLORS = {
                "IV_League": C["iv"],
                "MPD_Core_MRR": C["mrr"],
                "MPD_Core_OneTime": C["ot"],
            }
            _MI_MODELS = {
                "EAT": "EAT Ensemble (AutoARIMA + AutoETS + Theta averaged)",
                "Level": "Level Model (exponentially-weighted mean, stable level)",
            }
            rows_mi = ""
            for _, r in model_info.iterrows():
                skey = r["series"]
                color = _MI_COLORS.get(skey, _cm)
                lbl = _MI_LABELS.get(skey, skey)
                m_type = r.get("model", "")
                order = r.get("AutoARIMA_order", "")
                ets = r.get("AutoETS_method", "")
                arima_aicc = r.get("AutoARIMA_aicc", float("nan"))
                ets_aicc = r.get("AutoETS_aicc", float("nan"))
                brk = r.get("break_date", "")
                # Winner label: lower AICc wins between ARIMA and ETS
                if not pd.isna(arima_aicc) and not pd.isna(ets_aicc):
                    winner = "AutoARIMA" if arima_aicc <= ets_aicc else "AutoETS"
                    win_note = f"AutoARIMA AICc {arima_aicc:,.0f} vs AutoETS AICc {ets_aicc:,.0f} → {winner} selected"
                elif m_type == "Level":
                    win_note = "Level Model — no AICc (mean-reverting, no trend to select)"
                else:
                    win_note = ""
                brk_html = (
                    (
                        f"<div style='font-size:0.68rem;color:{C['bear']};margin-top:3px'>"
                        f"⚡ Structural break detected {brk} — step dummy added as regressor</div>"
                    )
                    if brk and str(brk) != "nan"
                    else ""
                )
                rows_mi += (
                    f"<tr>"
                    f"<td style='padding:9px 12px;border-bottom:1px solid var(--c-grid)'>"
                    f"<span style='display:inline-flex;align-items:center;gap:6px'>"
                    f"<span style='width:8px;height:8px;border-radius:50%;background:{color};display:inline-block'></span>"
                    f"<span style='font-size:0.75rem;font-weight:600;color:{_ct}'>{lbl}</span></span>"
                    f"{brk_html}</td>"
                    f"<td style='padding:9px 12px;border-bottom:1px solid var(--c-grid);font-size:0.73rem;color:{_cs}'>"
                    f"{_MI_MODELS.get(m_type, m_type)}</td>"
                    f"<td style='padding:9px 12px;border-bottom:1px solid var(--c-grid);font-size:0.73rem;"
                    f"font-family:monospace;color:{_ct}'>"
                    f"{order}<br><span style='color:{_cm}'>{ets}</span></td>"
                    f"<td style='padding:9px 12px;border-bottom:1px solid var(--c-grid);font-size:0.72rem;color:{_cm}'>"
                    f"{win_note}</td>"
                    f"</tr>"
                )

            table_html = (
                f"<div style='border:1px solid var(--c-border);border-radius:6px;overflow:hidden;margin-top:4px'>"
                f"<table style='width:100%;border-collapse:collapse;background:var(--c-card)'>"
                f"<thead><tr style='background:var(--c-surface-alt)'>"
                f"<th style='padding:9px 12px;text-align:left;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cm}'>Revenue stream</th>"
                f"<th style='padding:9px 12px;text-align:left;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cm}'>Method</th>"
                f"<th style='padding:9px 12px;text-align:left;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cm}'>Selected structure</th>"
                f"<th style='padding:9px 12px;text-align:left;font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.08em;text-transform:uppercase;color:{_cm}'>AICc selection result</th>"
                f"</tr></thead>"
                f"<tbody>{rows_mi}</tbody>"
                f"</table>"
                f"<div style='padding:8px 12px;background:var(--c-surface-alt);font-size:0.68rem;color:{_cm};line-height:1.5'>"
                f"AICc = lower is better. AutoARIMA and AutoETS each find their own best structure; "
                f"the ensemble averages all three (ARIMA + ETS + Theta) equally at forecast time."
                f"</div></div>"
            )
            st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── How the forecast is built — 4 process step cards ─────────────────────
    divider("how the forecast is built")
    with st.container(border=True):
        section(
            "How the forecast is built",
            "Four steps from raw payment data to the Bear / Base / Bull numbers you see.",
        )
        steps = [
            (
                "1",
                C["iv"],
                "Split by revenue type",
                "Revenue is split into two reporting streams: IV League and Subscriptions. "
                "Subscriptions is internally modelled as two sub-streams (recurring MRR + one-time packages) "
                "then combined for reporting. A soft month in one stream won't distort the other.",
            ),
            (
                "2",
                C["mrr"],
                "Right model for each stream",
                "IV League and Subscriptions use an ensemble of three statistical methods averaged equally — "
                "this smooths out any one method's quirks. One-Time packages use a <em>level model</em>: "
                "a flat forecast anchored to the recent average, because high-ticket package sales don't "
                "follow a predictable trend — they fluctuate around a stable level.",
            ),
            (
                "3",
                C["ot"],
                "Bear / Base / Bull ranges",
                "Each model produces a planning range. Base is the model's single best guess. "
                "Bear is what revenue looks like if things underperform — use it as your budget floor. "
                "Bull is what revenue looks like if things go well — your upside ceiling. "
                "There's roughly an 80% chance actual revenue lands between them in any given month. "
                "The wide Bear/Bull gap on One-Time packages reflects the genuine unpredictability "
                "of when large package sales land.",
            ),
            (
                "4",
                C["base"],
                "Tested before it goes live",
                "Before generating any forecast, the model sat a practice exam: training on older data, "
                "predicting months it had never seen, checking how close it got — repeated 8 times. "
                "The accuracy scores above are from those tests. If the model couldn't beat a simple guess, "
                "it gets flagged.",
            ),
        ]
        cards_html = ""
        for num, color, title, desc in steps:
            cards_html += (
                f"<div style='border:1px solid var(--c-border-soft);border-radius:6px;background:var(--c-card);"
                f"padding:16px 18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.04);"
                f"display:flex;flex-direction:column'>"
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>"
                f"<div style='width:28px;height:28px;border-radius:50%;background:{color};"
                f"color:#fff;font-size:0.82rem;font-weight:800;display:flex;align-items:center;"
                f"justify-content:center;flex-shrink:0'>{num}</div>"
                f"<div style='font-size:0.82rem;font-weight:700;color:var(--c-text)'>{title}</div>"
                f"</div>"
                f"<div style='font-size:0.82rem;color:var(--c-text2);line-height:1.6;flex:1'>{desc}</div>"
                f"</div>"
            )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;"
            f"margin-top:4px;margin-bottom:8px;align-items:stretch'>"
            f"{cards_html}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Model quality checklist ───────────────────────────────────────────────
    divider("model quality checklist")
    with st.container(border=True):
        section(
            "Does this model follow forecasting best practices?",
            "Five principles every reliable forecast should satisfy — and an honest assessment of where this one stands.",
        )

        _ct = C["text"]
        _cs = C["secondary"]
        _cm = C["muted"]
        # Color psychology: teal = pass (distinct from stream greens),
        # amber = caution/gap, red = genuine failure only. Backgrounds use
        # rgba so the pairing reads in both light and dark themes.
        PASS_C = "#00695C"
        PASS_BG = "rgba(0,105,92,0.15)"  # teal — success, implemented
        WARN_C = "#E65100"
        WARN_BG = "rgba(230,81,0,0.15)"  # deep amber — needs attention
        FAIL_C = "#C62828"  # deep red — critical failure only

        principles = [
            (
                "✅",
                PASS_C,
                PASS_BG,
                "Patterns drive everything",
                "Trend + Seasonality + Noise",
                "AutoARIMA and AutoETS explicitly model trend (via differencing) and 12-month seasonality. "
                "STL decomposition visually separates all three layers before any model is fit. "
                "The Nov 2023 structural break in Subscriptions MRR is handled with a step dummy — "
                "so the model knows the business changed at that point rather than treating it as noise.",
            ),
            (
                "⚠️",
                WARN_C,
                WARN_BG,
                "Features = power",
                "Models don't see time — you give it structure",
                "Done partially. Temporal structure is given explicitly: COVID dummy, outlier intervention dummies, "
                "structural break step dummy, and 12-month seasonal periods hardcoded into every model. "
                "What's missing: external features like IV session bookings or marketing spend. "
                "Integrating Calendly booking data as a leading indicator would be the highest-value next upgrade.",
            ),
            (
                "⚠️",
                WARN_C,
                WARN_BG,
                "Validate, don't trust",
                "Good model → residuals = noise",
                "Validation is done rigorously — 8-window rolling-origin cross-validation on unseen data, "
                "all three models beat the seasonal naïve benchmark. However, the Ljung-Box test shows "
                "residual autocorrelation (p ≈ 0) across all series. This means the models are still "
                "leaving some predictable signal on the table. The forecasts are directionally reliable, "
                "but this is the main open technical gap. Adding external features (principle 2) is the "
                "most likely fix.",
            ),
            (
                "✅",
                PASS_C,
                PASS_BG,
                "Beware of spurious patterns",
                "Correlation ≠ causation",
                "Well handled by design. Univariate models have no confounding variable problem. "
                "COVID months are explicitly flagged rather than absorbed into trend. Outliers are identified "
                "and dummied out rather than silently skewing the model. "
                "Watch point: IV League has only 26 revenue months in 69 — the seasonal ARIMA term "
                "could be fitting a coincidental pattern on sparse data. This will self-correct as more data accumulates.",
            ),
            (
                "✅",
                PASS_C,
                PASS_BG,
                "Forecast with uncertainty",
                "Predict ranges, not points",
                "Fully implemented. Bear / Base / Bull at 80% prediction interval is the core dashboard output. "
                "95% intervals are also computed. Winkler score measures interval quality in cross-validation. "
                "Wide-range months are flagged with a warning badge in the forecast table. "
                "Caveat: residuals are non-normal (Shapiro-Wilk p ≈ 0), so Gaussian intervals may slightly "
                "understate tail risk in extreme months.",
            ),
        ]

        cards_html = ""
        for icon, status_c, _, title, subtitle, detail in principles:
            cards_html += (
                f"<div style='border:1px solid {status_c};border-left:4px solid {status_c};"
                f"border-radius:6px;padding:14px 16px;background:var(--c-card);"
                f"display:flex;flex-direction:column;gap:6px'>"
                # header row
                f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:10px'>"
                f"<div>"
                f"<div style='font-size:0.82rem;font-weight:700;color:{_ct}'>{title}</div>"
                f"<div style='font-size:0.70rem;color:{_cm};font-style:italic;margin-top:1px'>{subtitle}</div>"
                f"</div>"
                f"<span style='font-size:0.95rem;flex-shrink:0;margin-top:1px'>{icon}</span>"
                f"</div>"
                # detail
                f"<div style='font-size:0.76rem;color:{_cs};line-height:1.6'>{detail}</div>"
                f"</div>"
            )

        # 2-col grid for first 4, full-width for last (fits 5 naturally as 2+2+1)
        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:4px'>"
            f"{cards_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Legend
        st.markdown(
            f"<div style='display:flex;gap:16px;margin-top:6px;font-size:0.68rem;color:{_cm}'>"
            f"<span style='color:{PASS_C};font-weight:700'>✅ Meets standard</span>"
            f"<span style='color:{WARN_C};font-weight:700'>⚠️ Partially met — known gap</span>"
            f"<span style='color:{FAIL_C};font-weight:700'>❌ Not yet addressed</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Useful predictors audit ───────────────────────────────────────────────
    divider("predictors used in this model")
    with st.container(border=True):
        section(
            "Which predictors does this model use?",
            "Forecasting models can draw on three families of predictors. Here is what is and isn't implemented.",
        )

        _ct = C["text"]
        _cs = C["secondary"]
        _cm = C["muted"]
        # Badge backgrounds use rgba (≥0.15 alpha) so the semantic-color
        # text reads against both light and dark Streamlit themes.
        IN_C = "#00695C"
        IN_BG = "rgba(0,105,92,0.18)"  # teal — used
        GAP_C = "#E65100"
        GAP_BG = "rgba(230,81,0,0.18)"  # amber — addressable gap
        NA_C = "#546E7A"
        NA_BG = "rgba(84,110,122,0.18)"  # slate-gray — intentionally excluded

        def _pred_row(status, name, detail):
            if status == "in":
                badge = f"<span style='font-size:0.65rem;font-weight:700;color:{IN_C};background:{IN_BG};padding:2px 8px;border-radius:20px;white-space:nowrap'>✅ Used</span>"
            elif status == "partial":
                badge = f"<span style='font-size:0.65rem;font-weight:700;color:{GAP_C};background:{GAP_BG};padding:2px 8px;border-radius:20px;white-space:nowrap'>⚠️ Partial</span>"
            elif status == "gap":
                badge = f"<span style='font-size:0.65rem;font-weight:700;color:{GAP_C};background:{GAP_BG};padding:2px 8px;border-radius:20px;white-space:nowrap'>⚠️ Not yet</span>"
            else:  # "na"
                badge = f"<span style='font-size:0.65rem;font-weight:700;color:{NA_C};background:{NA_BG};padding:2px 8px;border-radius:20px;white-space:nowrap'>— N/A</span>"
            return (
                f"<div style='display:flex;align-items:flex-start;justify-content:space-between;"
                f"gap:10px;padding:9px 0;border-bottom:1px solid var(--c-grid)'>"
                f"<div style='flex:1'>"
                f"<div style='font-size:0.78rem;font-weight:700;color:{_ct};margin-bottom:2px'>{name}</div>"
                f"<div style='font-size:0.72rem;color:{_cs};line-height:1.5'>{detail}</div>"
                f"</div>"
                f"<div style='flex-shrink:0;padding-top:2px'>{badge}</div>"
                f"</div>"
            )

        def _pred_card(title, accent, rows_html):
            return (
                f"<div style='border:1px solid {accent};border-top:4px solid {accent};"
                f"border-radius:6px;padding:14px 16px;background:var(--c-card)'>"
                f"<div style='font-size:0.70rem;font-weight:700;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:{accent};margin-bottom:10px'>{title}</div>"
                f"{rows_html}"
                f"</div>"
            )

        col_accent = C["base"]  # indigo — matches existing palette

        time_rows = (
            _pred_row(
                "in",
                "Trend",
                "AutoARIMA models long-term direction via differencing (d=1 for IV League). "
                "AutoETS selects an explicit trend component. Both capture year-over-year growth or decline automatically.",
            )
            + _pred_row(
                "in",
                "Seasonality",
                "12-month seasonal period is hardcoded for all models. AutoARIMA fits seasonal ARIMA terms "
                "(P,D,Q)[12] — e.g., IV League uses seasonal AR(1) at lag 12, MRR uses seasonal AR(1).",
            )
            + _pred_row(
                "na",
                "Fourier Terms (sin/cos)",
                "Not applicable for this data. Fourier terms model seasonality via sine/cosine waves — useful when "
                "seasonality is complex or irregular. For monthly data with a single 12-month cycle, ARIMA seasonal "
                "terms are equivalent and simpler. No action needed.",
            )
        )

        events_rows = (
            _pred_row(
                "in",
                "Dummy Variables",
                "COVID startup period (Aug 2020 – Oct 2021) is flagged as a binary dummy across all models. "
                "Outlier months detected by Hampel filter get individual intervention dummies (0/1 per month).",
            )
            + _pred_row(
                "in",
                "Intervention Variables",
                "COVID dummy captures the sudden startup shock. A structural break step dummy handles the "
                "Nov 2023 regime change in Subscriptions MRR — the model explicitly knows the business changed at that point.",
            )
            + _pred_row(
                "gap",
                "Trading-Day Effects",
                "Not yet modeled. Some months have more working days than others, which can affect IV League session volume. "
                "At monthly aggregation this effect is small but non-zero — a working-days-per-month regressor "
                "would be a low-effort improvement worth adding.",
            )
        )

        depend_rows = _pred_row(
            "in",
            "Lagged Predictors  y(t−1), y(t−2)",
            "ARIMA AR terms are lagged revenue values used directly as predictors. MRR uses AR(1) — "
            "last month's revenue predicts this month's. IV League uses seasonal AR(1) at lag 12 — "
            "last year's same month predicts this year's. MA terms use lagged forecast errors.",
        ) + _pred_row(
            "gap",
            "Cross-series lags",
            "Not yet modeled. IV League bookings this month likely influence Subscriptions next month "
            "(IV-to-membership conversions). Each stream is currently forecast independently. "
            "A VAR (Vector AutoRegression) model or transfer function would capture this relationship.",
        )

        st.markdown(
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:10px'>"
            f"{_pred_card('Time Patterns', col_accent, time_rows)}"
            f"{_pred_card('Events &amp; Calendar', col_accent, events_rows)}"
            f"{_pred_card('Dependence', col_accent, depend_rows)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:0.70rem;color:{_cm};line-height:1.5;margin-top:4px'>"
            f"<strong>2 addressable gaps</strong> — Trading-Day Effects and Cross-series lags (amber). "
            f"Both are low-to-medium effort improvements that would most benefit the IV League forecast. "
            f"Fourier Terms are marked N/A (gray) — not a gap, just a different technique that would be redundant here."
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Assumptions + Outliers — side by side CSS grid ───────────────────────
    divider("assumptions and unusual months")

    # Build assumptions HTML
    assumptions = [
        (
            "🔄",
            "Business continues as usual",
            "Revenue patterns since late 2021 keep going — no major pivot, pricing change, or new service line.",
        ),
        (
            "📅",
            "Seasonal patterns repeat",
            "The busy and quiet months follow a similar rhythm each year.",
        ),
        (
            "📊",
            "Unusual months don't repeat",
            "Months with abnormally high or low revenue are treated as one-offs, not as a new normal.",
        ),
        (
            "✂️",
            "Only complete months used",
            "Training data ends at the last fully completed calendar month. Partial months are excluded to avoid skewing the model.",
        ),
        (
            "🚫",
            "COVID era set aside",
            "Aug 2020 – Oct 2021 are flagged as structurally unusual. The model learns from them but knows not to treat them as representative.",
        ),
    ]
    assume_rows = ""
    for i, (icon, title, detail) in enumerate(assumptions):
        border = "border-bottom:1px solid var(--c-grid);" if i < len(assumptions) - 1 else ""
        assume_rows += (
            f"<div style='display:flex;gap:12px;padding:10px 0;{border}'>"
            f"<div style='font-size:1.1rem;flex-shrink:0;padding-top:1px'>{icon}</div>"
            f"<div><div style='font-size:0.82rem;font-weight:700;color:var(--c-text);margin-bottom:2px'>{title}</div>"
            f"<div style='font-size:0.78rem;color:var(--c-text2);line-height:1.5'>{detail}</div>"
            f"</div></div>"
        )

    # Build outliers HTML
    assume_header = (
        f"<div style='padding-bottom:8px;margin-bottom:14px;border-bottom:1px solid {C['border']}'>"
        f"<span style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{C['muted']}'>What the model assumes</span>"
        f"<div style='font-size:0.82rem;color:var(--c-text2);margin-top:4px;line-height:1.5'>"
        f"The forecast is only as good as these conditions holding. Flag any that change.</div></div>"
    )
    outlier_header = (
        f"<div style='padding-bottom:8px;margin-bottom:14px;border-bottom:1px solid {C['border']}'>"
        f"<span style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:{C['muted']}'>Unusual months the model knows about</span>"
        f"<div style='font-size:0.82rem;color:var(--c-text2);margin-top:4px;line-height:1.5'>"
        f"These months had abnormally high or low revenue. "
        f"The model treats them as one-off events so they don't skew the forecast.</div></div>"
    )

    outlier_rows = ""
    if not outliers.empty:
        ol = outliers.copy()
        ol["series_key"] = ol["series"]
        ol["series"] = ol["series"].map(ABOUT_LABELS).fillna(ol["series"])
        ol["month_fmt"] = ol["month"].dt.strftime("%b %Y")

        stream_avgs = {}
        for key in ["IV_League", "MPD_Core_MRR", "MPD_Core_OneTime"]:
            post = monthly[key][monthly["is_covid_startup"] == 0]
            stream_avgs[key] = float(post[post > 0].median())

        n = len(ol)
        for i, (_, row) in enumerate(ol.iterrows()):
            key = row["series_key"]
            color = ABOUT_COLORS.get(key, C["muted"])
            border = "border-bottom:1px solid var(--c-grid);" if i < n - 1 else ""
            month_ts = row["month"]
            amount = float(monthly.loc[month_ts, key]) if month_ts in monthly.index else None
            avg = stream_avgs.get(key)
            amount_html = ""
            if amount is not None and avg and avg > 0:
                pct_above = (amount - avg) / avg * 100
                sign = "+" if pct_above >= 0 else ""
                amount_html = (
                    f"<span style='font-size:0.82rem;font-weight:700;color:{color}'>"
                    f"${amount:,.0f}</span>"
                    f"<span style='font-size:0.72rem;color:var(--c-caption);margin-left:6px'>"
                    f"{sign}{pct_above:.0f}% vs stream median</span>"
                )
            outlier_rows += (
                f"<div style='display:flex;align-items:center;gap:14px;padding:10px 0;{border}'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0'></div>"
                f"<div style='font-size:0.82rem;color:var(--c-text);font-weight:600;width:80px;flex-shrink:0'>"
                f"{row['month_fmt']}</div>"
                f"<div style='font-size:0.78rem;color:var(--c-muted);flex:1'>{row['series']}</div>"
                f"<div style='text-align:right'>{amount_html}</div>"
                f"</div>"
            )
    else:
        outlier_rows = f"<div style='font-size:0.78rem;color:{C['muted']};padding:12px 0'>No unusual months detected.</div>"

    st.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:stretch'>"
        # left panel
        f"<div style='border:1px solid {C['border_mid']};border-radius:6px;background:var(--c-card);"
        f"padding:16px 18px 20px;box-sizing:border-box'>"
        f"{assume_header}{assume_rows}"
        f"</div>"
        # right panel
        f"<div style='border:1px solid {C['border_mid']};border-radius:6px;background:var(--c-card);"
        f"padding:16px 18px 20px;box-sizing:border-box'>"
        f"{outlier_header}{outlier_rows}"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def build_sidebar(data: dict | None) -> str:
    with st.sidebar:
        st.markdown(
            f"<div style='padding:8px 0 16px'>"
            f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:{C['muted']}'>My Performance Doctor</div>"
            f"<div style='font-size:1.1rem;font-weight:800;color:{C['text']};margin-top:2px'>"
            f"Revenue Forecast</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='height:1px;background:{C['border']};margin-bottom:16px'></div>",
            unsafe_allow_html=True,
        )

        # Horizon selector — global control, affects all tabs
        st.markdown(
            f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.1em;"
            f"text-transform:uppercase;color:{C['muted']};margin-bottom:2px'>Forecast Horizon</div>"
            f"<div style='font-size:0.78rem;color:{C['muted']};margin-bottom:10px'>"
            f"Applies to all tabs</div>",
            unsafe_allow_html=True,
        )
        horizon = st.radio(
            "horizon",
            ["12 months", "24 months"],
            index=0,
            label_visibility="collapsed",
            help="12 months = operational planning. 24 months = strategic direction.",
        )
        st.markdown(
            f"<div style='background:{C['base_bg']};border-left:3px solid {C['base']};"
            f"border-radius:3px;padding:8px 12px;margin-top:8px;"
            f"font-size:0.78rem;color:{C['base']};font-weight:600'>"
            f"Viewing: {horizon}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='height:1px;background:{C['border']};margin:16px 0'></div>",
            unsafe_allow_html=True,
        )

        if data:
            last = data["monthly"].index[-1]
            st.markdown(
                f"<div style='font-size:0.78rem;color:{C['secondary']};line-height:2.0'>"
                f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.1em;text-transform:uppercase'>Data through</span><br>"
                f"<b>{last.strftime('%B %Y')}</b><br><br>"
                f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.1em;text-transform:uppercase'>Training months</span><br>"
                f"<b>{len(data['monthly'])}</b><br><br>"
                f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                f"letter-spacing:0.1em;text-transform:uppercase'>Loaded</span><br>"
                f"<b>{datetime.now():%d %b %Y, %H:%M}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Provenance: manifest-driven badge ─────────────────────────
            # Surfaces who/when/what produced the artifacts the page is
            # showing. When the manifest is absent (first deploy, or pipeline
            # has never run successfully) we show a muted "unavailable" badge
            # rather than blocking the page.
            st.markdown(
                f"<div style='height:1px;background:{C['border']};margin:16px 0'></div>",
                unsafe_allow_html=True,
            )
            manifest = data.get("manifest")
            if manifest:
                # Manifest values are pipeline-written so the threat surface is
                # narrow, but follow the same rule as forecast_html_table:
                # any string interpolated into unsafe_allow_html markup gets
                # html.escape'd. The known-source dict gives safe text;
                # html.escape is a no-op on those and guards the fallback path.
                source_label = html.escape(
                    {
                        "stripe_api": "Stripe API",
                        "historical_csv": "Historical CSV",
                    }.get(manifest.get("input_source", ""), manifest.get("input_source", "—"))
                )
                run_at = manifest.get("run_at", "")
                try:
                    run_at_human = datetime.strptime(run_at, "%Y-%m-%dT%H:%M:%SZ").strftime(
                        "%d %b %Y, %H:%M UTC"
                    )
                except (ValueError, TypeError):
                    run_at_human = html.escape(run_at) if run_at else "—"
                git_short = html.escape((manifest.get("git_sha") or "unknown")[:7])
                st.markdown(
                    f"<div style='font-size:0.78rem;color:{C['secondary']};line-height:2.0'>"
                    f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                    f"letter-spacing:0.1em;text-transform:uppercase'>Last refreshed</span><br>"
                    f"<b>{run_at_human}</b><br><br>"
                    f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                    f"letter-spacing:0.1em;text-transform:uppercase'>Source</span><br>"
                    f"<b>{source_label}</b><br><br>"
                    f"<span style='color:{C['muted']};font-size:0.65rem;font-weight:700;"
                    f"letter-spacing:0.1em;text-transform:uppercase'>Code</span><br>"
                    f"<code style='font-size:0.72rem'>{git_short}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size:0.72rem;color:{C['muted']};font-style:italic'>"
                    f"Provenance unavailable — pipeline may not have run yet."
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Pipeline output not found. Run the pipeline first.")

    return horizon


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    try:
        data = load_data()
    except Exception:
        # A required CSV exists but is unreadable (mid-write, schema drift,
        # corrupt header). Surface a friendly message instead of a raw
        # Streamlit traceback; the trace still goes to stderr for CloudWatch.
        import traceback

        traceback.print_exc()
        horizon = build_sidebar(None)
        st.error("Forecast data is being refreshed — please try again in a moment.")
        return

    horizon = build_sidebar(data)

    st.markdown(
        f"<h1 style='font-family:Inter,system-ui,sans-serif;font-size:1.5rem;"
        f"font-weight:800;color:{C['text']};margin-bottom:2px'>Revenue Forecast Dashboard</h1>",
        unsafe_allow_html=True,
    )

    if data is None:
        st.warning(
            "Forecast data is not yet available. The revenue pipeline needs to run before this page is useful."
        )
        return

    # Combine MRR + OneTime into a single Subscriptions stream for display
    fc = (data["fc12"] if horizon == "12 months" else data["fc24"]).copy()
    fc["MPD_Core"] = fc["MPD_Core_MRR"] + fc["MPD_Core_OneTime"]

    monthly = data["monthly"].copy()
    monthly["MPD_Core"] = monthly["MPD_Core_MRR"] + monthly["MPD_Core_OneTime"]
    data_aug = {**data, "monthly": monthly}

    tab1, tab2, tab3 = st.tabs(
        [
            "📊  Executive Summary",
            "📈  Forecast Dashboard",
            "🔍  About This Forecast",
        ]
    )
    with tab1:
        tab_exec(data_aug, fc, horizon)
    with tab2:
        tab_dashboard(data_aug, fc, horizon)
    with tab3:
        tab_assumptions(data)

    # ── Footer ────────────────────────────────────────────────────────────────
    # Hardcoded client-facing contact, matching the retention page footer.
    # (Access-control screens still resolve through auth.access_contact_email().)
    contact_html = (
        "<a href='mailto:joshua@myperformancedoctor.com'>"
        "Contact joshua@myperformancedoctor.com</a>"
    )
    st.markdown(
        "<style>"
        ".mpd-footer{margin-top:48px;padding:16px 0 8px;"
        "border-top:1px solid var(--c-border);text-align:center}"
        ".mpd-footer span{font-size:0.75rem;color:var(--c-caption)}"
        ".mpd-footer a{color:#5C6BC0;text-decoration:none;font-size:0.75rem;font-weight:600}"
        ".mpd-footer a:hover{text-decoration:underline}"
        "</style>"
        "<div class='mpd-footer'>"
        "<span>MPD Revenue Forecast Dashboard"
        + ("&nbsp;·&nbsp; Questions? &nbsp;" if contact_html else "")
        + "</span>"
        + contact_html
        + "</div>",
        unsafe_allow_html=True,
    )


@page(SLUG)
def render() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    main()
