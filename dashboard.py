"""Standalone Streamlit entrypoint for the revenue forecast page.

In production this page is one route inside a multi-page dashboard behind ALB
OIDC auth and Google-Sheet authorization. This repo isolates just the revenue
page, so this entrypoint is deliberately thin: it owns `st.set_page_config` and
the theme-neutral `--c-*` CSS tokens the page's inline HTML inherits (the exact
token contract from the full dashboard), then calls the page's `render()`.

    uv run streamlit run dashboard.py

The page reads forecast artifacts from `outputs/` (the storage switch's local
backend), so run the pipeline first — at minimum the Milestone 1 data pipeline
(`uv run python scripts/run_pipeline.py`), or the full forecast pipeline
(`uv run python scripts/run_revenue_pipeline.py`) for the complete page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

st.set_page_config(page_title="MPD Revenue Forecast", page_icon="📈", layout="wide")

# Theme-neutral color tokens. The page never branches on light/dark; it maps
# onto these `--c-*` vars (text = currentColor) so Streamlit's theme toggle
# controls the whole surface via inheritance.
st.markdown(
    """
<style>
:root {
    --c-text:        currentColor;
    --c-text2:       color-mix(in srgb, currentColor 65%, transparent);
    --c-muted:       color-mix(in srgb, currentColor 55%, transparent);
    --c-caption:     color-mix(in srgb, currentColor 40%, transparent);
    --c-faint:       color-mix(in srgb, currentColor 28%, transparent);
    --c-card:        rgba(128,128,128,0.06);
    --c-bg:          transparent;
    --c-surface:     transparent;
    --c-surface-alt: rgba(128,128,128,0.03);
    --c-border:      color-mix(in srgb, currentColor 14%, transparent);
    --c-border-soft: color-mix(in srgb, currentColor 8%, transparent);
    --c-grid:        color-mix(in srgb, currentColor 6%, transparent);
}
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 4px !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: transparent;
    border-bottom: 1px solid var(--c-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 4px 4px 0 0; padding: 8px 22px;
    font-weight: 500; color: var(--c-muted); font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: var(--c-card) !important;
    color: var(--c-text) !important;
    border-bottom: 2px solid #5C6BC0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

from src.pages import revenue  # noqa: E402  (path + page_config set up above)

revenue.render()
