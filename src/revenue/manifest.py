"""Manifest writer — stamps every successful pipeline run with provenance.

The manifest is the single source of truth the dashboard's sidebar reads to
answer "when was this forecast computed, against what data, by what code?"
It's also the seed for the idempotent-no-op gate (next run reads the prior
manifest's `input_sha256` to decide whether to short-circuit) and the
regression gate (next run reads the prior manifest's `metrics` to compare
MASE).

Atomicity is delegated to ``src.storage.save_text_atomic``: local path uses
``.tmp`` + ``os.replace``, S3 relies on ``PutObject``'s per-key atomicity.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src import storage

from . import pipeline_version
from .phase3_models import ModelBundle

MANIFEST_KEY = "tables/00_manifest.json"
SCHEMA_VERSION = 1


# ── git SHA resolution ────────────────────────────────────────────────────────


def git_sha() -> str:
    """Resolve the current commit SHA.

    Priority: ``MPD_GIT_SHA`` env var (stamped onto the revenue ECS container
    from the deploy's image tag) → ``GITHUB_SHA`` env var (set by GHA) →
    ``git rev-parse HEAD`` (local dev) → the literal ``"unknown"`` so the
    manifest still writes when the workspace is not a git checkout.

    ``MPD_GIT_SHA`` is the production path: the revenue image ships without a
    git binary or ``.git``, so without it ``git rev-parse`` fails and every
    run records ``"unknown"`` — which both blanks the manifest provenance and
    silently defeats the no-op gate's code-change comparison. Public because
    the orchestrator's idempotency gate also needs it.
    """
    sha = os.getenv("MPD_GIT_SHA") or os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"


# ── Manifest construction ─────────────────────────────────────────────────────


def _metric(value: Any) -> float | None:
    """Coerce a metric to float, or None if NaN / missing."""
    if value is None:
        return None
    if isinstance(value, int | float | np.floating) and not np.isnan(value):
        return float(value)
    return None


def build_manifest(
    *,
    input_source: str,
    input_sha256: str,
    monthly: pd.DataFrame,
    bundle: ModelBundle,
    eval_results: dict[str, dict],
) -> dict[str, Any]:
    """Assemble the manifest dict. Pure — no I/O."""
    model_selections = {
        bundle.iv_league.name: bundle.iv_league.model_type,
        bundle.mpd_core_mrr.name: bundle.mpd_core_mrr.model_type,
        bundle.mpd_core_onetime.name: bundle.mpd_core_onetime.model_type,
    }

    metrics: dict[str, dict[str, Any]] = {}
    for series_name, r in (eval_results or {}).items():
        metrics[series_name] = {
            "MASE": _metric(r.get("MASE")),
            "RMSE": _metric(r.get("RMSE")),
            "Winkler_80": _metric(r.get("Winkler")),
            "LjungBox_p": _metric(r.get("LjungBox_p")),
            "beats_baseline": bool(r.get("beats_baseline")) if "beats_baseline" in r else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "run_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "pipeline_version": pipeline_version,
        "input_source": input_source,
        "input_sha256": input_sha256,
        "data_through": monthly.index.max().strftime("%Y-%m-%d"),
        "model_selections": model_selections,
        "metrics": metrics,
    }


def write_manifest(
    *,
    input_source: str,
    input_sha256: str,
    monthly: pd.DataFrame,
    bundle: ModelBundle,
    eval_results: dict[str, dict],
) -> dict[str, Any]:
    """Build and atomically write the manifest. Returns the dict it wrote."""
    manifest = build_manifest(
        input_source=input_source,
        input_sha256=input_sha256,
        monthly=monthly,
        bundle=bundle,
        eval_results=eval_results,
    )
    storage.save_text_atomic(
        MANIFEST_KEY,
        json.dumps(manifest, indent=2),
        content_type="application/json",
    )
    return manifest


def _is_genuine_absence(exc: Exception) -> bool:
    """True iff `exc` means the manifest genuinely does not exist yet.

    Local backend: `FileNotFoundError`. S3 backend: a `NoSuchKey` (or 404)
    `ClientError`. Everything else — `AccessDenied`, throttling, a corrupt
    (non-JSON) body, network errors — is NOT absence and must propagate so the
    gate fails closed rather than silently treating it as "first run".
    """
    if isinstance(exc, FileNotFoundError):
        return True
    response = getattr(exc, "response", None)  # botocore ClientError shape
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        return code in {"NoSuchKey", "404"}
    return False


def load_prior_manifest() -> dict[str, Any] | None:
    """Read the prior manifest if present, else None.

    Used by the idempotent no-op gate and the MASE regression gate; both treat
    a genuinely-absent manifest as "first run, no baseline to compare against".

    A read that fails for any *other* reason (e.g. the task role lacks
    `s3:GetObject`, or the stored manifest is corrupt) is NOT silently absorbed
    — it propagates, so the regression gate fails closed instead of waving a
    bad forecast through under the guise of a first run.
    """
    try:
        return storage.load_json(MANIFEST_KEY)
    except Exception as exc:
        if _is_genuine_absence(exc):
            return None
        raise
