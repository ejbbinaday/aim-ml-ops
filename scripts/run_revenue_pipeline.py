"""
MPD Revenue Forecasting Pipeline
==================================
Refresh-or-no-op pipeline. Designed to run unattended on a monthly cron and
also to be safe for ad-hoc local invocations.

Usage
-----
    python run_revenue_pipeline.py              # full pipeline
    python run_revenue_pipeline.py --phase 0    # pull a fresh Stripe snapshot
    python run_revenue_pipeline.py --phase 1    # single phase (0–5)
    python run_revenue_pipeline.py --skip-eval  # skip CV (faster monthly refresh)
    python run_revenue_pipeline.py --force      # ignore the idempotent no-op gate

Outputs are written to ``outputs/<…>`` or ``s3://$OUTPUTS_BUCKET/<…>``
depending on the ``OUTPUTS_BUCKET`` env var.

Inputs
------
* ``STRIPE_API_KEY`` (env var; ``rk_…`` restricted key recommended) — when set
  AND ``--phase`` is None or 0, Phase 0 fetches a fresh Stripe snapshot to
  ``tables/00_stripe_charges.csv``. When unset, Phase 1 falls back to
  ``historical_unified_payments.csv`` at the repo root.
* ``MPD_REVENUE_MASE_REGRESSION_RATIO`` (env var; default ``1.30``) — sets the
  Phase-4 MASE regression-gate threshold.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.revenue import (
    manifest as manifest_module,
)
from src.revenue import (
    phase0_ingest,
    phase1_data,
    phase2_eda,
    phase3_models,
    phase4_eval,
    phase5_forecast,
)
from src.revenue.phase2_eda import EDAResult


def _should_run_phase0(run_phase: int | None) -> bool:
    """Phase 0 fires when explicitly requested OR when the full pipeline is
    requested AND a Stripe API key is available. Explicit ``--phase N`` for
    N >= 1 never re-pulls (see design D3)."""
    if run_phase == 0:
        return True
    if run_phase is None:
        return bool(os.getenv("STRIPE_API_KEY"))
    return False


def run(
    run_phase: int | None = None,
    skip_eval: bool = False,
    force: bool = False,
) -> None:
    t0 = time.time()

    print("=" * 65)
    print("  MPD Revenue Forecasting Pipeline")
    print("=" * 65)

    # ── Phase 0: Stripe ingestion ─────────────────────────────────────────────
    if _should_run_phase0(run_phase):
        phase0_ingest.run()

    if run_phase == 0:
        elapsed = time.time() - t0
        print(f"\n  Phase 0 complete in {elapsed:.1f}s")
        print(
            "  Next: re-run `--phase 3` (or the full pipeline) before publishing — "
            "the saved bundle and prior manifest still reference the previous snapshot."
        )
        return

    # ── Idempotent no-op gate ─────────────────────────────────────────────────
    # Compares the SHA-256 of the input phase 1 *would* read against the prior
    # manifest's input_sha256, AND compares the current git SHA. Match on both
    # → log a no-op and exit 0. Skipped on full-history first runs (no prior
    # manifest), single-phase invocations (the operator asked for a specific
    # phase), and `--force` invocations (the operator wants to re-run).
    if run_phase is None and not force:
        prior = manifest_module.load_prior_manifest()
        if prior:
            current_sha = phase1_data.current_input_sha256()
            current_git = manifest_module.git_sha()
            if prior.get("input_sha256") == current_sha and prior.get("git_sha") == current_git:
                print(
                    f"\n  no-op: input and code unchanged since "
                    f"{prior.get('run_at', 'unknown')}"
                )
                return

    # ── Phase 1: data engineering ─────────────────────────────────────────────
    if run_phase is None or run_phase == 1:
        monthly, phase1_meta = phase1_data.run()
    else:
        monthly = pd.read_csv(
            "outputs/tables/01_monthly_series.csv",
            index_col="month",
            parse_dates=["month"],
        )
        phase1_meta = {
            "input_source": "historical_csv",
            "input_path": "outputs/tables/01_monthly_series.csv",
        }

    if run_phase == 1:
        return

    # ── Phase 2: EDA ──────────────────────────────────────────────────────────
    if run_phase is None or run_phase == 2:
        eda_result = phase2_eda.run(monthly)
    else:
        outlier_df = pd.read_csv(
            "outputs/tables/02_outliers.csv",
            parse_dates=["month"],
        )
        outlier_dates = {col: list(grp["month"]) for col, grp in outlier_df.groupby("series")}
        eda_result = EDAResult(outlier_dates=outlier_dates, break_dates={})

    if run_phase == 2:
        return

    # ── Phase 3: model fit ────────────────────────────────────────────────────
    if run_phase is None or run_phase == 3:
        bundle = phase3_models.run(
            monthly,
            eda_result.outlier_dates,
            eda_result.break_dates,
        )
    else:
        bundle = None  # phases 4/5 load the persisted bundle (with staleness check)

    if run_phase == 3:
        return

    # ── Phase 4: evaluation + regression gate ─────────────────────────────────
    # phase4_eval.run() runs the MASE regression gate internally — before it
    # persists the eval table or re-persists the bundle — so a hard regression
    # raises here and leaves the outputs store untouched (no half-published run
    # ahead of phase 5's forecast CSVs and the terminal manifest write).
    eval_results: dict[str, dict] = {}
    if (run_phase is None or run_phase == 4) and not skip_eval:
        eval_results = phase4_eval.run(bundle, monthly)
    elif skip_eval:
        print("\n── Phase 4: Skipped (--skip-eval) ───────────────────────────────")

    if run_phase == 4:
        return

    # ── Phase 5: forecasts ────────────────────────────────────────────────────
    if run_phase is None or run_phase == 5:
        phase5_forecast.run(bundle, monthly)

    # ── Manifest writer (terminal step on full-pipeline runs) ─────────────────
    # Only writes a fresh manifest after a full pipeline run. Single-phase
    # invocations (`--phase N`) intentionally leave the prior manifest intact:
    # the manifest's `metrics` come from Phase 4's eval results, and the
    # orchestrator can only vouch for those after a Phase 3 + 4 sequence it
    # owns end-to-end.
    if run_phase is None:
        # Re-load bundle so manifest reflects phase 4's post-eval `model_type`s.
        terminal_bundle = bundle if bundle is not None else phase3_models.load_bundle()
        current_sha = phase1_data.current_input_sha256()
        manifest = manifest_module.write_manifest(
            input_source=phase1_meta["input_source"],
            input_sha256=current_sha,
            monthly=monthly,
            bundle=terminal_bundle,
            eval_results=eval_results,
        )
        print(
            f"\n  Manifest written → tables/00_manifest.json "
            f"(data_through={manifest['data_through']}, "
            f"git_sha={manifest['git_sha'][:7]})"
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 65}")


def _install_run_timeout() -> None:
    """Bound the unattended wall-clock run, restoring GHA's `timeout-minutes: 30`.

    The ECS task definition has no run-duration cap, and EventBridge — having
    already launched the task — never sees a hung run, so a stuck Stripe call or
    model fit would run (and bill Fargate) indefinitely and silently. SIGALRM
    caps the run and raises, which exits non-zero; the task-failure EventBridge
    rule then alerts. Configurable via ``MPD_REVENUE_RUN_TIMEOUT_SECONDS`` (set
    to 0 to disable, e.g. for a long ad-hoc local run); no-op where SIGALRM is
    unavailable (non-POSIX).
    """
    timeout_s = int(os.getenv("MPD_REVENUE_RUN_TIMEOUT_SECONDS", "1800"))
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        return

    def _on_timeout(signum, frame):
        raise TimeoutError(
            f"revenue pipeline exceeded MPD_REVENUE_RUN_TIMEOUT_SECONDS={timeout_s}s "
            "and was aborted (likely a hung Stripe call or model fit)."
        )

    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(timeout_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPD Revenue Forecasting Pipeline")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=None,
        help="Run only a single phase (0–5). Omit to run the full pipeline.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Phase 4 cross-validation (faster monthly refresh).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the idempotent no-op gate (always re-run phases 1–5).",
    )
    args = parser.parse_args()
    _install_run_timeout()
    run(run_phase=args.phase, skip_eval=args.skip_eval, force=args.force)
