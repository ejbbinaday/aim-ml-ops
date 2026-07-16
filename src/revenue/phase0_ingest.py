"""
Phase 0 – Stripe Ingestion
==========================
Pulls every charge from the Stripe API where ``created >= SERIES_START``,
maps each charge into the column shape of ``historical_unified_payments.csv``
(the Stripe Dashboard export), and writes the result to
``tables/00_stripe_charges.csv`` via the storage switch.

Source-of-truth for the snapshot schema: see the spec
``automate-revenue-pipeline/specs/revenue-pipeline-automation/spec.md`` —
adding a metadata column SHALL be a code change here, not an automatic
side effect of MPD setting a new metadata key in Stripe.

The module reads its credentials exclusively from ``STRIPE_API_KEY``. It
does NOT call Secrets Manager — that responsibility belongs to the runtime
(GitHub Actions repo secrets in production, an env var in local dev).

Outputs
-------
outputs/tables/00_stripe_charges.csv (gitignored; PII)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src import storage

from .config import SERIES_START

SNAPSHOT_KEY = "tables/00_stripe_charges.csv"

# The three scopes a restricted key must carry. Listed in the error message
# when the startup probe fails so the operator can fix it in one trip.
# `Customers: read` is intentionally absent — per-customer fields are dropped
# from the snapshot (see the data-minimisation note below), so the customer
# resource is not expanded on the Stripe API call.
REQUIRED_SCOPES = (
    "Charges: read",
    "Balance transactions: read",
    "Invoices: read",
)

# Fixed metadata column set — hardcoded so the snapshot's schema is stable
# across runs. Adding a column is a code change, never a side effect of
# MPD setting a new metadata key in Stripe. `EmailAddress` is intentionally
# omitted vs. the historical CSV — it carries per-customer PII and no
# downstream phase reads it.
METADATA_COLUMNS = [
    "OrgName",
    "OrgCode",
    "Invoice number",
    "Site URL",
    "Order Type",
    "Payment Type",
    "Product Type",
    "Product",
    "Access Plan",
    "Order ID",
]

# Snapshot column order. PII-minimisation: `Customer Email`, `Customer
# Description`, `Customer ID`, and `Card ID` from the historical CSV are
# intentionally omitted. No phase 1–5 module reads them; keeping them would
# replicate per-customer PII into the S3 outputs bucket for no functional
# reason. The remaining identifiers (`Invoice ID`, `Transfer`, plus the
# org/site metadata) are charge- or org-level, not person-level.
SNAPSHOT_COLUMNS = [
    "id",
    "Created date (UTC)",
    "Amount",
    "Amount Refunded",
    "Currency",
    "Captured",
    "Converted Amount",
    "Converted Amount Refunded",
    "Converted Currency",
    "Decline Reason",
    "Description",
    "Fee",
    "Refunded date (UTC)",
    "Statement Descriptor",
    "Status",
    "Seller Message",
    "Taxes On Fee",
    "Invoice ID",
    "Transfer",
] + [f"{k} (metadata)" for k in METADATA_COLUMNS]


# ── Status mapping ────────────────────────────────────────────────────────────


def _map_status(stripe_status: str | None) -> str:
    """Stripe's `charge.status` → the historical CSV's `Status` column.

    `succeeded` → `Paid` is load-bearing: phase 1 filters on
    `Status == "Paid"` and would otherwise drop every row.
    """
    if not stripe_status:
        return ""
    if stripe_status == "succeeded":
        return "Paid"
    if stripe_status == "failed":
        return "Failed"
    if stripe_status == "pending":
        return "Pending"
    return stripe_status.capitalize()


# ── Currency helpers ──────────────────────────────────────────────────────────

# Most currencies (including AUD) are minor-unit ×100. A small set are zero-
# decimal (JPY, KRW, …). Stripe documents the full list; the snapshot is AUD
# in practice today so a 2-decimal default is correct.
_ZERO_DECIMAL_CURRENCIES = {
    "bif",
    "clp",
    "djf",
    "gnf",
    "jpy",
    "kmf",
    "krw",
    "mga",
    "pyg",
    "rwf",
    "ugx",
    "vnd",
    "vuv",
    "xaf",
    "xof",
    "xpf",
}


def _minor_to_major(amount_minor: int | None, currency: str | None) -> float | None:
    if amount_minor is None:
        return None
    if currency and currency.lower() in _ZERO_DECIMAL_CURRENCIES:
        return float(amount_minor)
    return amount_minor / 100.0


# ── Datetime formatting ───────────────────────────────────────────────────────


def _fmt_utc_dt(ts: int | None) -> str:
    """Unix timestamp → `YYYY-MM-DD HH:MM:SS` (UTC, no timezone suffix).

    Matches the historical CSV's `Created date (UTC)` format exactly so phase 1's
    `pd.read_csv(..., parse_dates=["Created date (UTC)"])` parses identically.
    """
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


# ── Per-charge mapping ────────────────────────────────────────────────────────


def _attr(obj: Any, key: str) -> Any:
    """Safe attribute/key access — handles both stripe objects and dicts."""
    if obj is None:
        return None
    if hasattr(obj, "get"):
        return obj.get(key)
    return getattr(obj, key, None)


def _charge_to_row(charge: Any) -> dict[str, Any]:
    """Map one Stripe ``Charge`` to the snapshot's column shape.

    Per-customer PII and pseudonyms (`Customer Email`, `Customer Description`,
    `Customer ID`, `Card ID`) are intentionally omitted — see SNAPSHOT_COLUMNS
    for the rationale. As a side effect, neither `charge.customer` nor
    `charge.payment_method_details` is read here, and `data.customer` is
    dropped from the Stripe `expand` list.

    The Stripe ``Charges`` API does not populate every column the Dashboard
    export carries — ``Converted Amount*`` fields come from the Balance/Payouts
    Reports API. For those, we leave the cell empty so the snapshot is still
    schema-compatible.
    """
    currency = _attr(charge, "currency")
    bt = _attr(charge, "balance_transaction") or {}
    outcome = _attr(charge, "outcome") or {}

    fee_minor = _attr(bt, "fee")
    fee_details = _attr(bt, "fee_details") or []
    taxes_on_fee_minor = sum(
        (_attr(fd, "amount") or 0) for fd in fee_details if (_attr(fd, "type") or "") == "tax"
    )

    refunds = _attr(charge, "refunds") or {}
    refund_data = _attr(refunds, "data") or []
    refunded_at = None
    if refund_data:
        # Earliest refund timestamp is good enough for the snapshot — the
        # historical CSV records a single refund date even when there are
        # multiple partial refunds.
        refunded_at = min(
            (_attr(r, "created") for r in refund_data if _attr(r, "created") is not None),
            default=None,
        )

    invoice = _attr(charge, "invoice")
    invoice_id = invoice if isinstance(invoice, str) else _attr(invoice, "id")

    row: dict[str, Any] = {
        "id": _attr(charge, "id"),
        "Created date (UTC)": _fmt_utc_dt(_attr(charge, "created")),
        "Amount": _minor_to_major(_attr(charge, "amount"), currency),
        "Amount Refunded": _minor_to_major(_attr(charge, "amount_refunded"), currency),
        "Currency": currency,
        "Captured": _attr(charge, "captured"),
        "Converted Amount": None,  # Balance/Payouts Reports API only
        "Converted Amount Refunded": None,  # ditto
        "Converted Currency": None,  # ditto
        "Decline Reason": _attr(charge, "failure_message"),
        "Description": _attr(charge, "description"),
        "Fee": _minor_to_major(fee_minor, currency),
        "Refunded date (UTC)": _fmt_utc_dt(refunded_at) if refunded_at else "",
        "Statement Descriptor": _attr(charge, "statement_descriptor"),
        "Status": _map_status(_attr(charge, "status")),
        "Seller Message": _attr(outcome, "seller_message"),
        "Taxes On Fee": _minor_to_major(taxes_on_fee_minor, currency)
        if taxes_on_fee_minor
        else None,
        "Invoice ID": invoice_id,
        "Transfer": _attr(charge, "transfer"),
    }

    metadata = _attr(charge, "metadata") or {}
    for key in METADATA_COLUMNS:
        row[f"{key} (metadata)"] = _attr(metadata, key)

    return row


# ── Public entry point ────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """
    Pull every charge from Stripe and write the snapshot atomically.

    Returns
    -------
    dict
        ``{"skipped": True}`` when ``STRIPE_API_KEY`` is unset.
        ``{"input_source": "stripe_api", "input_path": SNAPSHOT_KEY, "row_count": int}``
        on success.
    """
    api_key = os.getenv("STRIPE_API_KEY")
    if not api_key:
        print("  STRIPE_API_KEY unset — skipping phase 0, using on-disk CSV")
        return {"skipped": True}

    # Operator-provisioned config guard. The CDK secret template ships every
    # field as the literal "<replace_me>"; a non-empty-but-unreplaced key
    # otherwise sails past the `not api_key` check and fails several calls deep
    # with an opaque Stripe auth error (routed to the unattended DLQ). Bail now
    # with an actionable message — this restores the fast-fail the GHA workflow's
    # "Validate operator-provisioned config" preflight used to provide.
    if not api_key.startswith(("sk_", "rk_")):
        raise RuntimeError(
            "STRIPE_API_KEY is not a valid Stripe key (expected an `rk_` or `sk_` "
            f"prefix, got {api_key[:8]!r}…). If this is the CDK secret's "
            "`<replace_me>` placeholder, set the real restricted key with "
            "`aws secretsmanager update-secret` — see docs/revenue-pipeline-runbook.md."
        )

    if api_key.startswith(("sk_live_", "sk_test_")):
        print(
            "  WARN: STRIPE_API_KEY is a full secret key; "
            "restricted key (rk_...) is recommended for least-privilege"
        )

    # Imported lazily so `python run_revenue_pipeline.py --phase 1` works
    # without the `stripe` package installed (CI deps for the dashboard).
    import stripe

    stripe.api_key = api_key
    stripe.max_network_retries = 5

    # `data.customer` is intentionally omitted — per-customer fields are not
    # written to the snapshot, so expanding the customer object is wasted
    # bytes (and would require the `Customers: read` scope on the key).
    expand = ["data.balance_transaction", "data.invoice"]

    # ── Startup probe ─────────────────────────────────────────────────────────
    # Fails fast on under-scoped restricted keys before we burn quota walking
    # pagination. The probe lists at most one charge.
    print("\n── Phase 0: Stripe Ingestion ────────────────────────────────────")
    try:
        stripe.Charge.list(limit=1, expand=expand)
    except stripe.error.PermissionError as e:
        scopes = ", ".join(REQUIRED_SCOPES)
        raise stripe.error.PermissionError(
            f"Stripe restricted key missing required scope. "
            f"This key needs read access to: {scopes}. "
            f"Underlying error: {e}"
        ) from e

    # ── Full-history walk ─────────────────────────────────────────────────────
    created_gte = int(pd.Timestamp(SERIES_START).timestamp())
    print(f"  Pulling charges created >= {SERIES_START} (gte={created_gte})…")

    rows: list[dict[str, Any]] = []
    iterator = stripe.Charge.list(
        created={"gte": created_gte},
        limit=100,
        expand=expand,
    ).auto_paging_iter()

    for charge in iterator:
        rows.append(_charge_to_row(charge))

    print(f"  Pulled {len(rows):,} charges from Stripe")

    # ── DataFrame assembly ────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    # Empty result: still write a header-only CSV so downstream phases get a
    # valid (empty) input.
    if not df.empty:
        df = df.sort_values(by=["Created date (UTC)", "id"], kind="stable").reset_index(drop=True)

    # ── Atomic write ──────────────────────────────────────────────────────────
    # Local: tmp-suffix + os.replace inside the storage helper. S3: PutObject is
    # itself per-key atomic, so the helper writes directly to the final key.
    storage.save_csv_atomic(SNAPSHOT_KEY, df, index=False)

    print(f"  Snapshot written → {SNAPSHOT_KEY} ({len(df)} rows, {len(df.columns)} columns)")

    return {
        "input_source": "stripe_api",
        "input_path": SNAPSHOT_KEY,
        "row_count": len(df),
    }
