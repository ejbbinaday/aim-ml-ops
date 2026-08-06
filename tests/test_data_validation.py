"""Data-validation tests (Milestone 2 — Deliverable 2a).

The Pandera schema in `src/revenue/validation.py` is the pipeline's enforced
data-quality gate: the `validate` task fails the whole run on bad data. These
tests prove the gate rejects each class of violation the schema guards against
(missing required columns, wrong types, out-of-range values) and that a valid
extract passes through untouched.
"""

from __future__ import annotations

import pandera as pa
import pytest

from src.revenue.validation import validate_raw_charges

# ── Valid data passes ────────────────────────────────────────────────────────


def test_schema_accepts_valid_data(raw_frame):
    out = validate_raw_charges(raw_frame)
    assert len(out) == 5


# ── Missing required column ──────────────────────────────────────────────────


@pytest.mark.parametrize("column", ["Amount", "Status", "Created date (UTC)"])
def test_schema_rejects_missing_required_column(raw_frame, column):
    bad = raw_frame.drop(columns=[column])
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_charges(bad)


# ── Wrong type ───────────────────────────────────────────────────────────────


def test_schema_rejects_non_numeric_amount(raw_frame):
    # A Paid charge with a non-numeric amount is a data-integrity failure the
    # gate must surface loudly — silently coercing to 0 would understate revenue.
    bad = raw_frame
    bad["Amount"] = bad["Amount"].astype(object)
    bad.loc[0, "Amount"] = "not-a-number"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_charges(bad)


# ── Out-of-range values ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_date",
    [
        "2099-01-01 00:00:00",  # future-dated typo — would be silently dropped downstream
        "2009-06-15 00:00:00",  # pre-2015 — almost certainly a parse error
    ],
)
def test_schema_rejects_out_of_range_dates(raw_frame, bad_date):
    bad = raw_frame
    bad.loc[0, "Created date (UTC)"] = bad_date
    with pytest.raises(pa.errors.SchemaErrors):
        validate_raw_charges(bad)
