"""Data-quality contract for the raw payments extract (Pandera).

This is the enforced quality gate of the Milestone 1 pipeline: the `validate`
task runs `validate_raw_charges` on the extracted DataFrame and the pipeline
*fails* (raises `pandera.errors.SchemaError`) if the data violates the schema.
Nothing downstream runs on invalid data.

The schema mirrors what `phase1_data.load_and_clean` actually depends on:
  • Status              — transaction status; we later keep only "Paid".
  • Amount              — charge amount; must be coercible to float.
  • Description         — free-text used to categorise the revenue stream
                          (IV League / Core MRR / Core one-time). Nullable:
                          blank descriptions fall through to Core one-time.
  • Created date (UTC)  — timestamp; drives the month-start aggregation.
  • Currency (optional) — when present, non-AUD rows are dropped upstream.
"""

from __future__ import annotations

import pandas as pd
from pandera import Check, Column, DataFrameSchema

# Columns phase 1 reads. Description is nullable (blank → Core one-time).
RAW_CHARGES_SCHEMA = DataFrameSchema(
    {
        "Status": Column(str, nullable=False, coerce=True),
        # Deliberately stricter than clean_raw's `to_numeric(...).fillna(0.0)`
        # fallback: at the validate gate a Paid charge with a blank/non-numeric
        # amount is a data-integrity failure we want to surface loudly, not
        # silently coerce to 0 (which would understate revenue).
        "Amount": Column(float, nullable=False, coerce=True),
        "Description": Column(str, nullable=True, coerce=True),
        "Created date (UTC)": Column(
            "datetime64[ns]",
            nullable=False,
            coerce=True,
            # Bound both sides: a pre-2015 date is a likely parse error, and a
            # future-dated charge (data-entry typo, e.g. year 2099) would pass
            # a lower-bound-only check and then be silently dropped by the
            # month-start aggregation's SERIES_END cap — understating revenue
            # with no warning. Reject both here at the gate.
            checks=Check(
                lambda s: (s.dt.year >= 2015) & (s <= pd.Timestamp.now() + pd.Timedelta(days=1)),
                error="Created date (UTC) out of range (before 2015 or in the future)",
            ),
        ),
    },
    # Currency, id, org metadata etc. may be present; don't reject extra columns.
    strict=False,
    # Ordering of the extract is not part of the contract.
    ordered=False,
    coerce=True,
    name="raw_charges",
)


def validate_raw_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw charges extract, returning the coerced frame.

    Raises ``pandera.errors.SchemaErrors`` (plural — ``lazy=True`` aggregates
    every failure into one report) if any required column is missing or
    malformed. This is the pipeline's hard data-quality gate.
    """
    return RAW_CHARGES_SCHEMA.validate(df, lazy=True)
