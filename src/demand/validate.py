"""Checks that fail loudly: a run stops with the full list of problems instead of producing quiet nonsense."""

from __future__ import annotations

import pandas as pd

from demand.data import RAW_COLUMNS


class DataValidationError(ValueError):
    """Raised when a table breaks one or more checks; the message lists every problem found."""


def _raise(problems: list[str], what: str) -> None:
    if problems:
        raise DataValidationError(f"{what} failed {len(problems)} check(s):\n- " + "\n- ".join(problems))


def validate_raw(df: pd.DataFrame, min_rows: int = 1000) -> None:
    """Schema, types, nulls in key fields and date range of the raw invoice lines."""
    problems: list[str] = []
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        _raise([f"missing columns: {missing}"], "raw data")
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (expected at least {min_rows})")
    for c in ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"]:
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{n} nulls in {c}")
    if not pd.api.types.is_numeric_dtype(df["Quantity"]):
        problems.append("Quantity is not numeric")
    if not pd.api.types.is_numeric_dtype(df["Price"]):
        problems.append("Price is not numeric")
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        problems.append("InvoiceDate is not a datetime")
    else:
        lo, hi = df["InvoiceDate"].min(), df["InvoiceDate"].max()
        if lo < pd.Timestamp("2009-01-01") or hi > pd.Timestamp("2012-12-31"):
            problems.append(f"InvoiceDate outside 2009-2012: {lo} to {hi}")
    if pd.api.types.is_numeric_dtype(df["Quantity"]) and (df["Quantity"].abs() > 1_000_000).any():
        problems.append("Quantity has values beyond one million units")
    _raise(problems, "raw data")


def validate_panel(panel: pd.DataFrame, total_name: str, min_weeks: int = 60) -> None:
    """The weekly panel must be complete, non-negative, unique per (sku, week) and on a 7-day grid."""
    problems: list[str] = []
    for c in ["sku", "week", "units"]:
        if c not in panel.columns:
            _raise([f"missing column {c}"], "weekly panel")
    if panel[["sku", "week", "units"]].isna().any().any():
        problems.append("nulls in sku, week or units")
    dupes = int(panel.duplicated(["sku", "week"]).sum())
    if dupes:
        problems.append(f"{dupes} duplicate (sku, week) rows")
    if (panel["units"] < 0).any():
        problems.append("negative weekly units")
    counts = panel.groupby("sku")["week"].nunique()
    if counts.nunique() > 1:
        problems.append("series have different lengths (week grid not complete)")
    if counts.min() < min_weeks:
        problems.append(f"shortest series has {counts.min()} weeks (need {min_weeks})")
    weeks = pd.Series(sorted(panel["week"].unique()))
    gaps = weeks.diff().dropna().dt.days.unique()
    if len(gaps) and set(gaps) != {7}:
        problems.append(f"week grid is not every 7 days: gaps {sorted(gaps)}")
    if total_name not in set(panel["sku"]):
        problems.append(f"total series {total_name!r} missing")
    else:
        prod = panel[panel["sku"] != total_name].groupby("week")["units"].sum()
        tot = panel[panel["sku"] == total_name].set_index("week")["units"]
        if (prod.reindex(tot.index) > tot + 1e-6).any():
            problems.append("product units exceed the total in some week")
    _raise(problems, "weekly panel")
