import pandas as pd
import pytest
from conftest import synthetic_panel, synthetic_raw

from demand.data import build_panel, clean
from demand.validate import DataValidationError, validate_panel, validate_raw


def test_clean_counts_every_step():
    raw = synthetic_raw()
    lines, steps = clean(raw, "United Kingdom")
    assert steps[0]["removed"] == 5  # the five duplicated lines
    assert steps[-1]["remaining"] == len(lines)
    assert sum(s["removed"] for s in steps) + len(lines) == len(raw)
    assert (lines["units"] > 0).all() and lines["sku"].str.match(r"^\d{5}[A-Z]?$").all()


def test_build_panel_is_complete_and_valid():
    lines, _ = clean(synthetic_raw(), "United Kingdom")
    panel, products = build_panel(lines, top_n=2, min_weeks=10, total_name="TOTAL")
    assert len(products) == 2
    validate_panel(panel, "TOTAL", min_weeks=10)


def test_validate_raw_rejects_bad_schema_and_nulls():
    raw = synthetic_raw()
    validate_raw(raw)
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_raw(raw.drop(columns=["Price"]))
    bad = raw.copy()
    bad.loc[0, "Invoice"] = None
    with pytest.raises(DataValidationError, match="nulls in Invoice"):
        validate_raw(bad)


def test_validate_panel_lists_problems():
    p = synthetic_panel()
    bad = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    bad.loc[1, "units"] = -1
    with pytest.raises(DataValidationError) as err:
        validate_panel(bad, "TOTAL")
    msg = str(err.value)
    assert "duplicate" in msg and "negative" in msg
