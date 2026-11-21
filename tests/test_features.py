from datetime import date

import numpy as np
import pandas as pd

from demand.features import calendar_features, feature_columns, make_features, uk_bank_holidays


def test_bank_holidays_2010_and_weekend_substitution():
    hol = uk_bank_holidays([2010, 2011])
    assert date(2010, 4, 2) in hol and date(2010, 4, 5) in hol  # Good Friday, Easter Monday
    assert date(2010, 12, 27) in hol and date(2010, 12, 28) in hol  # Christmas and Boxing Day fell on a weekend
    assert date(2011, 5, 30) in hol and date(2011, 8, 29) in hol


def test_calendar_weeks_to_christmas():
    weeks = pd.Series(pd.to_datetime(["2011-12-19", "2011-11-28", "2011-12-26"]))
    cal = calendar_features(weeks)
    assert list(cal["weeks_to_christmas"])[:2] == [0, 3]
    assert cal["weeks_to_christmas"].iloc[2] >= 50  # after Christmas, counts to next year
    assert cal.loc[2, "bank_holiday_week"] == 1


def test_features_do_not_leak_beyond_horizon(panel):
    h = 2
    lags = [1, 2, 3, 4, 8]
    base = make_features(panel, h, lags)
    target_week = sorted(panel["week"].unique())[60]
    changed = panel.copy()
    future = changed["week"] > target_week - pd.Timedelta(weeks=h)
    changed.loc[future, "units"] = changed.loc[future, "units"] * 100 + 7
    after = make_features(changed, h, lags)
    cols = feature_columns(lags)
    a = base[base["week"] == target_week].set_index("sku")[cols]
    b = after[after["week"] == target_week].set_index("sku")[cols]
    pd.testing.assert_frame_equal(a, b)


def test_scale_free_target_matches_units(panel):
    f = make_features(panel, 1, [1, 2]).dropna(subset=["base"])
    assert np.allclose(f["target"] * f["base"], f["units"])
    assert (f["base"] >= 1).all()
