"""Scale-free lag, rolling and holiday features for one global model across products of any size.

For a forecast h weeks ahead, the row for target week t may only use units observed up to week t - h.
Every lag and rolling statistic is divided by the trailing mean of the base_window weeks ending at t - h,
and the target is units / that mean, so a product selling 50 a week and the all-products total share
one model.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

CALENDAR = ["week_of_year", "month", "weeks_to_christmas", "bank_holiday_week"]


def uk_bank_holidays(years: list[int]) -> list[date]:
    """England and Wales bank holidays (without one-off royal days); weekend dates are moved to Monday."""
    out: list[date] = []
    for y in years:
        e = easter(y)
        fixed = [date(y, 1, 1), date(y, 12, 25), date(y, 12, 26)]
        for d in fixed:
            while d.weekday() >= 5:
                d += timedelta(days=1)
            out.append(d)
        if out[-1] == out[-2]:  # Christmas moved onto Boxing Day: push Boxing Day one more day
            out[-1] = out[-1] + timedelta(days=1)
        first_may = date(y, 5, 1)
        early_may = first_may + timedelta(days=(7 - first_may.weekday()) % 7)
        last_may = date(y, 5, 31) - timedelta(days=date(y, 5, 31).weekday())
        last_aug = date(y, 8, 31) - timedelta(days=date(y, 8, 31).weekday())
        out += [e - timedelta(days=2), e + timedelta(days=1), early_may, last_may, last_aug]
    return sorted(out)


def calendar_features(weeks: pd.Series) -> pd.DataFrame:
    """Week of year, month, whole weeks until Christmas Day and a bank-holiday-in-week flag."""
    weeks = pd.to_datetime(weeks)
    years = sorted(set(weeks.dt.year) | set(weeks.dt.year + 1))
    hol = pd.to_datetime(pd.Series(uk_bank_holidays(years)))
    hol_weeks = set(hol.dt.to_period("W-SUN").dt.start_time)
    xmas = pd.to_datetime(weeks.dt.year.astype(str) + "-12-25")
    days = (xmas - weeks).dt.days
    days = days.where(days >= 0, days + 365)  # after Christmas Day, count to next Christmas
    return pd.DataFrame({
        "week_of_year": weeks.dt.isocalendar().week.astype(int).to_numpy(),
        "month": weeks.dt.month.to_numpy(),
        "weeks_to_christmas": np.floor(days.to_numpy() / 7).clip(min=0),
        "bank_holiday_week": weeks.isin(hol_weeks).astype(int).to_numpy(),
    }, index=weeks.index)


def make_features(panel: pd.DataFrame, horizon: int, lags: list[int], seasonal_lag: int = 52,
                  base_window: int = 12) -> pd.DataFrame:
    """One row per (sku, target week) with features that only look at weeks up to target - horizon.

    Columns: sku, week, units, base, target (units / base), lag_k for k in lags, seasonal lag, rolling
    mean/std/zero share, log base (product size) and the calendar features. Rows whose base is not yet
    defined are kept with NaN features; the gradient-boosting models handle missing values natively.
    """
    out = []
    for sku, g in panel.groupby("sku", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        u = g["units"].astype(float)
        seen = u.shift(horizon)  # the most recent week visible when forecasting h ahead
        base = seen.rolling(base_window, min_periods=base_window).mean().clip(lower=1.0)
        f = pd.DataFrame({"sku": sku, "week": g["week"], "units": u, "base": base})
        for k in lags:
            f[f"lag_{k}"] = u.shift(horizon + k - 1) / base
        f[f"lag_{seasonal_lag}"] = u.shift(seasonal_lag) / base if seasonal_lag >= horizon else np.nan
        f["roll_mean_4"] = seen.rolling(4).mean() / base
        f["roll_std_8"] = seen.rolling(8).std() / base
        f["zero_share_8"] = (seen == 0).astype(float).rolling(8).mean()
        f["log_base"] = np.log1p(base)
        f["target"] = u / base
        out.append(pd.concat([f, calendar_features(g["week"])], axis=1))
    return pd.concat(out, ignore_index=True)


def feature_columns(lags: list[int], seasonal_lag: int = 52) -> list[str]:
    """The model inputs, in a fixed order."""
    return ([f"lag_{k}" for k in lags] + [f"lag_{seasonal_lag}", "roll_mean_4", "roll_std_8", "zero_share_8",
                                           "log_base"] + CALENDAR)
