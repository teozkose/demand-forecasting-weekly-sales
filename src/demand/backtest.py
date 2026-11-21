"""Rolling-origin backtest.

The target weeks are the last `target_weeks` weeks of the panel. For each horizon h and each target
week t, every method is fitted on weeks up to t - h only (the cutoff) and forecasts week t. All horizons
share the same target weeks, so 1-week and 4-week errors are directly comparable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from demand.features import feature_columns, make_features
from demand.models import GlobalGBM, series_forecasts


def target_weeks(weeks: pd.Series | pd.DatetimeIndex, n_targets: int) -> list[pd.Timestamp]:
    """The last n_targets weeks of the grid."""
    return [pd.Timestamp(w) for w in sorted(pd.unique(pd.Series(weeks)))[-n_targets:]]


def run_series_methods(panel: pd.DataFrame, targets: list[pd.Timestamp], horizons: list[int], season: int,
                       lower: float, upper: float) -> pd.DataFrame:
    """Naive, seasonal naive, ETS and Theta forecasts for every (sku, horizon, target week)."""
    rows = []
    for sku, g in panel.groupby("sku", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        y = g["units"].to_numpy(dtype=float)
        pos = {pd.Timestamp(w): i for i, w in enumerate(g["week"])}
        for h in horizons:
            for t in targets:
                i = pos[t]
                fc = series_forecasts(y[: i - h + 1], h, season, lower, upper)
                rows.append({"sku": sku, "horizon": h, "week": t, "cutoff": g["week"][i - h], "actual": y[i], **fc})
    return pd.DataFrame(rows)


def run_gbm(panel: pd.DataFrame, targets: list[pd.Timestamp], horizons: list[int], cfg: dict[str, Any],
            params: dict[str, Any], total_name: str) -> pd.DataFrame:
    """Direct global gradient boosting: one fit per (horizon, cutoff) on product rows whose target week is
    on or before the cutoff, then a forecast for the target week h ahead (products and total)."""
    f = cfg["features"]
    cols = feature_columns(f["lags"], f["seasonal_lag"])
    iv = cfg["models"]["interval"]
    seed = cfg["evaluation"]["seed"]
    weeks = [pd.Timestamp(w) for w in sorted(pd.unique(panel["week"]))]
    out = []
    for h in horizons:
        feat = make_features(panel, h, f["lags"], f["seasonal_lag"], f["base_window"])
        train_pool = feat[(feat["sku"] != total_name) & feat["base"].notna()]
        for t in targets:
            cutoff = weeks[weeks.index(t) - h]
            model = GlobalGBM(cols, params, iv["lower"], iv["upper"], seed).fit(train_pool[train_pool["week"] <= cutoff])
            test = feat[feat["week"] == t]
            pred = model.predict(test)
            out.append(pd.concat([test[["sku", "week"]].assign(horizon=h), pred], axis=1))
    return pd.concat(out, ignore_index=True)


def backtest(panel: pd.DataFrame, cfg: dict[str, Any], params: dict[str, Any]) -> pd.DataFrame:
    """All methods, all horizons, all target weeks, in one long table."""
    b = cfg["backtest"]
    iv = cfg["models"]["interval"]
    total_name = cfg["panel"]["total_series"]
    targets = target_weeks(panel["week"], b["target_weeks"])
    stats = run_series_methods(panel, targets, b["horizons"], b["season"], iv["lower"], iv["upper"])
    gbm = run_gbm(panel, targets, b["horizons"], cfg, params, total_name)
    fc = stats.merge(gbm, on=["sku", "week", "horizon"], how="left", validate="one_to_one")
    if fc["gbm"].isna().any():
        raise RuntimeError("gradient boosting forecasts missing for some rows")
    fc["is_total"] = fc["sku"] == total_name
    return fc.sort_values(["horizon", "sku", "week"]).reset_index(drop=True)
