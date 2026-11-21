import numpy as np
import pandas as pd

from demand import evaluate as ev
from demand.backtest import backtest, target_weeks
from demand.models import POINT_MODELS, naive_forecast, seasonal_naive_forecast
from demand.pipeline import tune


def test_baselines_index_correctly():
    y = np.arange(60, dtype=float)
    assert naive_forecast(y, 3) == 59
    # target is index 59 + 3 = 62; same week a year earlier is index 10
    assert seasonal_naive_forecast(y, 3, season=52) == 10
    assert seasonal_naive_forecast(y[:20], 1, season=52) == 19  # too short: falls back to naive


def test_backtest_end_to_end_on_synthetic_panel(panel, small_cfg):
    params, grid = tune(panel, small_cfg)
    assert len(grid) == 1 and params["max_depth"] == 3
    fc = backtest(panel, small_cfg, params)
    n_series = panel["sku"].nunique()
    assert len(fc) == n_series * 6 * 2
    assert set(fc["week"].unique()) == set(target_weeks(panel["week"], 6))
    assert (fc["cutoff"] < fc["week"]).all()
    gap = (fc["week"] - fc["cutoff"]).dt.days // 7
    assert (gap == fc["horizon"]).all()
    for m in ["ets", "theta", "gbm"]:
        assert (fc[f"{m}_lo"] <= fc[m] + 1e-9).all() and (fc[m] <= fc[f"{m}_hi"] + 1e-9).all()
    assert fc[POINT_MODELS].notna().all().all() and (fc[POINT_MODELS] >= 0).all().all()
    prod = fc[(fc["horizon"] == 1) & ~fc["is_total"]]
    res = ev.bootstrap_products(prod, POINT_MODELS, "naive", 50, 0)
    # a smooth seasonal series: a fitted model should not be far worse than last week
    assert min(res[m]["wape"] for m in ["ets", "theta", "gbm"]) < res["naive"]["wape"] * 1.2
    assert isinstance(pd.Timestamp(fc["week"].max()), pd.Timestamp)
