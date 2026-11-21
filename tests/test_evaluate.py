import numpy as np
import pandas as pd
import pytest

from demand import evaluate as ev


def test_wape_mae_bias_known_values():
    a, p = np.array([10, 0, 30.0]), np.array([12, 2, 24.0])
    assert ev.wape(a, p) == pytest.approx(10 / 40)
    assert ev.mae(a, p) == pytest.approx(10 / 3)
    assert ev.bias(a, p) == pytest.approx(-2 / 40)


def test_wape_zero_actuals_does_not_divide_by_zero():
    assert np.isfinite(ev.wape([0, 0], [1, 1]))


def test_coverage_counts_bounds_inclusive():
    assert ev.coverage([1, 2, 3, 4], [1, 0, 4, 0], [1, 1, 5, 5]) == pytest.approx(0.5)


def _fake_forecasts(n_sku: int = 8, n_weeks: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sku):
        for w in range(n_weeks):
            a = float(rng.poisson(50))
            rows.append({"sku": str(s), "week": pd.Timestamp("2011-01-03") + pd.Timedelta(weeks=w), "actual": a,
                         "naive": a + rng.normal(0, 20), "seasonal_naive": a + rng.normal(0, 25),
                         "ets": a + rng.normal(0, 5), "theta": a + rng.normal(0, 8), "gbm": a + rng.normal(0, 10)})
    d = pd.DataFrame(rows)
    for m in ["ets", "theta", "gbm"]:
        d[f"{m}_lo"], d[f"{m}_hi"] = d[m] - 10, d[m] + 10
    return d


def test_bootstrap_ci_brackets_point_and_detects_better_model():
    d = _fake_forecasts()
    out = ev.bootstrap_products(d, ["naive", "ets"], "naive", n_boot=300, seed=1)
    lo, hi = out["ets"]["wape_ci"]
    assert lo <= out["ets"]["wape"] <= hi
    assert out["ets"]["vs_baseline_ci"][1] < 0  # clearly better than naive
    assert out["naive"]["vs_baseline"] == 0


def test_interval_table_and_volume_tiers():
    d = _fake_forecasts()
    iv = ev.interval_table(d, 0.8, n_boot=100, seed=0)
    assert set(iv) == {"ets", "theta", "gbm"}
    assert iv["ets"]["coverage"] > iv["gbm"]["coverage"] - 1e-9  # tighter errors, same width
    prods = pd.DataFrame({"sku": list("abcdef"), "mean_weekly_units": [60, 50, 40, 30, 20, 10]})
    tiers = ev.volume_tiers(prods)
    assert tiers["a"] == "high" and tiers["f"] == "low" and (tiers == "mid").sum() == 2


def test_inventory_more_service_means_more_stock():
    d = _fake_forecasts(n_weeks=12)
    weeks = sorted(d["week"].unique())
    calib, test = d[d["week"].isin(weeks[:4])], d[d["week"].isin(weeks[4:])]
    inv = ev.inventory_view(calib, test, ["naive", "ets"], [0.8, 0.95])
    for m in ["naive", "ets"]:
        g = inv[inv["model"] == m].sort_values("target_service")
        assert g["fill_rate"].is_monotonic_increasing
        assert g["leftover_per_week"].is_monotonic_increasing
    assert (inv["fill_rate"] <= 1).all() and "gbm_q90" in set(inv["model"])
