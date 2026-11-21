"""Error metrics, bootstrap intervals, interval coverage, segment errors and the inventory view."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from demand.models import INTERVAL_MODELS, POINT_MODELS


def wape(actual: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: sum |error| / sum actual. Defined even with zero-sales weeks."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(a - p).sum() / max(a.sum(), 1e-9))


def mae(actual: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error in units."""
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(pred, float))))


def bias(actual: np.ndarray | pd.Series, pred: np.ndarray | pd.Series) -> float:
    """Signed bias as a share of actual units: positive means over-forecasting."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float((p - a).sum() / max(a.sum(), 1e-9))


def coverage(actual: np.ndarray | pd.Series, lo: np.ndarray | pd.Series, hi: np.ndarray | pd.Series) -> float:
    """Share of actuals inside [lo, hi]."""
    a = np.asarray(actual, float)
    return float(np.mean((a >= np.asarray(lo, float)) & (a <= np.asarray(hi, float))))


def headline_window(fc: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Rows whose target week is among the last n_weeks target weeks."""
    weeks = sorted(fc["week"].unique())[-n_weeks:]
    return fc[fc["week"].isin(weeks)]


def _sku_sums(prod: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    g = prod.assign(abs_err=(prod[model] - prod["actual"]).abs()).groupby("sku")
    return g["abs_err"].sum().to_numpy(), g["actual"].sum().to_numpy()


def bootstrap_products(prod: pd.DataFrame, models: list[str], baseline: str, n_boot: int,
                       seed: int) -> dict[str, dict[str, Any]]:
    """Pooled WAPE per model with a 95% CI from resampling products (the unit that varies), and the
    paired WAPE difference against the baseline on the same resampled products."""
    rng = np.random.default_rng(seed)
    n = prod["sku"].nunique()
    idx = rng.integers(0, n, size=(n_boot, n))
    base_err, act = _sku_sums(prod, baseline)
    base_boot = base_err[idx].sum(1) / act[idx].sum(1)
    out = {}
    for m in models:
        err, _ = _sku_sums(prod, m)
        boot = err[idx].sum(1) / act[idx].sum(1)
        diff = boot - base_boot
        out[m] = {
            "wape": round(float(err.sum() / act.sum()), 4),
            "wape_ci": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "vs_baseline": round(float(err.sum() / act.sum() - base_err.sum() / act.sum()), 4),
            "vs_baseline_ci": [round(float(x), 4) for x in np.percentile(diff, [2.5, 97.5])],
            "mae": round(mae(prod["actual"], prod[m]), 1),
            "bias": round(bias(prod["actual"], prod[m]), 4),
        }
    return out


def per_product(prod: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    """WAPE, MAE and bias per product and model, with a 95% CI on WAPE from resampling that product's weeks."""
    rng = np.random.default_rng(seed)
    rows = []
    for sku, g in prod.groupby("sku", sort=True):
        a = g["actual"].to_numpy(float)
        idx = rng.integers(0, len(a), size=(n_boot, len(a)))
        for m in models:
            p = g[m].to_numpy(float)
            e = np.abs(a - p)
            boot = e[idx].sum(1) / np.maximum(a[idx].sum(1), 1e-9)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({"sku": sku, "model": m, "wape": wape(a, p), "wape_lo": float(lo), "wape_hi": float(hi),
                         "mae": mae(a, p), "bias": bias(a, p)})
    return pd.DataFrame(rows)


def interval_table(prod: pd.DataFrame, nominal: float, n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    """Coverage of each model's interval (95% CI over products) and its width relative to actual units."""
    rng = np.random.default_rng(seed)
    skus = prod["sku"].unique()
    idx = rng.integers(0, len(skus), size=(n_boot, len(skus)))
    out = {}
    for m in INTERVAL_MODELS:
        inside = ((prod["actual"] >= prod[f"{m}_lo"]) & (prod["actual"] <= prod[f"{m}_hi"])).groupby(prod["sku"])
        hits, cnt = inside.sum().reindex(skus).to_numpy(), inside.size().reindex(skus).to_numpy()
        boot = hits[idx].sum(1) / cnt[idx].sum(1)
        out[m] = {
            "coverage": round(float(hits.sum() / cnt.sum()), 4),
            "coverage_ci": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "relative_width": round(float((prod[f"{m}_hi"] - prod[f"{m}_lo"]).sum() / prod["actual"].sum()), 3),
            "nominal": nominal,
        }
    return out


def volume_tiers(products: pd.DataFrame) -> pd.Series:
    """Split products into equal-count high, mid and low tiers by mean weekly units."""
    ranks = products.set_index("sku")["mean_weekly_units"].rank(method="first", ascending=False)
    n = len(ranks)
    return pd.cut(ranks, bins=[0, n / 3, 2 * n / 3, n], labels=["high", "mid", "low"]).astype(str)


def segment_errors(prod: pd.DataFrame, tiers: pd.Series, peak_start: pd.Timestamp,
                   models: list[str] = POINT_MODELS) -> pd.DataFrame:
    """Pooled WAPE by volume tier and by peak (pre-Christmas) versus earlier weeks."""
    d = prod.assign(tier=prod["sku"].map(tiers), period=np.where(prod["week"] >= peak_start, "peak", "pre-peak"))
    rows = []
    for col in ["tier", "period"]:
        for seg, g in d.groupby(col, sort=False):
            row = {"segment_type": col, "segment": seg, "products": g["sku"].nunique(), "weeks": g["week"].nunique(),
                   "actual_units": float(g["actual"].sum())}
            row.update({m: wape(g["actual"], g[m]) for m in models})
            rows.append(row)
    return pd.DataFrame(rows)


def inventory_view(calib: pd.DataFrame, test: pd.DataFrame, models: list[str], service_levels: list[float],
                   quantile_model: str = "gbm", quantile_level: float = 0.9) -> pd.DataFrame:
    """Safety stock and fill rate implied by each model's one-week-ahead errors.

    Weekly review, one-week lead time: before each week the planner stocks up to forecast + z * sigma,
    where sigma is that product's RMSE for that model on the calibration weeks (before the test weeks).
    Fill rate = units served from stock / units demanded; leftover = stock not sold that week. The
    quantile model's own upper bound is scored as a fourth policy at its nominal level.
    """
    rows = []
    for m in models:
        sigma = calib.groupby("sku").apply(lambda g, m=m: float(np.sqrt(np.mean((g[m] - g["actual"]) ** 2))))
        for sl in service_levels:
            z = float(norm.ppf(sl))
            ss = test["sku"].map(sigma).to_numpy() * z
            stock = np.maximum(test[m].to_numpy() + ss, 0)
            rows.append(_policy_row(m, sl, test, stock, ss))
    hi = test[f"{quantile_model}_hi"].to_numpy()
    rows.append(_policy_row(f"{quantile_model}_q{int(quantile_level * 100)}", quantile_level, test, hi,
                            hi - test[quantile_model].to_numpy()))
    return pd.DataFrame(rows)


def _policy_row(model: str, sl: float, test: pd.DataFrame, stock: np.ndarray, ss: np.ndarray) -> dict[str, Any]:
    a = test["actual"].to_numpy(float)
    weeks = test["week"].nunique()
    return {"model": model, "target_service": sl,
            "fill_rate": float(np.minimum(a, stock).sum() / max(a.sum(), 1e-9)),
            "weeks_short": float(np.mean(stock < a)),
            "safety_stock_per_week": float(ss.sum() / weeks),
            "leftover_per_week": float(np.maximum(stock - a, 0).sum() / weeks)}


def window_sensitivity(fc: pd.DataFrame, windows: list[int], models: list[str] = POINT_MODELS) -> pd.DataFrame:
    """Pooled product WAPE (1 week ahead) over the last n target weeks, for several n."""
    rows = []
    for n in windows:
        g = headline_window(fc, n)
        rows.append({"weeks": n, **{m: wape(g["actual"], g[m]) for m in models}})
    return pd.DataFrame(rows)
