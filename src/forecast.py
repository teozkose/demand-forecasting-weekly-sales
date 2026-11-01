"""One-week-ahead forecasts over a rolling holdout, four methods, honest error reporting.

Methods
  naive            next week = this week
  seasonal_naive   next week = same week last year
  holt_winters     exponential smoothing with damped trend and yearly seasonality (statsmodels)
  gradient_boost   one model across all products on lag, rolling-mean and calendar features (scikit-learn)

Evaluation: the last HOLDOUT weeks of each series. Every forecast for week t uses only data up to t-1
(the statistical models are refit at each step; the boosting model is trained once on the pre-holdout
weeks and fed actual lags, which is the standard one-step-ahead setup). Errors: MAE and WAPE
(sum |error| / sum actual), which stays defined when a product has zero-sales weeks.
"""
import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.prepare import PRODUCTS, WEEKLY

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
METRICS = os.path.join(RESULTS, "metrics.json")
FORECASTS = os.path.join(RESULTS, "forecasts.csv")
ERRORS = os.path.join(RESULTS, "errors_by_product.csv")
HOLDOUT = 12
SEASON = 52
LAGS = [1, 2, 3, 4, 8, 52]
SEED = 42
warnings.filterwarnings("ignore")


def features(panel: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features so one model serves products of any size and the total: every lag is
    expressed relative to the trailing 12-week mean, and the target is units / that mean."""
    out = []
    for sku, g in panel.groupby("sku"):
        g = g.sort_values("week").copy()
        base = g["units"].shift(1).rolling(12).mean().clip(lower=1.0)
        g["base"] = base
        for lag in LAGS:
            g[f"lag_{lag}"] = g["units"].shift(lag) / base
        g["roll_mean_4"] = g["units"].shift(1).rolling(4).mean() / base
        g["week_of_year"] = g["week"].dt.isocalendar().week.astype(int)
        g["target"] = g["units"] / base
        out.append(g)
    return pd.concat(out, ignore_index=True)


def stat_forecasts(y: pd.Series) -> dict:
    """One-step-ahead forecasts for the last HOLDOUT points of y, refitting at every step."""
    n = len(y)
    naive, snaive, hw = [], [], []
    for t in range(n - HOLDOUT, n):
        hist = y.iloc[:t]
        naive.append(hist.iloc[-1])
        snaive.append(hist.iloc[-SEASON] if t >= SEASON else hist.iloc[-1])
        # Two full years would be needed for a yearly seasonal component; the history here is shorter,
        # so this is Holt's damped-trend smoothing without a seasonal term.
        model = ExponentialSmoothing(hist.astype(float), trend="add", damped_trend=True,
                                     initialization_method="estimated").fit(optimized=True)
        hw.append(max(float(model.forecast(1).iloc[0]), 0.0))
    return {"naive": naive, "seasonal_naive": snaive, "holt_damped": hw}


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / max(actual.sum(), 1e-9))


def plot(fc: pd.DataFrame, errors: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    tot = fc[fc["sku"] == "TOTAL"].sort_values("week")
    axes[0].plot(tot["week"], tot["actual"], color="black", marker="o", label="actual")
    for m, style in [("seasonal_naive", "--"), ("holt_damped", "-."), ("gradient_boost", "-")]:
        axes[0].plot(tot["week"], tot[m], style, label=m)
    axes[0].set_title(f"All products, last {HOLDOUT} weeks: one-week-ahead forecasts")
    axes[0].set_ylabel("units per week")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)
    prod = errors[errors["sku"] != "TOTAL"]
    axes[1].bar(prod.columns[1:], prod.iloc[:, 1:].median())
    axes[1].set_title("Median WAPE across the 20 products")
    axes[1].set_ylabel("WAPE")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    panel = pd.read_parquet(WEEKLY)
    feat = features(panel).dropna()
    cols = [c for c in feat.columns if c.startswith(("lag_", "roll_"))] + ["week_of_year"]
    last_week = panel["week"].max()
    cutoff = sorted(panel["week"].unique())[-HOLDOUT]
    train = feat[feat["week"] < cutoff]
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=20,
                                       random_state=SEED).fit(train[cols], train["target"])

    rows = []
    for sku, g in panel.groupby("sku"):
        y = g.sort_values("week")["units"].reset_index(drop=True)
        weeks = g.sort_values("week")["week"].reset_index(drop=True)
        stats = stat_forecasts(y)
        test = feat[(feat["sku"] == sku) & (feat["week"] >= cutoff)].sort_values("week")
        gb_pred = (gb.predict(test[cols]).clip(0, 5) * test["base"].values)
        for i, t in enumerate(range(len(y) - HOLDOUT, len(y))):
            rows.append({"sku": sku, "week": weeks[t], "actual": y[t], "naive": stats["naive"][i],
                         "seasonal_naive": stats["seasonal_naive"][i], "holt_damped": stats["holt_damped"][i],
                         "gradient_boost": float(gb_pred[i])})
    fc = pd.DataFrame(rows)
    fc.to_csv(FORECASTS, index=False)

    methods = ["naive", "seasonal_naive", "holt_damped", "gradient_boost"]
    errors = fc.groupby("sku").apply(lambda g: pd.Series({m: wape(g["actual"], g[m]) for m in methods})).reset_index()
    errors.to_csv(ERRORS, index=False)
    prod = fc[fc["sku"] != "TOTAL"]
    tot = fc[fc["sku"] == "TOTAL"]
    summary = {m: {"products_wape_pooled": round(wape(prod["actual"], prod[m]), 3),
                   "products_wape_median": round(float(errors.loc[errors["sku"] != "TOTAL", m].median()), 3),
                   "products_mae": round(float((prod["actual"] - prod[m]).abs().mean()), 1),
                   "total_wape": round(wape(tot["actual"], tot[m]), 3),
                   "total_mae": round(float((tot["actual"] - tot[m]).abs().mean()), 0)} for m in methods}
    wins = errors[errors["sku"] != "TOTAL"].set_index("sku")[methods].idxmin(axis=1).value_counts().to_dict()
    best_products = min(methods, key=lambda m: summary[m]["products_wape_pooled"])
    best_total = min(methods, key=lambda m: summary[m]["total_wape"])
    plot(fc, errors, os.path.join(RESULTS, "forecasts.png"))

    m = {"setup": {"products": int(panel["sku"].nunique() - 1), "weeks": int(panel["week"].nunique()),
                   "holdout_weeks": HOLDOUT, "holdout_start": str(pd.Timestamp(cutoff).date()),
                   "holdout_end": str(last_week.date()), "lags": LAGS},
         "methods": summary, "best_method_products": best_products, "best_method_total": best_total,
         "products_won_by_method": wins}
    json.dump(m, open(METRICS, "w"), indent=2)
    print(pd.DataFrame(summary).T.to_string())
    print("products won:", wins)
    print(f"saved {METRICS}")
