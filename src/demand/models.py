"""Forecasting methods.

Per-series (refit at every cutoff, h-step forecasts from the last observed week):
  naive            the last observed week, repeated
  seasonal_naive   the same week one year earlier (falls back to naive without a year of history)
  ets              Holt's damped-trend exponential smoothing; 80% interval from in-sample residuals
  theta            the Theta method (statsmodels ThetaModel) with its analytic 80% interval
Global (one model across all products, a direct model per horizon):
  gbm              histogram gradient boosting on scale-free lags, rolling stats and holiday features;
                   two quantile models give the 10th and 90th percentiles
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

POINT_MODELS = ["naive", "seasonal_naive", "ets", "theta", "gbm"]
INTERVAL_MODELS = ["ets", "theta", "gbm"]


def naive_forecast(history: np.ndarray, horizon: int) -> float:
    """Last observed value."""
    return float(history[-1])


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = 52) -> float:
    """Value from the same week one season before the target; naive when the history is too short."""
    idx = len(history) - season + horizon - 1
    return float(history[idx]) if 0 <= idx < len(history) else float(history[-1])


def ets_forecast(history: np.ndarray, horizon: int, z: float) -> tuple[float, float, float]:
    """Damped-trend Holt forecast h steps ahead with a normal interval of +-z residual sd * sqrt(h).

    Two full years would be needed for a yearly seasonal component; the history here is shorter, so the
    model has no seasonal term and calendar effects are left to the global model.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(history.astype(float), trend="add", damped_trend=True,
                                   initialization_method="estimated").fit(optimized=True)
    point = max(float(fit.forecast(horizon)[-1]), 0.0)
    sd = float(np.std(history - fit.fittedvalues, ddof=1)) * np.sqrt(horizon)
    return point, max(point - z * sd, 0.0), point + z * sd


def theta_forecast(history: np.ndarray, horizon: int, alpha: float) -> tuple[float, float, float]:
    """Theta method forecast h steps ahead with its (1 - alpha) prediction interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ThetaModel(pd.Series(history.astype(float)), period=1, deseasonalize=False).fit()
        point = float(res.forecast(horizon).iloc[-1])
        pi = res.prediction_intervals(horizon, alpha=alpha).iloc[-1]
    return max(point, 0.0), max(float(pi.iloc[0]), 0.0), max(float(pi.iloc[1]), 0.0)


def series_forecasts(history: np.ndarray, horizon: int, season: int, lower: float, upper: float) -> dict[str, float]:
    """Every per-series method for one history and horizon, with interval bounds where available."""
    z = float(norm.ppf(upper))
    out = {"naive": naive_forecast(history, horizon),
           "seasonal_naive": seasonal_naive_forecast(history, horizon, season)}
    out["ets"], out["ets_lo"], out["ets_hi"] = ets_forecast(history, horizon, z)
    out["theta"], out["theta_lo"], out["theta_hi"] = theta_forecast(history, horizon, alpha=1 - (upper - lower))
    return out


class GlobalGBM:
    """Point and quantile gradient boosting on scale-free features; predictions are scaled back by base."""

    def __init__(self, features: list[str], params: dict[str, Any], lower: float = 0.1, upper: float = 0.9,
                 seed: int = 42) -> None:
        self.features = features
        self.params = params
        self.lower, self.upper, self.seed = lower, upper, seed
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _make(self, loss: str, quantile: float | None = None) -> HistGradientBoostingRegressor:
        kw = {"loss": loss, "random_state": self.seed, **self.params}
        if quantile is not None:
            kw["quantile"] = quantile
        return HistGradientBoostingRegressor(**kw)

    def fit(self, rows: pd.DataFrame) -> GlobalGBM:
        """Fit on rows with a defined target (units / base)."""
        rows = rows.dropna(subset=["target"])
        x, y = rows[self.features], rows["target"].clip(upper=10)
        self.models = {
            "point": self._make("squared_error").fit(x, y),
            "lo": self._make("quantile", self.lower).fit(x, y),
            "hi": self._make("quantile", self.upper).fit(x, y),
        }
        return self

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Point, lower and upper forecasts in units; bounds are sorted so lo <= point <= hi."""
        x = rows[self.features]
        base = rows["base"].to_numpy()
        preds = np.column_stack([self.models[k].predict(x) for k in ("lo", "point", "hi")]).clip(0, 10)
        preds = np.sort(preds, axis=1) * base[:, None]
        return pd.DataFrame(preds, columns=["gbm_lo", "gbm", "gbm_hi"], index=rows.index)


def tune_gbm(rows: pd.DataFrame, features: list[str], fit_end: pd.Timestamp, valid_end: pd.Timestamp,
             base_params: dict[str, Any], grid: dict[str, list[Any]], seed: int = 42) -> tuple[dict[str, Any], list]:
    """Pick learning rate and depth on a time-ordered block: fit on weeks <= fit_end, score pooled WAPE
    on (fit_end, valid_end]. Returns the best params and every grid score."""
    train = rows[(rows["week"] <= fit_end)].dropna(subset=["target"])
    valid = rows[(rows["week"] > fit_end) & (rows["week"] <= valid_end)].dropna(subset=["target"])
    scores = []
    for lr, depth in product(grid["learning_rate"], grid["max_depth"]):
        params = {**base_params, "learning_rate": lr, "max_depth": depth}
        m = HistGradientBoostingRegressor(random_state=seed, **params).fit(train[features], train["target"].clip(upper=10))
        pred = m.predict(valid[features]).clip(0, 10) * valid["base"].to_numpy()
        err = float(np.abs(pred - valid["units"]).sum() / max(valid["units"].sum(), 1e-9))
        scores.append({"learning_rate": lr, "max_depth": depth, "wape": round(err, 4)})
    best = min(scores, key=lambda s: s["wape"])
    return {**base_params, "learning_rate": best["learning_rate"], "max_depth": best["max_depth"]}, scores
