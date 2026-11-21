"""Synthetic data shared by the tests; nothing here touches the network or data/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand.config import load_config


def synthetic_panel(n_products: int = 6, n_weeks: int = 80, seed: int = 0, total_name: str = "TOTAL") -> pd.DataFrame:
    """Weekly units with level differences, a Christmas bump, noise and some zero weeks, plus a total."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2010-01-04", periods=n_weeks, freq="7D")
    bump = np.where(weeks.month.isin([10, 11]), 1.6, 1.0)
    rows = []
    for i in range(n_products):
        level = 20 * (i + 1)
        units = rng.poisson(level * bump).astype(float)
        units[rng.random(n_weeks) < 0.05] = 0.0
        rows.append(pd.DataFrame({"sku": f"{10000 + i}", "week": weeks, "units": units}))
    panel = pd.concat(rows, ignore_index=True)
    total = panel.groupby("week")["units"].sum() * 1.5
    return pd.concat([panel, pd.DataFrame({"sku": total_name, "week": weeks, "units": total.to_numpy()})],
                     ignore_index=True)


def synthetic_raw(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Invoice lines in the source schema, with a few cancellations, returns, fees and duplicates."""
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2010-01-04") + pd.to_timedelta(rng.integers(0, 70 * 7 * 24, n), unit="h")
    df = pd.DataFrame({
        "Invoice": [f"{500000 + i // 3}" for i in range(n)],
        "StockCode": rng.choice(["10001", "10002", "10003A", "POST", "M"], n, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        "Description": "ITEM",
        "Quantity": rng.integers(1, 50, n),
        "InvoiceDate": ts,
        "Price": rng.uniform(0.5, 5, n).round(2),
        "Customer ID": 12345.0,
        "Country": rng.choice(["United Kingdom", "France"], n, p=[0.9, 0.1]),
    })
    df.loc[:9, "Invoice"] = "C" + df.loc[:9, "Invoice"]
    df.loc[10:19, "Quantity"] = -3
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


@pytest.fixture
def panel() -> pd.DataFrame:
    return synthetic_panel()


@pytest.fixture
def small_cfg() -> dict:
    """The real config, shrunk so a full backtest runs in seconds."""
    cfg = load_config()
    cfg["backtest"].update({"target_weeks": 6, "calibration_weeks": 2, "horizons": [1, 2]})
    cfg["models"]["gbm"].update({"max_iter": 30, "grid": {"learning_rate": [0.1], "max_depth": [3]}, "tuning_weeks": 4})
    cfg["evaluation"].update({"n_boot": 50, "sensitivity_windows": [2, 4]})
    return cfg
