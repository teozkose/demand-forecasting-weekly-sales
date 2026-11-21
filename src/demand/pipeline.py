"""End-to-end pipeline. Steps: data (download + validate raw), panel (clean + weekly series + validate),
run (tune, backtest, evaluate, figures, metrics.json), report (README + docs page).

Usage: python -m demand.pipeline {data,panel,run,report,all}
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from demand import evaluate as ev
from demand import figures as fg
from demand.backtest import backtest, target_weeks
from demand.config import load_config, resolve
from demand.data import build_panel, clean, data_summary, download, load_raw
from demand.features import feature_columns, make_features
from demand.models import POINT_MODELS, tune_gbm
from demand.validate import validate_panel, validate_raw


def step_data(cfg: dict[str, Any]) -> None:
    """Download (or reuse) the raw invoice lines and validate them."""
    path = download(cfg["data"]["url"], resolve(cfg["data"]["raw"]), cfg["data"]["min_raw_bytes"])
    validate_raw(load_raw(path))
    print("raw data valid")


def step_panel(cfg: dict[str, Any]) -> None:
    """Clean the lines, build the weekly panel, validate it and save the data summary."""
    raw = load_raw(resolve(cfg["data"]["raw"]))
    validate_raw(raw)
    lines, steps = clean(raw, cfg["data"]["country"])
    p = cfg["panel"]
    panel, products = build_panel(lines, p["top_n"], p["min_weeks_sold"], p["total_series"])
    validate_panel(panel, p["total_series"])
    panel.to_parquet(resolve(cfg["data"]["panel"]), index=False)
    results = resolve(cfg["paths"]["results"])
    results.mkdir(exist_ok=True)
    products.to_csv(results / "products.csv", index=False)
    summary = data_summary(raw, lines, panel, products, p["total_series"])
    summary["cleaning"] = steps
    (results / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"panel: {summary['products']} products + total, {summary['weeks']} weeks "
          f"({summary['first_week']} to {summary['last_week']})")


def tune(panel: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], list]:
    """Choose boosting depth and learning rate on the weeks just before the backtest (1 week ahead)."""
    f, g, b = cfg["features"], cfg["models"]["gbm"], cfg["backtest"]
    feat = make_features(panel, 1, f["lags"], f["seasonal_lag"], f["base_window"])
    feat = feat[(feat["sku"] != cfg["panel"]["total_series"]) & feat["base"].notna()]
    first_target = target_weeks(panel["week"], b["target_weeks"])[0]
    weeks = sorted(w for w in pd.unique(panel["week"]) if w < first_target)
    valid_end, fit_end = pd.Timestamp(weeks[-1]), pd.Timestamp(weeks[-1 - g["tuning_weeks"]])
    base = {"max_iter": g["max_iter"], "min_samples_leaf": g["min_samples_leaf"]}
    return tune_gbm(feat, feature_columns(f["lags"], f["seasonal_lag"]), fit_end, valid_end, base, g["grid"],
                    cfg["evaluation"]["seed"])


def step_run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Backtest every method and write forecasts, tables, figures and metrics.json."""
    results = resolve(cfg["paths"]["results"])
    p, b, e, iv = cfg["panel"], cfg["backtest"], cfg["evaluation"], cfg["models"]["interval"]
    panel = pd.read_parquet(resolve(cfg["data"]["panel"]))
    validate_panel(panel, p["total_series"])
    products = pd.read_csv(results / "products.csv", dtype={"sku": str})
    params, grid = tune(panel, cfg)
    print("tuned:", params)

    fc = backtest(panel, cfg, params)
    fc.to_csv(results / "forecasts.csv", index=False)
    headline_n = b["target_weeks"] - b["calibration_weeks"]
    head = ev.headline_window(fc, headline_n)
    calib = fc[~fc["week"].isin(head["week"].unique())]
    nominal = iv["upper"] - iv["lower"]

    summary, intervals, totals = {}, {}, {}
    for h in b["horizons"]:
        prod = head[(head["horizon"] == h) & ~head["is_total"]]
        summary[str(h)] = ev.bootstrap_products(prod, POINT_MODELS, "naive", e["n_boot"], e["seed"])
        intervals[str(h)] = ev.interval_table(prod, nominal, e["n_boot"], e["seed"])
        tot = head[(head["horizon"] == h) & head["is_total"]]
        totals[str(h)] = {m: {"wape": round(ev.wape(tot["actual"], tot[m]), 4),
                              "bias": round(ev.bias(tot["actual"], tot[m]), 4)} for m in POINT_MODELS}

    h1 = head[(head["horizon"] == 1) & ~head["is_total"]]
    per = ev.per_product(h1, POINT_MODELS, e["n_boot"], e["seed"])
    per = per.merge(products[["sku", "description"]], on="sku", how="left")
    per.to_csv(results / "product_errors.csv", index=False)
    tiers = ev.volume_tiers(products)
    seg = ev.segment_errors(h1, tiers, pd.Timestamp(b["peak_start"]))
    seg.to_csv(results / "segment_errors.csv", index=False)
    calib1 = calib[(calib["horizon"] == 1) & ~calib["is_total"]]
    inv = ev.inventory_view(calib1, h1, POINT_MODELS, cfg["inventory"]["service_levels"], "gbm", iv["upper"])
    inv.to_csv(results / "inventory.csv", index=False)
    sens = ev.window_sensitivity(fc[(fc["horizon"] == 1) & ~fc["is_total"]], e["sensitivity_windows"])
    sens.to_csv(results / "window_sensitivity.csv", index=False)

    fg.total_backtest(fc, results / "backtest_total.png")
    fg.wape_with_ci(summary, results / "wape_by_model.png")
    fg.coverage_plot(intervals, nominal, results / "interval_coverage.png")
    fg.inventory_plot(inv, results / "inventory_tradeoff.png")
    fg.segment_plot(seg, POINT_MODELS, results / "segment_errors.png")

    best = min(POINT_MODELS, key=lambda m: summary["1"][m]["wape"])
    wins = per.loc[per.groupby("sku")["wape"].idxmin(), "model"].value_counts().to_dict()
    weeks = sorted(head["week"].unique())
    all_targets = sorted(fc["week"].unique())
    m = {
        "setup": {"horizons": b["horizons"], "headline_weeks": headline_n, "calibration_weeks": b["calibration_weeks"],
                  "headline_start": str(pd.Timestamp(weeks[0]).date()), "headline_end": str(pd.Timestamp(weeks[-1]).date()),
                  "calibration_start": str(pd.Timestamp(all_targets[0]).date()), "peak_start": b["peak_start"],
                  "n_boot": e["n_boot"], "nominal_coverage": nominal, "fits_per_series": len(all_targets) * len(b["horizons"])},
        "tuning": {"chosen": {k: params[k] for k in ("learning_rate", "max_depth")}, "grid": grid},
        "products": summary, "total": totals, "intervals": intervals,
        "best_model_h1": best, "products_won_h1": {k: int(v) for k, v in wins.items()},
        "tiers": {s: tiers[tiers == s].index.tolist() for s in ["high", "mid", "low"]},
        "segments": seg.round(4).to_dict(orient="records"),
        "inventory": inv.round(4).to_dict(orient="records"),
        "window_sensitivity": sens.round(4).to_dict(orient="records"),
    }
    (results / "metrics.json").write_text(json.dumps(m, indent=2))
    print(pd.DataFrame({h: {k: v["wape"] for k, v in s.items()} for h, s in summary.items()}).to_string())
    return m


def step_report(cfg: dict[str, Any]) -> None:
    """Render README.md and docs/index.html from the saved results."""
    from demand.report import render_all

    render_all(cfg)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["data", "panel", "run", "report", "all"])
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    steps = {"data": step_data, "panel": step_panel, "run": step_run, "report": step_report}
    for name in (["data", "panel", "run", "report"] if args.step == "all" else [args.step]):
        steps[name](cfg)


if __name__ == "__main__":
    main()
