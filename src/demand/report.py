"""Render README.md (from README.template.md) and docs/index.html from results/, so every number and
every conclusion in the write-up comes from the last run."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from demand.config import ROOT, resolve
from demand.figures import LABELS
from demand.models import INTERVAL_MODELS, POINT_MODELS


def pct(x: float, digits: int = 1) -> str:
    """0.1234 -> '12.3%'."""
    return f"{100 * x:.{digits}f}%"


def ci(pair: list[float], digits: int = 3) -> str:
    """[a, b] -> 'a to b'."""
    return f"{pair[0]:.{digits}f} to {pair[1]:.{digits}f}"


def units(x: float) -> str:
    """Thousands separator, no decimals."""
    return f"{x:,.0f}"


def model_rows(m: dict[str, Any], h: str) -> list[list[str]]:
    """Accuracy table for one horizon: products pooled with CIs, plus the total series."""
    rows = []
    for k in POINT_MODELS:
        s, t = m["products"][h][k], m["total"][h][k]
        diff = "baseline" if k == "naive" else f"{s['vs_baseline']:+.3f} ({ci(s['vs_baseline_ci'])})"
        rows.append([LABELS[k], f"{s['wape']:.3f}", ci(s["wape_ci"]), diff, f"{s['mae']:.0f}", pct(s["bias"]),
                     f"{t['wape']:.3f}"])
    return rows


def interval_rows(m: dict[str, Any]) -> list[list[str]]:
    """Coverage and width of each 80% interval, by horizon."""
    rows = []
    for h, d in m["intervals"].items():
        for k in INTERVAL_MODELS:
            s = d[k]
            rows.append([f"{h}", LABELS[k], pct(s["coverage"]), f"{pct(s['coverage_ci'][0])} to {pct(s['coverage_ci'][1])}",
                         f"{s['relative_width']:.2f}"])
    return rows


def segment_rows(m: dict[str, Any]) -> list[list[str]]:
    """WAPE by volume tier and by season period."""
    return [[r["segment_type"], r["segment"], str(r["products"]), units(r["actual_units"])]
            + [f"{r[k]:.3f}" for k in POINT_MODELS] for r in m["segments"]]


def inventory_rows(m: dict[str, Any]) -> list[list[str]]:
    """Fill rate, short weeks, safety stock and leftover stock per policy."""
    return [[LABELS.get(r["model"], "Boosting 90th percentile as stock level"), pct(r["target_service"], 0),
             pct(r["fill_rate"]), pct(r["weeks_short"], 0), units(r["safety_stock_per_week"]),
             units(r["leftover_per_week"])] for r in m["inventory"]]


def sensitivity_rows(m: dict[str, Any]) -> list[list[str]]:
    """Pooled WAPE for different backtest windows."""
    return [[f"last {r['weeks']}"] + [f"{r[k]:.3f}" for k in POINT_MODELS] for r in m["window_sensitivity"]]


def product_rows(per: pd.DataFrame, best: str) -> list[list[str]]:
    """Per-product error of the best model, worst first, with week-bootstrap CIs."""
    d = per[per["model"] == best].sort_values("wape", ascending=False)
    naive = per[per["model"] == "naive"].set_index("sku")["wape"]
    return [[r.sku, str(r.description)[:34], f"{r.wape:.2f}", f"{r.wape_lo:.2f} to {r.wape_hi:.2f}", f"{r.mae:.0f}",
             pct(r.bias, 0), f"{naive[r.sku]:.2f}"] for r in d.itertuples()]


def cleaning_rows(s: dict[str, Any]) -> list[list[str]]:
    """Cleaning steps with counts."""
    return [[c["step"], units(c["removed"]), units(c["remaining"])] for c in s["cleaning"]]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    """GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def html_table(header: list[str], rows: list[list[str]]) -> str:
    """Plain HTML table with escaped cells."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


ACC_HEADER = ["Model", "Product WAPE", "95% CI", "vs naive (95% CI)", "MAE (units)", "Bias", "Total WAPE"]


def tables(m: dict[str, Any], per: pd.DataFrame, s: dict[str, Any]) -> dict[str, tuple[list[str], list[list[str]]]]:
    """Every table in the write-up, as (header, rows)."""
    return {
        "TABLE_H1": (ACC_HEADER, model_rows(m, "1")),
        "TABLE_H4": (ACC_HEADER, model_rows(m, "4")),
        "INTERVAL_TABLE": (["Weeks ahead", "Model", "Coverage", "95% CI", "Width / actual"], interval_rows(m)),
        "SEGMENT_TABLE": (["Split", "Segment", "Products", "Units"] + [LABELS[k] for k in POINT_MODELS], segment_rows(m)),
        "INVENTORY_TABLE": (["Forecast", "Target", "Fill rate", "Weeks short", "Safety stock / wk", "Leftover / wk"],
                            inventory_rows(m)),
        "SENSITIVITY_TABLE": (["Backtest window"] + [LABELS[k] for k in POINT_MODELS], sensitivity_rows(m)),
        "PRODUCT_TABLE": (["SKU", "Product", "WAPE", "95% CI", "MAE", "Bias", "Naive WAPE"], product_rows(per, m["best_model_h1"])),
        "CLEANING_TABLE": (["Step", "Lines removed", "Lines left"], cleaning_rows(s)),
    }


def values(m: dict[str, Any], s: dict[str, Any]) -> dict[str, str]:
    """Scalar placeholders for the templates."""
    best = m["best_model_h1"]
    p1, p4 = m["products"]["1"], m["products"]["4"]
    best_total = min(POINT_MODELS, key=lambda k: m["total"]["1"][k]["wape"])
    inv = pd.DataFrame(m["inventory"])
    sl = 0.95
    inv_best = inv[(inv["model"] == best) & (inv["target_service"] == sl)].iloc[0]
    inv_naive = inv[(inv["model"] == "naive") & (inv["target_service"] == sl)].iloc[0]
    seg = pd.DataFrame(m["segments"]).set_index("segment")
    tiers = seg[seg["segment_type"] == "tier"]
    sens = pd.DataFrame(m["window_sensitivity"]).set_index("weeks")
    sens_best = all(min(POINT_MODELS, key=lambda k, r=r: r[k]) == best for _, r in sens.iterrows())
    iv = m["intervals"]
    st = m["setup"]
    return {
        "BEST": LABELS[best], "BEST_WAPE": f"{p1[best]['wape']:.3f}", "BEST_CI": ci(p1[best]["wape_ci"]),
        "NAIVE_WAPE": f"{p1['naive']['wape']:.3f}", "CUT": pct(-p1[best]["vs_baseline"] / p1["naive"]["wape"], 0),
        "DIFF_CI": ci(p1[best]["vs_baseline_ci"]), "BEST_BIAS": pct(p1[best]["bias"], 0),
        "GBM_WAPE": f"{p1['gbm']['wape']:.3f}", "GBM_BIAS": pct(p1["gbm"]["bias"], 0),
        "BEST_H4": f"{p4[best]['wape']:.3f}", "GBM_H4": f"{p4['gbm']['wape']:.3f}", "NAIVE_H4": f"{p4['naive']['wape']:.3f}",
        "BEST_TOTAL": LABELS[best_total], "BEST_TOTAL_WAPE": f"{m['total']['1'][best_total]['wape']:.3f}",
        "NAIVE_TOTAL_WAPE": f"{m['total']['1']['naive']['wape']:.3f}",
        "GBM_TOTAL_H4": f"{m['total']['4']['gbm']['wape']:.3f}", "ETS_TOTAL_H4": f"{m['total']['4']['ets']['wape']:.3f}",
        "WINS": ", ".join(f"{LABELS[k]} {v}" for k, v in m["products_won_h1"].items()),
        "COV_GBM_1": pct(iv["1"]["gbm"]["coverage"], 0), "COV_ETS_1": pct(iv["1"]["ets"]["coverage"], 0),
        "COV_THETA_4": pct(iv["4"]["theta"]["coverage"], 0), "COV_GBM_4": pct(iv["4"]["gbm"]["coverage"], 0),
        "FILL_BEST": pct(inv_best["fill_rate"]), "FILL_NAIVE": pct(inv_naive["fill_rate"]),
        "LEFT_BEST": units(inv_best["leftover_per_week"]), "LEFT_NAIVE": units(inv_naive["leftover_per_week"]),
        "SL": pct(sl, 0),
        "TIER_HIGH": f"{tiers.loc['high', best]:.3f}", "TIER_LOW": f"{tiers.loc['low', best]:.3f}",
        "PEAK": f"{seg.loc['peak', best]:.3f}", "PREPEAK": f"{seg.loc['pre-peak', best]:.3f}",
        "PEAK_NAIVE": f"{seg.loc['peak', 'naive']:.3f}", "PREPEAK_NAIVE": f"{seg.loc['pre-peak', 'naive']:.3f}",
        "SENS_LINE": ("ranks first in every window" if sens_best else "does not rank first in every window"),
        "H_START": st["headline_start"], "H_END": st["headline_end"], "C_START": st["calibration_start"],
        "HEADLINE_WEEKS": str(st["headline_weeks"]), "CALIB_WEEKS": str(st["calibration_weeks"]),
        "PEAK_START": st["peak_start"], "N_BOOT": f"{st['n_boot']:,}", "FITS": str(st["fits_per_series"]),
        "LR": str(m["tuning"]["chosen"]["learning_rate"]), "DEPTH": str(m["tuning"]["chosen"]["max_depth"]),
        "RAW_ROWS": units(s["raw_rows"]), "CLEAN_ROWS": units(s["clean_rows"]), "DATE_MIN": s["date_min"],
        "DATE_MAX": s["date_max"], "SKUS": units(s["skus_sold"]), "WEEKS": str(s["weeks"]),
        "FIRST_WEEK": s["first_week"], "LAST_WEEK": s["last_week"], "PRODUCTS": str(s["products"]),
        "REV_SHARE": pct(s["product_revenue_share"]), "ZERO_SHARE": pct(s["zero_weeks_share"]),
    }


def fill(template: str, vals: dict[str, str], tabs: dict[str, tuple], table_fn: Any) -> str:
    """Replace {{KEY}} placeholders with values and rendered tables; unknown keys raise."""
    allv = {**vals, **{k: table_fn(h, r) for k, (h, r) in tabs.items()}}
    return re.sub(r"\{\{(\w+)\}\}", lambda mt: allv[mt.group(1)], template)


def _img(path: Path, alt: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{data}"></figure>'


FIGURES = [
    ("backtest_total", "All-products weekly units against one-week-ahead forecasts and the boosting 80% band"),
    ("wape_by_model", "Pooled product WAPE with 95% bootstrap intervals, 1 and 4 weeks ahead"),
    ("interval_coverage", "Coverage of each 80% prediction interval"),
    ("segment_errors", "WAPE by volume tier and by pre-peak versus peak weeks"),
    ("inventory_tradeoff", "Fill rate against leftover stock for each forecast and service target"),
]


def render_html(v: dict[str, str], t: dict[str, str], results: Path) -> str:
    """One self-contained page: TL;DR, figures, tables, method, limitations."""
    f = {name: _img(results / f"{name}.png", alt) for name, alt in FIGURES}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Demand Forecasting</title>
<style>
:root {{ --fg:#1a202c; --muted:#4a5568; --bg:#ffffff; --line:#e2e8f0; --accent:#2b6cb0; --card:#f7fafc; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e2e8f0; --muted:#a0aec0; --bg:#171923;
  --line:#2d3748; --accent:#63b3ed; --card:#1a202c; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.6 system-ui,-apple-system,sans-serif;
  max-width:960px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:40px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
.sub {{ color:var(--muted); margin-top:0; }}
.tldr {{ background:var(--card); border-left:4px solid var(--accent); padding:12px 16px; border-radius:4px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:6px; padding:12px; }}
.kpi b {{ display:block; font-size:24px; color:var(--accent); }}
.kpi span {{ color:var(--muted); font-size:14px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:12px 0; display:block; overflow-x:auto; }}
th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; white-space:nowrap; }}
figure {{ margin:16px 0; }} img {{ max-width:100%; height:auto; background:#fff; border-radius:4px; }}
</style></head><body>
<h1>Weekly Demand Forecasting</h1>
<p class="sub">How many units of each top product will sell next week, how sure are we, and how much stock does that imply?</p>
<div class="tldr"><b>TL;DR.</b> Over a {v['HEADLINE_WEEKS']}-week rolling-origin backtest ({v['H_START']} to {v['H_END']},
the run-up to Christmas), {v['BEST']} forecasts the {v['PRODUCTS']} top products with a pooled WAPE of
<b>{v['BEST_WAPE']}</b> (95% CI {v['BEST_CI']}), {v['CUT']} below the naive last-week forecast ({v['NAIVE_WAPE']}).
For the all-products total, {v['BEST_TOTAL']} is best (WAPE {v['BEST_TOTAL_WAPE']}). Safety stock sized on
pre-peak errors under-delivers in the peak: a {v['SL']} target fills only {v['FILL_BEST']} of demand.</div>
<div class="kpis">
<div class="kpi"><b>{v['BEST_WAPE']}</b><span>product WAPE, {v['BEST']} (naive {v['NAIVE_WAPE']})</span></div>
<div class="kpi"><b>{v['BEST_TOTAL_WAPE']}</b><span>total WAPE, {v['BEST_TOTAL']}</span></div>
<div class="kpi"><b>{v['COV_ETS_1']}</b><span>coverage of the ETS 80% interval</span></div>
<div class="kpi"><b>{v['FILL_BEST']}</b><span>fill rate at a {v['SL']} target</span></div>
</div>
<h2>Accuracy, one week ahead</h2>
{t['TABLE_H1']}
{f['wape_by_model']}
{f['backtest_total']}
<h2>Accuracy, four weeks ahead</h2>
{t['TABLE_H4']}
<h2>Prediction intervals</h2>
{t['INTERVAL_TABLE']}
{f['interval_coverage']}
<h2>Where it is wrong</h2>
{t['SEGMENT_TABLE']}
{f['segment_errors']}
<h3>Per product ({v['BEST']}, worst first)</h3>
{t['PRODUCT_TABLE']}
<h2>Inventory view</h2>
<p>Weekly review, one-week lead time. Stock up to forecast + z &times; (that product's RMSE on the
{v['CALIB_WEEKS']} calibration weeks from {v['C_START']}); fill rate is units served from stock over units demanded.</p>
{t['INVENTORY_TABLE']}
{f['inventory_tradeoff']}
<h2>Sensitivity to the backtest window</h2>
{t['SENSITIVITY_TABLE']}
<h2>Data</h2>
<p>{v['RAW_ROWS']} invoice lines from a UK online gift and homeware wholesaler, {v['DATE_MIN']} to {v['DATE_MAX']}
(UCI Online Retail II, CC BY 4.0). After cleaning, {v['CLEAN_ROWS']} lines and {v['SKUS']} products; the
{v['PRODUCTS']} forecast products carry {v['REV_SHARE']} of revenue. {v['WEEKS']} full weeks, {v['FIRST_WEEK']} to {v['LAST_WEEK']}.</p>
{t['CLEANING_TABLE']}
<h2>Method</h2>
<p>Every forecast for week t is fitted only on weeks up to t &minus; h ({v['FITS']} refits per series). Methods:
naive, seasonal naive, damped Holt (ETS), Theta, and one global histogram gradient-boosting model on scale-free
lags, rolling statistics and UK holiday features (direct model per horizon; learning rate {v['LR']}, depth
{v['DEPTH']}, chosen on the 8 weeks before the backtest). Its 80% interval comes from 10th and 90th percentile
quantile models. CIs: {v['N_BOOT']} bootstrap draws over products (pooled metrics) or weeks (per product).</p>
<h2>Limitations</h2>
<ul>
<li>One retailer, 104 weeks: too short for a fitted yearly seasonal term, and the backtest covers one peak season.</li>
<li>No price, promotion or stock-out data: zero-sales weeks may be stock-outs, not zero demand.</li>
<li>Single trade orders cause spikes that no history-only model anticipates; order books would help.</li>
<li>The inventory view ignores lead-time variability, order costs and pack sizes.</li>
</ul>
</body></html>
"""


def render_all(cfg: dict[str, Any]) -> None:
    """Write README.md and docs/index.html."""
    results = resolve(cfg["paths"]["results"])
    m = json.loads((results / "metrics.json").read_text())
    s = json.loads((results / "data_summary.json").read_text())
    per = pd.read_csv(results / "product_errors.csv", dtype={"sku": str})
    v, tabs = values(m, s), tables(m, per, s)
    template = (ROOT / "README.template.md").read_text()
    (ROOT / "README.md").write_text(fill(template, v, tabs, md_table))
    docs = resolve(cfg["paths"]["docs"])
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(render_html(v, {k: html_table(h, r) for k, (h, r) in tabs.items()}, results))
    print("rendered README.md and docs/index.html")
