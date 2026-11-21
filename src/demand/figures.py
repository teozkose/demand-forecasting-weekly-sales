"""Figures for the README and the report page. Every function writes one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"naive": "Naive", "seasonal_naive": "Seasonal naive", "ets": "Damped Holt (ETS)", "theta": "Theta",
          "gbm": "Global gradient boosting"}
COLORS = {"naive": "#8c8c8c", "seasonal_naive": "#b07aa1", "ets": "#4e79a7", "theta": "#59a14f", "gbm": "#e15759"}


def total_backtest(fc: pd.DataFrame, path: Path, horizon: int = 1) -> Path:
    """All-products series: actuals against each model's forecasts, with the boosting 80% band."""
    d = fc[fc["is_total"] & (fc["horizon"] == horizon)].sort_values("week")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.fill_between(d["week"], d["gbm_lo"], d["gbm_hi"], color=COLORS["gbm"], alpha=0.15, label="boosting 80% interval")
    ax.plot(d["week"], d["actual"], color="black", marker="o", lw=2, label="actual")
    for m in ["naive", "ets", "theta", "gbm"]:
        ax.plot(d["week"], d[m], color=COLORS[m], lw=1.4, label=LABELS[m])
    ax.set_title(f"All products, {horizon}-week-ahead forecasts over the backtest weeks")
    ax.set_ylabel("units per week")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def wape_with_ci(summary: dict, path: Path) -> Path:
    """Pooled product WAPE with 95% bootstrap CIs, one panel per horizon."""
    horizons = sorted(summary, key=int)
    fig, axes = plt.subplots(1, len(horizons), figsize=(5.2 * len(horizons), 3.8), sharey=True)
    axes = axes if len(horizons) > 1 else [axes]
    for ax, h in zip(axes, horizons, strict=True):
        s = summary[h]
        models = list(s)
        vals = [s[m]["wape"] for m in models]
        err = [[s[m]["wape"] - s[m]["wape_ci"][0] for m in models], [s[m]["wape_ci"][1] - s[m]["wape"] for m in models]]
        ax.barh([LABELS[m] for m in models], vals, xerr=err, color=[COLORS[m] for m in models], capsize=3)
        ax.set_title(f"{h} week(s) ahead")
        ax.set_xlabel("pooled WAPE across products (lower is better)")
        ax.invert_yaxis()
    fig.suptitle("Forecast error with 95% bootstrap intervals (resampling products)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def coverage_plot(intervals: dict, nominal: float, path: Path) -> Path:
    """Empirical coverage of each 80% interval against the nominal level, by horizon."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    horizons = sorted(intervals, key=int)
    width = 0.8 / len(horizons)
    models = list(intervals[horizons[0]])
    for j, h in enumerate(horizons):
        cov = [intervals[h][m]["coverage"] for m in models]
        lo = [c - intervals[h][m]["coverage_ci"][0] for c, m in zip(cov, models, strict=True)]
        hi = [intervals[h][m]["coverage_ci"][1] - c for c, m in zip(cov, models, strict=True)]
        xs = [i + j * width for i in range(len(models))]
        ax.bar(xs, cov, width=width, yerr=[lo, hi], capsize=3, label=f"{h} week(s) ahead", alpha=0.85)
    ax.axhline(nominal, color="black", ls="--", lw=1, label=f"nominal {nominal:.0%}")
    ax.set_xticks([i + width * (len(horizons) - 1) / 2 for i in range(len(models))], [LABELS[m] for m in models])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of actuals inside the interval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def inventory_plot(inv: pd.DataFrame, path: Path) -> Path:
    """Fill rate against average leftover stock for each model and service-level target."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, g in inv.groupby("model", sort=False):
        key = m if m in COLORS else "gbm"
        style = "o-" if m in COLORS else "*"
        ax.plot(g["leftover_per_week"], g["fill_rate"], style, color=COLORS[key], ms=9 if m not in COLORS else 6,
                label=LABELS.get(m, "Boosting 90th percentile"))
    ax.set_xlabel("leftover stock per week, all products (units)")
    ax.set_ylabel("fill rate (units served / demanded)")
    ax.set_title("Service bought per unit of leftover stock (90 / 95 / 98% targets)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def segment_plot(seg: pd.DataFrame, models: list[str], path: Path) -> Path:
    """WAPE by volume tier and by pre-peak versus pre-Christmas peak weeks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, kind in zip(axes, ["tier", "period"], strict=True):
        d = seg[seg["segment_type"] == kind].set_index("segment")[models]
        d.columns = [LABELS[m] for m in models]
        d.plot.bar(ax=ax, color=[COLORS[m] for m in models], rot=0, legend=kind == "tier")
        ax.set_title("by volume tier" if kind == "tier" else "pre-peak vs pre-Christmas peak")
        ax.set_ylabel("pooled WAPE")
        ax.set_xlabel("")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
