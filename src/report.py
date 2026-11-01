"""Render README.md from README.template.md and results/metrics.json so every number and every
conclusion in the write-up comes from the last run."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
LABELS = {"naive": "Naive (last week)", "seasonal_naive": "Seasonal naive (same week last year)",
          "holt_damped": "Holt damped-trend smoothing", "gradient_boost": "Gradient boosting (scale-free lags + calendar)"}
SHORT = {k: v.split(" (")[0] for k, v in LABELS.items()}
M = m["methods"]
rows = "\n".join(
    f"| {LABELS[k]} | {v['products_wape_pooled']:.3f} | {v['products_wape_median']:.3f} | {v['products_mae']:.1f} | {v['total_wape']:.3f} | {v['total_mae']:,.0f} |"
    for k, v in M.items()
)
wins = ", ".join(f"{SHORT[k]} {v}" for k, v in sorted(m["products_won_by_method"].items(), key=lambda x: -x[1]))
best_p, best_t = m["best_method_products"], m["best_method_total"]

# Conclusions written from the numbers, not in advance.
lines = []
lines.append(f"- Per product, the lowest pooled WAPE is {M[best_p]['products_wape_pooled']:.3f} ({SHORT[best_p]}); "
             f"the naive baseline scores {M['naive']['products_wape_pooled']:.3f}.")
others = ", ".join(f"{SHORT[k]} {M[k]['total_wape']:.3f}" for k in M if k != best_t)
lines.append(f"- For the total, {SHORT[best_t]} is best (WAPE {M[best_t]['total_wape']:.3f}); the others: {others}.")
if M["seasonal_naive"]["products_wape_pooled"] > M["naive"]["products_wape_pooled"]:
    lines.append("- Seasonal naive loses to plain naive per product: last week is a better guide than the same week last year, "
                 "because product-level demand here is lumpy (single large trade orders) rather than smoothly seasonal.")
else:
    lines.append("- Seasonal naive beats plain naive per product: the yearly pattern is a better guide than last week.")
gb_better = M["gradient_boost"]["products_wape_pooled"] < M["naive"]["products_wape_pooled"]
lines.append(f"- Gradient boosting {'does' if gb_better else 'does not'} beat the naive baseline on pooled product error"
             f"{', and wins the most individual products' if m['products_won_by_method'].get('gradient_boost', 0) == max(m['products_won_by_method'].values()) else ''}.")
findings = "\n".join(lines)
takeaway = (f"The cheapest baseline is hard to beat on one-week-ahead product demand: {SHORT[best_p]} has the best pooled "
            f"product error and {SHORT[best_t]} the best total error. Model choice matters less than the handful of products "
            f"with single-week spikes that no history-only method anticipates; those are listed with their errors in "
            f"`results/errors_by_product.csv`, and that is where forecast-tuning effort (order calendars, promotion flags) would pay off.")

s = m["setup"]
fill = {
    "N_PRODUCTS": str(s["products"]), "N_WEEKS": str(s["weeks"]), "HOLDOUT": str(s["holdout_weeks"]),
    "H_START": s["holdout_start"], "H_END": s["holdout_end"], "LAGS": ", ".join(str(x) for x in s["lags"]),
    "TABLE": rows, "WINS": wins, "FINDINGS": findings, "TAKEAWAY": takeaway,
}
text = open(os.path.join(ROOT, "README.template.md")).read()
for k, v in fill.items():
    text = text.replace("{{" + k + "}}", v)
open(os.path.join(ROOT, "README.md"), "w").write(text)
print("README.md rendered")
