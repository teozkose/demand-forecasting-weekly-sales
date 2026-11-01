# Weekly Demand Forecasting

One-week-ahead unit forecasts for a retailer's top products, with simple baselines kept in the comparison so the fancier model has to earn its place.

**Data:** two years of invoice lines from a UK online gift retailer (UCI Online Retail II), aggregated to {{N_WEEKS}} full weeks of units for the {{N_PRODUCTS}} highest-revenue products that sold in at least 90 weeks, plus the total across all products. Zero-sales weeks are kept as zeros.

**Setup:** the last {{HOLDOUT}} weeks ({{H_START}} to {{H_END}}, which include the pre-Christmas ramp) are the holdout. Every forecast for a week uses only data up to the week before. The statistical models are refit at each step; the boosting model is trained once on the earlier weeks and fed actual lags. Error is reported as WAPE (sum of absolute errors over sum of actuals), which stays defined for products with zero weeks, plus MAE in units.

## Results

| Method | Products WAPE (pooled) | Products WAPE (median) | Products MAE | Total WAPE | Total MAE |
|---|---|---|---|---|---|
{{TABLE}}

Products won (lowest WAPE per product): {{WINS}}.

{{FINDINGS}}

![Forecasts](results/forecasts.png)

Takeaway: {{TAKEAWAY}}

Method notes: the boosting model uses lags {{LAGS}} and a 4-week rolling mean, all divided by the trailing 12-week mean, plus week of year; its target is units relative to that same mean, so one model serves products of any size and the total. Holt's smoothing uses a damped additive trend and no seasonal term, because a yearly season needs two full years of history and the training window is shorter.

## How it works

```
data/download.py     fetch the workbook, cache as parquet (not committed)
src/prepare.py       clean, pick products, build complete weekly series; results/products.csv
src/forecast.py      rolling one-step-ahead forecasts for 4 methods; results/forecasts.csv, errors_by_product.csv, metrics.json
src/report.py        render this README from results/metrics.json
notebooks/01_analysis.ipynb   the same steps with the series and errors shown
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

About two minutes, plus the one-time workbook download. Every number above is read from `results/metrics.json`, which the run regenerates.

Data: [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI Machine Learning Repository (CC BY 4.0).
