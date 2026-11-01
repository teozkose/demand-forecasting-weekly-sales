"""Build complete weekly unit series for the products worth forecasting.

Output: data/weekly.parquet, long format (sku, week, units) with every week present (zeros filled),
for the TOP_N products by revenue that were sold in at least MIN_WEEKS distinct weeks, plus a
synthetic "TOTAL" series that sums all products in the clean data.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "transactions.parquet")
WEEKLY = os.path.join(ROOT, "data", "weekly.parquet")
PRODUCTS = os.path.join(ROOT, "results", "products.csv")

TOP_N = 20
MIN_WEEKS = 90


def load_clean(path: str = RAW) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[(df["Country"] == "United Kingdom") & (df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["Invoice"].str.startswith("C")]                      # cancellations
    df = df[df["StockCode"].str.match(r"^\d{5}[A-Za-z]?$")]          # drops postage, manual, fees
    df = df.rename(columns={"StockCode": "sku", "Quantity": "units", "Price": "price",
                            "InvoiceDate": "ts", "Description": "description"})
    df["week"] = df["ts"].dt.to_period("W").dt.start_time
    df["revenue"] = df["units"] * df["price"]
    return df


def complete_weeks(series: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    out = []
    for sku, g in series.groupby("sku"):
        s = g.set_index("week")["units"].reindex(weeks, fill_value=0)
        out.append(pd.DataFrame({"sku": sku, "week": weeks, "units": s.values}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = load_clean()
    # drop the partial first and last weeks so every week in the panel is a full week
    weeks = pd.date_range(df["week"].min(), df["week"].max(), freq="7D")[1:-1]
    df = df[df["week"].isin(weeks)]

    stats = df.groupby("sku").agg(weeks=("week", "nunique"), revenue=("revenue", "sum"))
    keep = stats[stats["weeks"] >= MIN_WEEKS].sort_values("revenue", ascending=False).head(TOP_N)
    names = df.groupby("sku")["description"].agg(lambda s: s.mode().iloc[0].strip())
    keep["description"] = names.reindex(keep.index)

    weekly_sku = df[df["sku"].isin(keep.index)].groupby(["sku", "week"])["units"].sum().reset_index()
    panel = complete_weeks(weekly_sku, weeks)
    total = df.groupby("week")["units"].sum().reindex(weeks, fill_value=0)
    panel = pd.concat([panel, pd.DataFrame({"sku": "TOTAL", "week": weeks, "units": total.values})], ignore_index=True)

    os.makedirs(os.path.dirname(PRODUCTS), exist_ok=True)
    panel.to_parquet(WEEKLY, index=False)
    keep.reset_index().to_csv(PRODUCTS, index=False)
    print(f"{len(weeks)} full weeks from {weeks[0].date()} to {weeks[-1].date()}; {len(keep)} products + TOTAL")
    print(f"saved {WEEKLY} and {PRODUCTS}")
