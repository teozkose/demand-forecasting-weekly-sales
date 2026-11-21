"""Download the UCI Online Retail II workbook, clean the invoice lines and build complete weekly series.

The source is a 45 MB Excel workbook with two yearly sheets (Dec 2009 to Dec 2011, about 1.07 million
invoice lines from a UK online gift and homeware wholesaler). Reading it takes about a minute, so the
lines are cached as parquet and every later step reads the cache.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
SKU_PATTERN = r"^\d{5}[A-Za-z]?$"  # five digits plus an optional variant letter; drops postage, fees, manual lines


def download(url: str, dest: Path, min_bytes: int = 1_000_000) -> Path:
    """Fetch the zip, read both workbook sheets and cache them as parquet. Skips when the cache exists."""
    if dest.exists() and dest.stat().st_size > min_bytes:
        print(f"already present: {dest}")
        return dest
    print(f"downloading {url}")
    blob = urllib.request.urlopen(url, timeout=180).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        xlsx = z.read(name)
    print("reading workbook (about a minute)")
    book = pd.ExcelFile(io.BytesIO(xlsx))
    df = pd.concat([pd.read_excel(book, sheet) for sheet in book.sheet_names], ignore_index=True)
    for c in ["Invoice", "StockCode", "Description", "Country"]:
        df[c] = df[c].astype(str)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    print(f"saved {dest}: {len(df):,} rows")
    return dest


def load_raw(path: Path) -> pd.DataFrame:
    """Read the cached invoice lines."""
    return pd.read_parquet(path)


def clean(raw: pd.DataFrame, country: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Keep real product sales in one country and record how many lines each step removes.

    Returns the cleaned lines (sku, description, units, price, revenue, ts, week) and a list of
    {"step", "removed", "remaining"} dicts for the README's cleaning table.
    """
    steps: list[dict[str, Any]] = []
    df = raw

    def record(step: str, new: pd.DataFrame) -> pd.DataFrame:
        steps.append({"step": step, "removed": int(len(df) - len(new)), "remaining": int(len(new))})
        return new

    df = record("exact duplicate invoice lines", df.drop_duplicates())
    df = record("cancellation invoices (number starts with C)", df[~df["Invoice"].str.startswith("C")])
    df = record("zero or negative quantity or price", df[(df["Quantity"] > 0) & (df["Price"] > 0)])
    df = record(f"customers outside {country}", df[df["Country"] == country])
    df = record("non-product codes (postage, fees, manual, samples)", df[df["StockCode"].str.match(SKU_PATTERN)])
    out = pd.DataFrame({
        "sku": df["StockCode"].str.upper(),
        "description": df["Description"].str.strip(),
        "units": df["Quantity"].astype("int64"),
        "price": df["Price"].astype(float),
        "ts": pd.to_datetime(df["InvoiceDate"]),
    })
    out["revenue"] = out["units"] * out["price"]
    out["week"] = out["ts"].dt.to_period("W-SUN").dt.start_time
    return out.reset_index(drop=True), steps


def full_weeks(lines: pd.DataFrame) -> pd.DatetimeIndex:
    """Every Monday-start week between the first and last sale, minus the partial first and last week."""
    weeks = pd.date_range(lines["week"].min(), lines["week"].max(), freq="7D")
    return weeks[1:-1]


def complete(series: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex each sku onto the full week grid, filling weeks without sales with zero units."""
    out = []
    for sku, g in series.groupby("sku", sort=True):
        s = g.set_index("week")["units"].reindex(weeks, fill_value=0)
        out.append(pd.DataFrame({"sku": sku, "week": weeks, "units": s.to_numpy(dtype=float)}))
    return pd.concat(out, ignore_index=True)


def build_panel(lines: pd.DataFrame, top_n: int, min_weeks: int, total_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly units for the top_n revenue products that sold in at least min_weeks weeks, plus a total series.

    Returns (panel, products): panel is long (sku, week, units); products has one row per kept sku with
    its description, revenue, weeks sold and mean weekly units.
    """
    weeks = full_weeks(lines)
    lines = lines[lines["week"].isin(weeks)]
    stats = lines.groupby("sku").agg(weeks_sold=("week", "nunique"), revenue=("revenue", "sum"), units=("units", "sum"))
    keep = stats[stats["weeks_sold"] >= min_weeks].sort_values("revenue", ascending=False).head(top_n).copy()
    names = lines[lines["sku"].isin(keep.index)].groupby("sku")["description"].agg(lambda s: s.mode().iloc[0])
    keep["description"] = names.reindex(keep.index)
    keep["mean_weekly_units"] = keep["units"] / len(weeks)
    weekly = lines[lines["sku"].isin(keep.index)].groupby(["sku", "week"], as_index=False)["units"].sum()
    panel = complete(weekly, weeks)
    total = lines.groupby("week")["units"].sum().reindex(weeks, fill_value=0)
    panel = pd.concat(
        [panel, pd.DataFrame({"sku": total_name, "week": weeks, "units": total.to_numpy(dtype=float)})],
        ignore_index=True,
    )
    return panel, keep.reset_index()


def data_summary(raw: pd.DataFrame, lines: pd.DataFrame, panel: pd.DataFrame, products: pd.DataFrame,
                 total_name: str) -> dict[str, Any]:
    """Facts for the README's Data section."""
    prod = panel[panel["sku"] != total_name]
    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(lines)),
        "date_min": str(pd.Timestamp(raw["InvoiceDate"].min()).date()),
        "date_max": str(pd.Timestamp(raw["InvoiceDate"].max()).date()),
        "skus_sold": int(lines["sku"].nunique()),
        "weeks": int(panel["week"].nunique()),
        "first_week": str(pd.Timestamp(panel["week"].min()).date()),
        "last_week": str(pd.Timestamp(panel["week"].max()).date()),
        "products": int(products.shape[0]),
        "product_revenue_share": float(products["revenue"].sum() / lines["revenue"].sum()),
        "zero_weeks_share": float((prod["units"] == 0).mean()),
    }
