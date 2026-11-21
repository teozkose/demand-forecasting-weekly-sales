"""Fetch the UCI Online Retail II workbook (CC BY 4.0) and cache it as data/transactions.parquet.

Same as `make data`; kept so the data step can be run on its own: python data/download.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand.config import load_config  # noqa: E402
from demand.pipeline import step_data  # noqa: E402

if __name__ == "__main__":
    step_data(load_config())
