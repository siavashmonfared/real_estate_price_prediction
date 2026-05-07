"""Redfin Data Center bulk CSV loader (free, no API key)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_estimator.config import settings

REDFIN_FILE = "redfin_zip_market_data.csv"


def _redfin_path() -> Path:
    return settings.data.raw_path / REDFIN_FILE


_cache: pd.DataFrame | None = None


def load_redfin_data() -> pd.DataFrame:
    """Load Redfin market data from downloaded CSV.

    Expected columns after processing: zip_code, median_price_per_sqft,
    median_sale_price, months_of_supply
    """
    global _cache
    if _cache is not None:
        return _cache

    path = _redfin_path()
    if not path.exists():
        return pd.DataFrame(columns=["zip_code", "median_price_per_sqft"])

    df = pd.read_csv(path)

    # Redfin CSVs vary in column names; normalize
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "zip" in lower or "region" in lower:
            col_map[col] = "zip_code"
        elif "median" in lower and "price" in lower and "sqft" in lower:
            col_map[col] = "median_price_per_sqft"
        elif "median" in lower and "sale" in lower and "price" in lower:
            col_map[col] = "median_sale_price"

    df = df.rename(columns=col_map)

    if "zip_code" not in df.columns:
        return pd.DataFrame(columns=["zip_code", "median_price_per_sqft"])

    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)

    _cache = df
    return _cache


def get_median_price_per_sqft(zip_code: str) -> float | None:
    """Get the median price per sqft for a ZIP code from Redfin data."""
    df = load_redfin_data()
    if df.empty or "median_price_per_sqft" not in df.columns:
        return None

    match = df[df["zip_code"] == zip_code]
    if match.empty:
        return None

    val = match.iloc[-1]["median_price_per_sqft"]
    return float(val) if pd.notna(val) else None
