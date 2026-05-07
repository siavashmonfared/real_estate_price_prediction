"""Census ACS median household income data (from downloaded or API data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_estimator.config import settings

CENSUS_ACS_FILE = "census_acs_income.csv"


def _acs_path() -> Path:
    return settings.data.raw_path / CENSUS_ACS_FILE


_cache: pd.DataFrame | None = None


def load_acs_data() -> pd.DataFrame:
    """Load Census ACS median income data.

    Expected columns: zip_code, median_income
    """
    global _cache
    if _cache is not None:
        return _cache

    path = _acs_path()
    if not path.exists():
        return pd.DataFrame(columns=["zip_code", "median_income"])

    df = pd.read_csv(path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "zip" in lower or "zcta" in lower or "geo" in lower:
            col_map[col] = "zip_code"
        elif "income" in lower or "b19013" in lower:
            col_map[col] = "median_income"

    df = df.rename(columns=col_map)

    if "zip_code" not in df.columns:
        return pd.DataFrame(columns=["zip_code", "median_income"])

    df["zip_code"] = df["zip_code"].astype(str).str.extract(r"(\d{5})")[0]
    df = df.dropna(subset=["zip_code"])

    if "median_income" in df.columns:
        df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")

    _cache = df
    return _cache


def get_median_income(zip_code: str) -> float | None:
    """Get median household income for a ZIP code."""
    df = load_acs_data()
    if df.empty or "median_income" not in df.columns:
        return None

    match = df[df["zip_code"] == zip_code]
    if match.empty:
        return None

    val = match.iloc[-1]["median_income"]
    return float(val) if pd.notna(val) else None
