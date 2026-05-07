"""FHFA House Price Index data loader (free bulk CSV/XLSX download)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from housing_estimator.config import settings

FHFA_FILE = "fhfa_hpi.csv"

_cache: pd.DataFrame | None = None


def _fhfa_path() -> Path:
    return settings.data.raw_path / FHFA_FILE


def load_fhfa_data() -> pd.DataFrame:
    """Load FHFA HPI data from downloaded CSV.

    Supports both 5-digit ZIP (annual) and 3-digit ZIP (quarterly) formats.
    Returns DataFrame with columns: zip_code, year, hpi
    """
    global _cache
    if _cache is not None:
        return _cache

    path = _fhfa_path()
    if not path.exists():
        return pd.DataFrame(columns=["zip_code", "year", "hpi"])

    df = pd.read_csv(path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if "zip" in lower:
            col_map[col] = "zip_code"
        elif lower == "year" or lower == "yr":
            col_map[col] = "year"
        elif lower == "hpi" or (lower.startswith("hpi") and "base" not in lower and "change" not in lower):
            col_map[col] = "hpi"
        elif "index" in lower and "hpi" not in col_map.values():
            col_map[col] = "hpi"

    df = df.rename(columns=col_map)

    for required in ["zip_code", "year", "hpi"]:
        if required not in df.columns:
            return pd.DataFrame(columns=["zip_code", "year", "hpi"])

    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["hpi"] = pd.to_numeric(df["hpi"], errors="coerce")
    df = df.dropna(subset=["zip_code", "year", "hpi"])

    _cache = df[["zip_code", "year", "hpi"]].copy()
    return _cache


def get_hpi_for_zip(zip_code: str) -> dict[str, float | None]:
    """Get HPI data for a ZIP code.

    Returns dict with keys: hpi_current, hpi_1yr_change
    Tries exact 5-digit match first, then falls back to 3-digit prefix match.
    """
    df = load_fhfa_data()
    if df.empty:
        return {"hpi_current": None, "hpi_1yr_change": None}

    zip5 = zip_code.zfill(5)
    subset = df[df["zip_code"] == zip5].sort_values("year")

    # Fall back to 3-digit prefix if no exact match
    if subset.empty:
        zip3 = zip5[:3]
        subset = df[df["zip_code"].str[:3] == zip3].sort_values("year")

    if subset.empty:
        return {"hpi_current": None, "hpi_1yr_change": None}

    latest = subset.iloc[-1]
    hpi_current = float(latest["hpi"])

    # 1-year change (annual data: previous row is 1 year back)
    hpi_1yr_change = None
    if len(subset) >= 2:
        prev_hpi = float(subset.iloc[-2]["hpi"])
        if prev_hpi > 0:
            hpi_1yr_change = (hpi_current - prev_hpi) / prev_hpi

    return {"hpi_current": hpi_current, "hpi_1yr_change": hpi_1yr_change}


def get_hpi_at_year(zip_code: str, year: int) -> float | None:
    """Get the HPI value for a ZIP at a specific year (for time-adjusting prices)."""
    df = load_fhfa_data()
    if df.empty:
        return None

    zip5 = zip_code.zfill(5)
    subset = df[(df["zip_code"] == zip5) & (df["year"] == year)]

    # Fall back to 3-digit prefix
    if subset.empty:
        zip3 = zip5[:3]
        subset = df[(df["zip_code"].str[:3] == zip3) & (df["year"] == year)]

    if subset.empty:
        return None

    return float(subset["hpi"].mean())
