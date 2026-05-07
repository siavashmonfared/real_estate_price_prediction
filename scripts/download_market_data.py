"""Download market data: FHFA HPI, Redfin bulk CSVs, Census ACS income."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from rich.console import Console

from housing_estimator.config import settings

console = Console()

# FHFA HPI data (3-digit ZIP level, quarterly)
# FHFA periodically moves files; try multiple known paths
FHFA_URLS = [
    "https://www.fhfa.gov/sites/default/files/2025-11/HPI_AT_3zip.csv",
    "https://www.fhfa.gov/sites/default/files/2025-02/HPI_AT_3zip.csv",
    "https://www.fhfa.gov/hpi/download/monthly/hpi_at_3zip.csv",
]

# Redfin Data Center — ZIP-level market data
# Note: Redfin periodically changes URLs. This is the bulk download pattern.
REDFIN_URL = (
    "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/"
    "zip_code_market_tracker.tsv000.gz"
)


def download_file(url: str, dest: Path, description: str) -> bool:
    console.print(f"  Downloading {description}...")
    console.print(f"    URL: {url}")
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            size_mb = len(resp.content) / 1_048_576
            console.print(f"    Saved: {dest} ({size_mb:.1f} MB)")
            return True
    except httpx.HTTPError as e:
        console.print(f"    [yellow]Failed: {e}[/yellow]")
        return False


def download_fhfa():
    dest = settings.data.raw_path / "fhfa_hpi.csv"
    if dest.exists():
        console.print("  [yellow]FHFA HPI data already exists, skipping.[/yellow]")
        return True
    for url in FHFA_URLS:
        ok = download_file(url, dest, "FHFA House Price Index (3-digit ZIP)")
        if ok:
            return True
    return False


def download_redfin():
    dest = settings.data.raw_path / "redfin_zip_market_data.csv"
    if dest.exists():
        console.print("  [yellow]Redfin data already exists, skipping.[/yellow]")
        return True

    # Try downloading — this is a large file and may not always be available
    dest_gz = settings.data.raw_path / "redfin_zip_raw.tsv.gz"
    ok = download_file(REDFIN_URL, dest_gz, "Redfin ZIP-level market data")
    if ok:
        try:
            import pandas as pd

            console.print("    Processing Redfin data...")
            df = pd.read_csv(dest_gz, sep="\t", compression="gzip")
            console.print(f"    Raw columns: {df.columns.tolist()[:15]}...")

            # Redfin TSV has columns like: period_begin, period_end, period_duration,
            # region_type, region_type_id, table_id, is_seasonally_adjusted, region,
            # city, state, state_code, property_type, property_type_id,
            # median_sale_price, median_sale_price_mom, median_sale_price_yoy,
            # median_ppsf (price per sqft), homes_sold, ...

            # Filter to ZIP code rows only
            if "region_type" in df.columns:
                df = df[df["region_type"] == "zip_code"]
                console.print(f"    Filtered to ZIP rows: {len(df)}")

            # Keep latest period, non-seasonally-adjusted
            if "is_seasonally_adjusted" in df.columns:
                df = df[df["is_seasonally_adjusted"] == False]
            if "period_end" in df.columns:
                df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
                latest = df["period_end"].max()
                df = df[df["period_end"] == latest]
                console.print(f"    Latest period: {latest}, rows: {len(df)}")

            # All property types combined (if available)
            if "property_type" in df.columns:
                all_types = df[df["property_type"] == "All Residential"]
                if len(all_types) > 0:
                    df = all_types

            # Extract the right columns: 'region' is the ZIP code value
            out_rows = []
            for _, row in df.iterrows():
                zip_code = str(row.get("region", "")).strip()
                if not zip_code.isdigit() or len(zip_code) != 5:
                    continue

                median_price = row.get("median_sale_price")
                median_ppsf = row.get("median_ppsf")
                if pd.isna(median_price) and pd.isna(median_ppsf):
                    continue

                out_rows.append({
                    "zip_code": zip_code,
                    "median_sale_price": median_price if pd.notna(median_price) else None,
                    "median_price_per_sqft": median_ppsf if pd.notna(median_ppsf) else None,
                })

            if out_rows:
                out = pd.DataFrame(out_rows)
                out.to_csv(dest, index=False)
                console.print(f"    Processed {len(out)} ZIP codes → {dest}")
                dest_gz.unlink(missing_ok=True)
                return True
            else:
                console.print("    [yellow]No valid ZIP rows found in Redfin data[/yellow]")
        except Exception as e:
            console.print(f"    [yellow]Processing failed: {e}[/yellow]")

    console.print("    [yellow]Redfin data unavailable (non-critical).[/yellow]")
    return False


def main():
    raw_dir = settings.data.raw_path
    raw_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Downloading market data...[/bold]")
    download_fhfa()
    download_redfin()

    console.print("\n[bold]Market data download complete.[/bold]")
    console.print("[dim]Note: Some market data sources may be unavailable. "
                  "The model will work with whatever data is available.[/dim]")


if __name__ == "__main__":
    main()
