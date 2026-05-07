#!/usr/bin/env python3
"""Run a non-private local-estimate example.

This script uses the public MIT main address as a safe demonstration address.
The property features below are illustrative inputs for exercising the model
workflow; they are not meant to describe MIT's campus or produce an appraisal.
For a real residential estimate, replace ADDRESS and PropertyFeatures with
verified subject-property facts.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console

from housing_estimator.estimator import geocode_address
from housing_estimator.features.property import PropertyFeatures, PropertyType
from housing_estimator.models.local_model import fetch_training_sales, train_and_predict
from housing_estimator.output import render_local_estimate

console = Console()

ADDRESS = "77 Massachusetts Ave, Cambridge, MA 02139"


def main() -> None:
    console.print(f"\n[bold]Local Market Estimate Example for:[/bold] {ADDRESS}\n")
    console.print("[bold]Step 1: Geocoding address...[/bold]")
    geo = asyncio.run(geocode_address(ADDRESS))
    if not geo:
        console.print("[red]Could not geocode address.[/red]")
        return

    console.print(f"  Matched: {geo.matched_address}")
    console.print(f"  Lat/Lon: {geo.lat:.6f}, {geo.lon:.6f}\n")

    console.print("[bold]Step 2: Example property details[/bold]")
    features = PropertyFeatures(
        bedrooms=3,
        bathrooms=2.0,
        sqft=1500,
        lot_sqft=5000,
        year_built=1990,
        property_type=PropertyType.SINGLE_FAMILY,
        latitude=geo.lat,
        longitude=geo.lon,
        zip_code=geo.zip_code,
    )

    console.print(f"  Bedrooms:   {features.bedrooms}")
    console.print(f"  Bathrooms:  {features.bathrooms}")
    console.print(f"  Sqft:       {features.sqft:,.0f}")
    console.print(f"  Lot Sqft:   {features.lot_sqft:,.0f}")
    console.print(f"  Year Built: {features.year_built}")
    console.print("  Type:       Single Family Residential")
    console.print()

    console.print("[bold]Step 3: Fetching nearby sales from Redfin...[/bold]")
    sales, search_radius, search_days = fetch_training_sales(
        lat=geo.lat,
        lon=geo.lon,
        target_count=200,
    )

    if len(sales) < 20:
        console.print(f"[red]Only {len(sales)} usable sales found; need at least 20.[/red]")
        return

    console.print("\n[bold]Step 4: Training local XGBoost model...[/bold]")
    estimate = train_and_predict(features, sales, search_radius, search_days)
    if estimate is None:
        console.print("[red]Could not generate estimate.[/red]")
        return

    closest = sorted(
        [s for s in sales if s.sqft and s.sqft > 0],
        key=lambda s: s.distance_miles or 999,
    )[:10]

    render_local_estimate(
        address=geo.matched_address,
        features=features,
        estimate=estimate,
        closest_sales=closest,
    )


if __name__ == "__main__":
    main()
