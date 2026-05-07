"""Main orchestrator — combines geocoding, ML prediction, and comp analysis."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from rich.console import Console

from housing_estimator.comps.engine import CompsResult, find_comps
from housing_estimator.config import settings
from housing_estimator.datasources.census_acs import get_median_income
from housing_estimator.datasources.fhfa import get_hpi_for_zip
from housing_estimator.datasources.redfin_bulk import get_median_price_per_sqft
from housing_estimator.features.property import PropertyFeatures
from housing_estimator.geocoder.census import GeocodingResult, geocode_census
from housing_estimator.geocoder.nominatim import geocode_nominatim
from housing_estimator.models.predict import PredictionResult, predict

console = Console()


@dataclass
class EstimateResult:
    address: str
    geocoding: GeocodingResult | None = None
    features: PropertyFeatures | None = None
    ml_prediction: PredictionResult | None = None
    comps_result: CompsResult | None = None
    blended_estimate: float | None = None
    blended_low: float | None = None
    blended_high: float | None = None
    comp_weight: float = 0.0
    ml_weight: float = 1.0
    errors: list[str] = field(default_factory=list)


async def geocode_address(address: str) -> GeocodingResult | None:
    """Geocode an address, trying Census first then Nominatim fallback."""
    try:
        result = await geocode_census(address)
        if result:
            return result
    except Exception as e:
        console.print(f"[yellow]Census geocoder failed: {e}[/yellow]")

    try:
        result = await geocode_nominatim(address)
        if result:
            return result
    except Exception as e:
        console.print(f"[yellow]Nominatim geocoder failed: {e}[/yellow]")

    return None


def enrich_features(features: PropertyFeatures) -> PropertyFeatures:
    """Add market context data to property features."""
    if not features.zip_code:
        return features

    zip_code = features.zip_code

    # FHFA HPI
    hpi = get_hpi_for_zip(zip_code)
    features.zip_hpi_current = hpi.get("hpi_current")
    features.zip_hpi_1yr_change = hpi.get("hpi_1yr_change")

    # Redfin price/sqft
    features.zip_median_price_per_sqft = get_median_price_per_sqft(zip_code)

    # Census ACS income
    features.zip_median_income = get_median_income(zip_code)

    return features


def blend_estimates(
    ml: PredictionResult | None,
    comps: CompsResult | None,
) -> tuple[float | None, float | None, float | None, float, float]:
    """Blend ML and comp estimates. Returns (estimate, low, high, comp_weight, ml_weight)."""
    blend_cfg = settings.blending

    if ml is None and (comps is None or comps.comp_estimate is None):
        return None, None, None, 0.0, 0.0

    if ml is None and comps and comps.comp_estimate:
        return comps.comp_estimate, comps.comp_low, comps.comp_high, 1.0, 0.0

    if comps is None or comps.comp_estimate is None or not comps.comps:
        assert ml is not None
        return ml.point_estimate, ml.low, ml.high, 0.0, 1.0

    # Determine blending weights based on comp count
    n_comps = len(comps.comps)
    if n_comps >= blend_cfg.high_comp_threshold:
        comp_w, ml_w = blend_cfg.weights["high_comp"]
    elif n_comps >= blend_cfg.mid_comp_threshold:
        comp_w, ml_w = blend_cfg.weights["mid_comp"]
    else:
        comp_w, ml_w = blend_cfg.weights["low_comp"]

    assert ml is not None
    estimate = comp_w * comps.comp_estimate + ml_w * ml.point_estimate

    # Confidence range: union of both
    low_vals = [v for v in [ml.low, comps.comp_low] if v is not None]
    high_vals = [v for v in [ml.high, comps.comp_high] if v is not None]
    low = min(low_vals) if low_vals else None
    high = max(high_vals) if high_vals else None

    return estimate, low, high, comp_w, ml_w


def estimate_price(
    address: str,
    features: PropertyFeatures,
) -> EstimateResult:
    """Run the full estimation pipeline."""
    result = EstimateResult(address=address)

    # Step 1: Geocode
    console.print("[bold]Geocoding address...[/bold]")
    geo = asyncio.run(geocode_address(address))
    result.geocoding = geo

    if geo:
        console.print(f"  Matched: {geo.matched_address}")
        console.print(f"  Lat/Lon: {geo.lat:.6f}, {geo.lon:.6f}")
        console.print(f"  ZIP: {geo.zip_code}")
        features.latitude = geo.lat
        features.longitude = geo.lon
        features.zip_code = geo.zip_code
    else:
        result.errors.append("Could not geocode address")
        console.print("[red]Could not geocode address[/red]")

    # Step 2: Enrich with market data
    console.print("[bold]Loading market context...[/bold]")
    features = enrich_features(features)
    result.features = features

    # Step 3: ML prediction
    console.print("[bold]Running ML prediction...[/bold]")
    try:
        ml_pred = predict(features)
        result.ml_prediction = ml_pred
        console.print(f"  ML estimate: ${ml_pred.point_estimate:,.0f}")
    except FileNotFoundError as e:
        result.errors.append(str(e))
        console.print(f"[red]{e}[/red]")
        ml_pred = None

    # Step 4: Comparable sales
    console.print("[bold]Searching for comparable sales...[/bold]")
    comps_result = find_comps(features)
    result.comps_result = comps_result
    console.print(f"  Found {len(comps_result.comps)} comparable sales")

    # Step 5: Blend
    console.print("[bold]Blending estimates...[/bold]")
    estimate, low, high, comp_w, ml_w = blend_estimates(ml_pred, comps_result)
    result.blended_estimate = estimate
    result.blended_low = low
    result.blended_high = high
    result.comp_weight = comp_w
    result.ml_weight = ml_w

    return result
