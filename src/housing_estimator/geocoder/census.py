"""US Census Bureau Geocoder (free, no API key required)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class GeocodingResult:
    lat: float
    lon: float
    matched_address: str
    zip_code: str
    state_fips: str
    county_fips: str
    tract: str


CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"


async def geocode_census(address: str) -> GeocodingResult | None:
    """Geocode an address using the US Census Bureau geocoder.

    Returns None if no match is found.
    """
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(CENSUS_GEOCODER_URL, params=params)
        resp.raise_for_status()

    data = resp.json()
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        return None

    match = matches[0]
    coords = match["coordinates"]
    geo = match.get("geographies", {})

    # Extract census tract info
    tracts = geo.get("Census Tracts", [{}])
    tract_info = tracts[0] if tracts else {}

    return GeocodingResult(
        lat=float(coords["y"]),
        lon=float(coords["x"]),
        matched_address=match.get("matchedAddress", address),
        zip_code=match.get("addressComponents", {}).get("zip", ""),
        state_fips=tract_info.get("STATE", ""),
        county_fips=tract_info.get("COUNTY", ""),
        tract=tract_info.get("TRACT", ""),
    )
