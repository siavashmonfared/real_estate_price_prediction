"""OpenStreetMap Nominatim geocoder fallback (free, no API key required)."""

from __future__ import annotations

from housing_estimator.config import settings
from housing_estimator.geocoder.census import GeocodingResult

import httpx


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


async def geocode_nominatim(address: str) -> GeocodingResult | None:
    """Geocode using OpenStreetMap Nominatim as a fallback.

    Note: Nominatim doesn't provide FIPS codes or census tracts,
    so those fields will be empty.
    """
    headers = {"User-Agent": settings.geocoding.nominatim_user_agent}
    params = {
        "q": address,
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
        "countrycodes": "us",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(NOMINATIM_URL, params=params, headers=headers)
        resp.raise_for_status()

    results = resp.json()
    if not results:
        return None

    result = results[0]
    addr = result.get("address", {})

    return GeocodingResult(
        lat=float(result["lat"]),
        lon=float(result["lon"]),
        matched_address=result.get("display_name", address),
        zip_code=addr.get("postcode", ""),
        state_fips="",
        county_fips="",
        tract="",
    )
