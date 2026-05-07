"""Recent sales lookup via Redfin's public search API.

Fetches individual recently sold properties near a given lat/lon coordinate.
No API key required — uses Redfin's public-facing search endpoints.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import httpx
from geopy.distance import geodesic
from rich.console import Console

console = Console()

_REDFIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Redfin's stingray API for map-based search
_REDFIN_GIS_URL = "https://www.redfin.com/stingray/api/gis"
# Redfin's search endpoint for sold homes
_REDFIN_SEARCH_URL = "https://www.redfin.com/stingray/api/gis-csv"


@dataclass
class SoldProperty:
    address: str
    price: float
    sale_date: str
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft: Optional[float] = None
    lot_sqft: Optional[float] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_miles: Optional[float] = None
    price_per_sqft: Optional[float] = None
    url: Optional[str] = None


def _bbox_from_center(lat: float, lon: float, radius_miles: float) -> dict:
    """Calculate a bounding box from a center point and radius."""
    # Approximate: 1 degree lat ~ 69 miles, 1 degree lon ~ 69 * cos(lat) miles
    import math
    lat_delta = radius_miles / 69.0
    lon_delta = radius_miles / (69.0 * math.cos(math.radians(lat)))
    return {
        "south": lat - lat_delta,
        "north": lat + lat_delta,
        "west": lon - lon_delta,
        "east": lon + lon_delta,
    }


def fetch_recent_sales_redfin(
    lat: float,
    lon: float,
    radius_miles: float = 1.0,
    days_back: int = 90,
) -> list[SoldProperty]:
    """Fetch recently sold properties from Redfin near a given coordinate.

    Uses Redfin's GIS CSV download endpoint which returns sold homes in a bounding box.
    """
    bbox = _bbox_from_center(lat, lon, radius_miles)
    sold_within = datetime.now() - timedelta(days=days_back)
    sold_within_str = sold_within.strftime("%Y-%m-%d")

    params = {
        "al": 1,
        "num_homes": 350,
        "ord": "redfin-recommended-asc",
        "page_number": 1,
        "poly": _bbox_to_poly(bbox),
        "sold_within_days": days_back,
        "status": 9,  # sold
        "uipt": "1,2,3,4,5,6",  # all property types
        "v": 8,
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=_REDFIN_HEADERS) as client:
            resp = client.get(_REDFIN_SEARCH_URL, params=params)
            if resp.status_code == 403:
                console.print("  [yellow]Redfin blocked the request (rate limit)[/yellow]")
                return _fallback_redfin_scrape(lat, lon, radius_miles, days_back)
            if resp.status_code != 200:
                console.print(f"  [yellow]Redfin CSV returned {resp.status_code}[/yellow]")
                return _fallback_redfin_scrape(lat, lon, radius_miles, days_back)

            return _parse_redfin_csv(resp.text, lat, lon, radius_miles)
    except httpx.HTTPError as e:
        console.print(f"  [yellow]Redfin request failed: {e}[/yellow]")
        return _fallback_redfin_scrape(lat, lon, radius_miles, days_back)


def _bbox_to_poly(bbox: dict) -> str:
    """Convert bounding box to Redfin polygon format."""
    s, n, w, e = bbox["south"], bbox["north"], bbox["west"], bbox["east"]
    # Redfin uses a simple polygon string: lng lat pairs
    return f"{w:.6f} {s:.6f},{e:.6f} {s:.6f},{e:.6f} {n:.6f},{w:.6f} {n:.6f},{w:.6f} {s:.6f}"


def _parse_redfin_csv(csv_text: str, center_lat: float, center_lon: float, radius_miles: float) -> list[SoldProperty]:
    """Parse Redfin's CSV download response into SoldProperty objects."""
    import csv
    import io

    results: list[SoldProperty] = []
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        try:
            price_str = (row.get("PRICE") or row.get("SOLD PRICE") or "")
            price_str = price_str.replace("$", "").replace(",", "").strip()
            if not price_str:
                continue
            price = float(price_str)
            if price <= 0:
                continue

            lat_str = (row.get("LATITUDE") or "").strip()
            lon_str = (row.get("LONGITUDE") or "").strip()
            if not lat_str or not lon_str:
                continue

            prop_lat = float(lat_str)
            prop_lon = float(lon_str)

            distance = geodesic((center_lat, center_lon), (prop_lat, prop_lon)).miles
            if distance > radius_miles:
                continue

            sqft_str = (row.get("SQUARE FEET") or "").replace(",", "").strip()
            sqft = float(sqft_str) if sqft_str else None

            beds_str = (row.get("BEDS") or "").strip()
            beds = int(float(beds_str)) if beds_str else None

            baths_str = (row.get("BATHS") or "").strip()
            baths = float(baths_str) if baths_str else None

            lot_str = (row.get("LOT SIZE") or "").replace(",", "").strip()
            lot = float(lot_str) if lot_str else None

            year_str = (row.get("YEAR BUILT") or "").strip()
            year = int(year_str) if year_str else None

            sale_date = (row.get("SOLD DATE") or row.get("LAST SALE DATE") or "").strip()

            address_parts = [
                row.get("ADDRESS", ""),
                row.get("CITY", ""),
                row.get("STATE OR PROVINCE", row.get("STATE", "")),
                row.get("ZIP OR POSTAL CODE", row.get("ZIP", "")),
            ]
            address = ", ".join(p for p in address_parts if p)

            ppsf = (price / sqft) if sqft and sqft > 0 else None

            results.append(SoldProperty(
                address=address,
                price=price,
                sale_date=sale_date,
                bedrooms=beds,
                bathrooms=baths,
                sqft=sqft,
                lot_sqft=lot,
                year_built=year,
                property_type=row.get("PROPERTY TYPE", None),
                latitude=prop_lat,
                longitude=prop_lon,
                distance_miles=round(distance, 2),
                price_per_sqft=round(ppsf, 2) if ppsf else None,
                url=row.get("URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)")
                   or row.get("URL")
                   or None,
            ))
        except (ValueError, TypeError):
            continue

    results.sort(key=lambda x: x.distance_miles or 999)
    return results


def _fallback_redfin_scrape(
    lat: float, lon: float, radius_miles: float, days_back: int
) -> list[SoldProperty]:
    """Fallback: scrape Redfin's HTML search results for sold homes."""
    bbox = _bbox_from_center(lat, lon, radius_miles)

    # Build a Redfin filter URL for sold homes
    # Redfin uses URL-encoded filter params
    params = {
        "al": 1,
        "isMapView": "true",
        "market": "national",
        "num_homes": 100,
        "ord": "days-on-redfin-asc",
        "page_number": 1,
        "poly": _bbox_to_poly(bbox),
        "region_id": 0,
        "region_type": 6,
        "sold_within_days": days_back,
        "status": 9,
        "uipt": "1,2,3,4,5,6",
        "v": 8,
    }

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=_REDFIN_HEADERS) as client:
            resp = client.get(_REDFIN_GIS_URL, params=params)
            if resp.status_code != 200:
                return []

            # Redfin GIS API returns "{}&&{...json...}"
            text = resp.text
            if text.startswith("{}&&"):
                text = text[4:]

            data = json.loads(text)
            homes = data.get("payload", {}).get("homes", [])

            results: list[SoldProperty] = []
            for home in homes:
                try:
                    price = home.get("price", {}).get("value", 0)
                    if price <= 0:
                        continue

                    lat_h = home.get("latLong", {}).get("latitude")
                    lon_h = home.get("latLong", {}).get("longitude")
                    if not lat_h or not lon_h:
                        continue

                    distance = geodesic((lat, lon), (lat_h, lon_h)).miles
                    if distance > radius_miles:
                        continue

                    sqft = home.get("sqFt", {}).get("value")
                    beds = home.get("beds")
                    baths = home.get("baths")
                    year_built = home.get("yearBuilt", {}).get("value") if isinstance(home.get("yearBuilt"), dict) else home.get("yearBuilt")
                    lot_size = home.get("lotSize", {}).get("value") if isinstance(home.get("lotSize"), dict) else None
                    sold_date = home.get("soldDate", home.get("lastSaleDate", ""))
                    if isinstance(sold_date, (int, float)):
                        sold_date = datetime.fromtimestamp(sold_date / 1000).strftime("%Y-%m-%d")

                    addr_info = home.get("streetLine", {})
                    address = addr_info.get("value", "") if isinstance(addr_info, dict) else str(addr_info)
                    city = home.get("city", "")
                    state = home.get("state", "")
                    zip_code = home.get("zip", "")
                    full_addr = f"{address}, {city}, {state} {zip_code}".strip(", ")

                    ptype = home.get("propertyType")
                    ppsf = (price / sqft) if sqft and sqft > 0 else None

                    url = home.get("url", "")
                    if url and not url.startswith("http"):
                        url = f"https://www.redfin.com{url}"

                    results.append(SoldProperty(
                        address=full_addr,
                        price=price,
                        sale_date=str(sold_date),
                        bedrooms=beds,
                        bathrooms=baths,
                        sqft=float(sqft) if sqft else None,
                        lot_sqft=float(lot_size) if lot_size else None,
                        year_built=int(year_built) if year_built else None,
                        property_type=str(ptype) if ptype else None,
                        latitude=lat_h,
                        longitude=lon_h,
                        distance_miles=round(distance, 2),
                        price_per_sqft=round(ppsf, 2) if ppsf else None,
                        url=url,
                    ))
                except (ValueError, TypeError, KeyError):
                    continue

            results.sort(key=lambda x: x.distance_miles or 999)
            return results

    except (httpx.HTTPError, json.JSONDecodeError) as e:
        console.print(f"  [yellow]Redfin fallback failed: {e}[/yellow]")
        return []
