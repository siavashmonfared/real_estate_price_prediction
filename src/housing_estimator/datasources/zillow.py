"""Zillow property detail lookup via public web pages.

Zillow embeds structured JSON-LD data on property pages, which includes
beds, baths, sqft, lot size, year built, and property type. No API key needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

import httpx
from rich.console import Console

console = Console()

_ZILLOW_SEARCH_URL = "https://www.zillow.com/homes/{query}_rb/"
_ZILLOW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


@dataclass
class ZillowPropertyData:
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    sqft: Optional[float] = None
    lot_sqft: Optional[float] = None
    year_built: Optional[int] = None
    property_type: Optional[str] = None
    zestimate: Optional[float] = None
    address: Optional[str] = None
    url: Optional[str] = None


def _address_to_zillow_slug(address: str) -> str:
    """Convert an address to a Zillow URL-friendly slug."""
    slug = address.strip()
    slug = re.sub(r"[,#]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _extract_json_ld(html: str) -> dict | None:
    """Extract JSON-LD structured data from the HTML page."""
    pattern = r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>'
    matches = re.findall(pattern, html, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            # Look for SingleFamilyResidence or similar
            if isinstance(data, dict):
                if data.get("@type") in (
                    "SingleFamilyResidence", "Residence", "House",
                    "Apartment", "Product",
                ):
                    return data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("@type") in (
                        "SingleFamilyResidence", "Residence", "House",
                        "Apartment", "Product",
                    ):
                        return item
        except json.JSONDecodeError:
            continue
    return None


def _extract_next_data(html: str) -> dict | None:
    """Extract property data from Zillow's __NEXT_DATA__ or preloaded state."""
    # Try __NEXT_DATA__
    pattern = r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try gdp-data or preloaded API responses
    pattern = r'"apiCache"\s*:\s*(\{.+?\})\s*[,}]'
    match = re.search(pattern, html, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def _parse_property_from_next_data(data: dict) -> ZillowPropertyData | None:
    """Navigate the __NEXT_DATA__ structure to find property details."""
    result = ZillowPropertyData()

    # Zillow's NEXT_DATA nests property info in various locations
    # Try the common path: props.pageProps.componentProps.gdpClientCache
    try:
        props = data.get("props", {}).get("pageProps", {})

        # Try componentProps path
        comp_props = props.get("componentProps", {})
        gdp_cache = comp_props.get("gdpClientCache", "{}")
        if isinstance(gdp_cache, str):
            gdp_cache = json.loads(gdp_cache)

        # The cache contains stringified JSON keyed by query hashes
        property_data = None
        for value in (gdp_cache.values() if isinstance(gdp_cache, dict) else []):
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = value
            else:
                parsed = value

            if isinstance(parsed, dict) and "property" in parsed:
                property_data = parsed["property"]
                break

        if not property_data:
            # Try direct path
            property_data = props.get("property", {})

        if property_data:
            result.bedrooms = property_data.get("bedrooms")
            result.bathrooms = property_data.get("bathrooms")

            living_area = property_data.get("livingArea") or property_data.get("livingAreaValue")
            if living_area:
                result.sqft = float(living_area)

            lot_size = property_data.get("lotSize") or property_data.get("lotAreaValue")
            if lot_size:
                result.lot_sqft = float(lot_size)

            result.year_built = property_data.get("yearBuilt")
            result.zestimate = property_data.get("zestimate")
            result.property_type = property_data.get("homeType")

            addr = property_data.get("address", {})
            if isinstance(addr, dict):
                parts = [
                    addr.get("streetAddress", ""),
                    addr.get("city", ""),
                    addr.get("state", ""),
                    addr.get("zipcode", ""),
                ]
                result.address = ", ".join(p for p in parts if p)

            return result if any([result.bedrooms, result.sqft]) else None
    except (KeyError, TypeError, json.JSONDecodeError):
        pass

    return None


def _parse_property_from_json_ld(data: dict) -> ZillowPropertyData | None:
    """Parse property details from JSON-LD structured data."""
    result = ZillowPropertyData()

    result.address = data.get("name") or data.get("description", "")
    result.url = data.get("url")

    floor_size = data.get("floorSize", {})
    if isinstance(floor_size, dict):
        val = floor_size.get("value")
        if val:
            result.sqft = float(val)

    num_rooms = data.get("numberOfRooms")
    if num_rooms:
        result.bedrooms = int(num_rooms)

    # JSON-LD sometimes uses different schema
    result.bedrooms = result.bedrooms or data.get("numberOfBedrooms")
    result.bathrooms = data.get("numberOfBathroomsTotal") or data.get("numberOfFullBathrooms")

    return result if any([result.bedrooms, result.sqft]) else None


def lookup_property(address: str) -> ZillowPropertyData | None:
    """Look up property details from Zillow for the given address.

    Returns None if the lookup fails or no data is found.
    """
    slug = _address_to_zillow_slug(address)
    url = _ZILLOW_SEARCH_URL.format(query=slug)

    try:
        with httpx.Client(
            timeout=15.0,
            follow_redirects=True,
            headers=_ZILLOW_HEADERS,
        ) as client:
            resp = client.get(url)
            if resp.status_code == 403:
                console.print("  [yellow]Zillow blocked the request (captcha/rate limit)[/yellow]")
                return None
            if resp.status_code != 200:
                return None

            html = resp.text
    except httpx.HTTPError as e:
        console.print(f"  [yellow]Zillow request failed: {e}[/yellow]")
        return None

    # Try JSON-LD first (most reliable structured data)
    json_ld = _extract_json_ld(html)
    if json_ld:
        result = _parse_property_from_json_ld(json_ld)
        if result:
            result.url = str(resp.url)
            return result

    # Fall back to __NEXT_DATA__
    next_data = _extract_next_data(html)
    if next_data:
        result = _parse_property_from_next_data(next_data)
        if result:
            result.url = str(resp.url)
            return result

    return None


def zillow_type_to_property_type(zillow_type: str | None) -> str:
    """Map Zillow's homeType to our PropertyType enum values."""
    if not zillow_type:
        return "single_family"

    mapping = {
        "SINGLE_FAMILY": "single_family",
        "CONDO": "condo",
        "TOWNHOUSE": "townhouse",
        "MULTI_FAMILY": "multi_family",
        "APARTMENT": "condo",
        "COOPERATIVE": "condo",
        "MANUFACTURED": "single_family",
        "LOT": "other",
    }
    return mapping.get(zillow_type.upper(), "single_family")
