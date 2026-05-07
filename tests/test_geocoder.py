"""Tests for geocoder modules with mocked HTTP responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from housing_estimator.geocoder.census import geocode_census, GeocodingResult


MOCK_CENSUS_RESPONSE = {
    "result": {
        "addressMatches": [
            {
                "matchedAddress": "1600 PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500",
                "coordinates": {"x": -77.0365, "y": 38.8977},
                "addressComponents": {
                    "zip": "20500",
                    "streetName": "PENNSYLVANIA",
                    "city": "WASHINGTON",
                    "state": "DC",
                },
                "geographies": {
                    "Census Tracts": [
                        {
                            "STATE": "11",
                            "COUNTY": "001",
                            "TRACT": "006202",
                        }
                    ]
                },
            }
        ]
    }
}


MOCK_CENSUS_NO_MATCH = {
    "result": {
        "addressMatches": []
    }
}


class MockResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_geocode_census_success():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=MockResponse(MOCK_CENSUS_RESPONSE))

    with patch("housing_estimator.geocoder.census.httpx.AsyncClient", return_value=mock_client):
        result = await geocode_census("1600 Pennsylvania Ave, Washington DC")

    assert result is not None
    assert isinstance(result, GeocodingResult)
    assert abs(result.lat - 38.8977) < 0.001
    assert abs(result.lon - (-77.0365)) < 0.001
    assert result.zip_code == "20500"
    assert result.state_fips == "11"
    assert result.county_fips == "001"


@pytest.mark.asyncio
async def test_geocode_census_no_match():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=MockResponse(MOCK_CENSUS_NO_MATCH))

    with patch("housing_estimator.geocoder.census.httpx.AsyncClient", return_value=mock_client):
        result = await geocode_census("Not A Real Address 12345")

    assert result is None
