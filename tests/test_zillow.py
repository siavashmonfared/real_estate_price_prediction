"""Tests for Zillow property lookup with mocked responses."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from housing_estimator.datasources.zillow import (
    ZillowPropertyData,
    _address_to_zillow_slug,
    _extract_json_ld,
    _parse_property_from_json_ld,
    zillow_type_to_property_type,
    lookup_property,
)


def test_address_to_slug():
    assert _address_to_zillow_slug("123 Main St, Seattle, WA 98101") == "123-Main-St-Seattle-WA-98101"
    assert _address_to_zillow_slug("456 Oak Ave #2, Portland, OR") == "456-Oak-Ave-2-Portland-OR"


def test_extract_json_ld_single_family():
    html = """
    <html><head>
    <script type="application/ld+json">
    {"@type": "SingleFamilyResidence", "name": "123 Main St",
     "floorSize": {"value": 1850}, "numberOfRooms": 3}
    </script>
    </head></html>
    """
    result = _extract_json_ld(html)
    assert result is not None
    assert result["@type"] == "SingleFamilyResidence"


def test_extract_json_ld_no_match():
    html = "<html><head></head><body>no data here</body></html>"
    assert _extract_json_ld(html) is None


def test_parse_json_ld():
    data = {
        "@type": "SingleFamilyResidence",
        "name": "123 Main St, Seattle, WA",
        "floorSize": {"value": 1850},
        "numberOfRooms": 3,
        "numberOfBathroomsTotal": 2.5,
    }
    result = _parse_property_from_json_ld(data)
    assert result is not None
    assert result.sqft == 1850.0
    assert result.bedrooms == 3
    assert result.bathrooms == 2.5


def test_parse_json_ld_missing_data():
    data = {"@type": "SingleFamilyResidence", "name": "test"}
    result = _parse_property_from_json_ld(data)
    assert result is None  # No beds or sqft


def test_zillow_type_mapping():
    assert zillow_type_to_property_type("SINGLE_FAMILY") == "single_family"
    assert zillow_type_to_property_type("CONDO") == "condo"
    assert zillow_type_to_property_type("TOWNHOUSE") == "townhouse"
    assert zillow_type_to_property_type("MULTI_FAMILY") == "multi_family"
    assert zillow_type_to_property_type("APARTMENT") == "condo"
    assert zillow_type_to_property_type(None) == "single_family"
    assert zillow_type_to_property_type("UNKNOWN_TYPE") == "single_family"


def test_lookup_property_http_error():
    """When the HTTP request fails, should return None gracefully."""
    with patch("housing_estimator.datasources.zillow.httpx.Client") as mock_client_cls:
        import httpx
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        result = lookup_property("123 Main St, Seattle, WA")
        assert result is None


def test_lookup_property_403_blocked():
    """When Zillow returns 403, should return None gracefully."""
    with patch("housing_estimator.datasources.zillow.httpx.Client") as mock_client_cls:
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = lookup_property("123 Main St, Seattle, WA")
        assert result is None
