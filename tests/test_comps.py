"""Tests for comparable sales engine with synthetic data."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from housing_estimator.comps.engine import find_comps, CompsResult
from housing_estimator.features.property import PropertyFeatures, PropertyType


def _make_synthetic_comps() -> pd.DataFrame:
    """Create synthetic comp data near Seattle."""
    return pd.DataFrame({
        "price": [500_000, 520_000, 480_000, 550_000, 600_000, 450_000, 700_000, 400_000],
        "bedrooms": [3, 3, 3, 4, 4, 2, 5, 2],
        "bathrooms": [2.0, 2.0, 1.5, 2.5, 3.0, 1.0, 3.0, 1.0],
        "sqft": [1500, 1600, 1400, 1800, 2200, 1100, 2800, 1000],
        "lot_sqft": [5000, 5500, 4800, 6000, 7000, 3000, 8000, 2500],
        "year_built": [1990, 1995, 1985, 2000, 2010, 1975, 2015, 1960],
        "stories": [1, 2, 1, 2, 2, 1, 2, 1],
        "latitude": [47.610, 47.612, 47.608, 47.615, 47.620, 47.605, 47.630, 47.650],
        "longitude": [-122.330, -122.332, -122.328, -122.335, -122.340, -122.325, -122.350, -122.370],
        "zip_code": ["98101"] * 8,
        "property_type": ["single_family"] * 8,
        "sale_year": [2023, 2023, 2022, 2023, 2024, 2022, 2024, 2021],
    })


@pytest.fixture
def subject_property():
    return PropertyFeatures(
        bedrooms=3,
        bathrooms=2.0,
        sqft=1500.0,
        year_built=1990,
        property_type=PropertyType.SINGLE_FAMILY,
        latitude=47.611,
        longitude=-122.331,
        zip_code="98101",
    )


def test_find_comps_returns_results(subject_property):
    synthetic = _make_synthetic_comps()
    with patch("housing_estimator.comps.engine._load_comp_data", return_value=synthetic):
        result = find_comps(subject_property)

    assert isinstance(result, CompsResult)
    assert len(result.comps) > 0
    assert result.comp_estimate is not None
    assert result.comp_estimate > 0


def test_find_comps_sorted_by_similarity(subject_property):
    synthetic = _make_synthetic_comps()
    with patch("housing_estimator.comps.engine._load_comp_data", return_value=synthetic):
        result = find_comps(subject_property)

    scores = [c.similarity_score for c in result.comps]
    assert scores == sorted(scores, reverse=True)


def test_find_comps_empty_data(subject_property):
    with patch("housing_estimator.comps.engine._load_comp_data", return_value=pd.DataFrame()):
        result = find_comps(subject_property)

    assert len(result.comps) == 0
    assert result.comp_estimate is None


def test_find_comps_no_location():
    features = PropertyFeatures(
        bedrooms=3,
        bathrooms=2.0,
        sqft=1500.0,
        year_built=1990,
    )
    with patch("housing_estimator.comps.engine._load_comp_data", return_value=_make_synthetic_comps()):
        result = find_comps(features)

    assert len(result.comps) == 0


def test_comps_adjusted_prices_reasonable(subject_property):
    synthetic = _make_synthetic_comps()
    with patch("housing_estimator.comps.engine._load_comp_data", return_value=synthetic):
        result = find_comps(subject_property)

    for comp in result.comps:
        assert comp.adjusted_price > 10_000
        # Adjusted price shouldn't deviate too wildly from original
        ratio = comp.adjusted_price / comp.price
        assert 0.3 < ratio < 3.0, f"Adjusted price ratio {ratio} seems unreasonable"
