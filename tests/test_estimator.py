"""End-to-end tests for the estimator with mocked data sources."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from housing_estimator.comps.engine import CompsResult, CompSale
from housing_estimator.estimator import (
    EstimateResult,
    blend_estimates,
    enrich_features,
)
from housing_estimator.features.property import PropertyFeatures, PropertyType
from housing_estimator.geocoder.census import GeocodingResult
from housing_estimator.models.predict import PredictionResult


def test_blend_estimates_ml_only():
    ml = PredictionResult(point_estimate=500_000, low=400_000, high=600_000)
    estimate, low, high, comp_w, ml_w = blend_estimates(ml, None)

    assert estimate == 500_000
    assert low == 400_000
    assert high == 600_000
    assert comp_w == 0.0
    assert ml_w == 1.0


def test_blend_estimates_comps_only():
    comps = CompsResult(
        comps=[CompSale(500_000, 510_000, 0.5, 0.9, 3, 2.0, 1500, 1990)] * 3,
        comp_estimate=510_000,
        comp_low=480_000,
        comp_high=540_000,
    )
    estimate, low, high, comp_w, ml_w = blend_estimates(None, comps)

    assert estimate == 510_000
    assert comp_w == 1.0
    assert ml_w == 0.0


def test_blend_estimates_high_comps():
    """With 5+ comps, should weight 60% comp / 40% ML."""
    ml = PredictionResult(point_estimate=500_000, low=400_000, high=600_000)
    comps = CompsResult(
        comps=[CompSale(500_000, 520_000, 0.5, 0.9, 3, 2.0, 1500, 1990)] * 6,
        comp_estimate=520_000,
        comp_low=490_000,
        comp_high=550_000,
    )
    estimate, low, high, comp_w, ml_w = blend_estimates(ml, comps)

    expected = 0.60 * 520_000 + 0.40 * 500_000
    assert estimate == pytest.approx(expected)
    assert comp_w == 0.60
    assert ml_w == 0.40


def test_blend_estimates_mid_comps():
    """With 3-4 comps, should weight 40% comp / 60% ML."""
    ml = PredictionResult(point_estimate=500_000, low=400_000, high=600_000)
    comps = CompsResult(
        comps=[CompSale(500_000, 520_000, 0.5, 0.9, 3, 2.0, 1500, 1990)] * 4,
        comp_estimate=520_000,
        comp_low=490_000,
        comp_high=550_000,
    )
    estimate, _, _, comp_w, ml_w = blend_estimates(ml, comps)

    expected = 0.40 * 520_000 + 0.60 * 500_000
    assert estimate == pytest.approx(expected)
    assert comp_w == 0.40


def test_blend_estimates_low_comps():
    """With <3 comps, should weight 20% comp / 80% ML."""
    ml = PredictionResult(point_estimate=500_000, low=400_000, high=600_000)
    comps = CompsResult(
        comps=[CompSale(500_000, 520_000, 0.5, 0.9, 3, 2.0, 1500, 1990)] * 2,
        comp_estimate=520_000,
        comp_low=490_000,
        comp_high=550_000,
    )
    estimate, _, _, comp_w, ml_w = blend_estimates(ml, comps)

    expected = 0.20 * 520_000 + 0.80 * 500_000
    assert estimate == pytest.approx(expected)


def test_blend_estimates_none():
    estimate, low, high, comp_w, ml_w = blend_estimates(None, None)
    assert estimate is None


def test_enrich_features_with_no_zip():
    features = PropertyFeatures(
        bedrooms=3, bathrooms=2.0, sqft=1500.0, year_built=1990,
    )
    result = enrich_features(features)
    # Should return features unchanged (no zip to look up)
    assert result.zip_median_income is None


@patch("housing_estimator.estimator.get_median_income", return_value=75_000.0)
@patch("housing_estimator.estimator.get_median_price_per_sqft", return_value=350.0)
@patch("housing_estimator.estimator.get_hpi_for_zip", return_value={"hpi_current": 450.0, "hpi_1yr_change": 0.05})
def test_enrich_features_with_zip(mock_hpi, mock_redfin, mock_income):
    features = PropertyFeatures(
        bedrooms=3, bathrooms=2.0, sqft=1500.0, year_built=1990,
        zip_code="98101",
    )
    result = enrich_features(features)

    assert result.zip_median_income == 75_000.0
    assert result.zip_median_price_per_sqft == 350.0
    assert result.zip_hpi_current == 450.0
    assert result.zip_hpi_1yr_change == 0.05
