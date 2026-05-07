"""Tests for feature engineering transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from housing_estimator.features.engineering import (
    FEATURE_COLUMNS,
    compute_age,
    engineer_training_features,
    features_to_dataframe,
    adjust_price_by_hpi,
    CURRENT_YEAR,
)
from housing_estimator.features.property import PropertyFeatures, PropertyType


def test_compute_age():
    assert compute_age(2000) == CURRENT_YEAR - 2000
    assert compute_age(CURRENT_YEAR) == 0
    assert compute_age(CURRENT_YEAR + 5) == 0  # future year clamps to 0


def test_features_to_dataframe():
    features = PropertyFeatures(
        bedrooms=3,
        bathrooms=2.0,
        sqft=1500.0,
        year_built=1990,
        property_type=PropertyType.SINGLE_FAMILY,
        latitude=47.6,
        longitude=-122.3,
    )
    df = features_to_dataframe(features)

    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1
    assert df["bedrooms"].iloc[0] == 3
    assert df["bathrooms"].iloc[0] == 2.0
    assert df["sqft"].iloc[0] == 1500.0
    assert df["log_sqft"].iloc[0] == pytest.approx(np.log1p(1500.0), rel=1e-5)
    assert df["age"].iloc[0] == CURRENT_YEAR - 1990
    assert df["type_single_family"].iloc[0] == 1
    assert df["type_condo"].iloc[0] == 0


def test_features_to_dataframe_condo():
    features = PropertyFeatures(
        bedrooms=2,
        bathrooms=1.0,
        sqft=800.0,
        year_built=2010,
        property_type=PropertyType.CONDO,
    )
    df = features_to_dataframe(features)
    assert df["type_condo"].iloc[0] == 1
    assert df["type_single_family"].iloc[0] == 0


def test_engineer_training_features():
    data = pd.DataFrame({
        "bedrooms": [3, 4],
        "bathrooms": [2.0, 3.0],
        "sqft": [1500.0, 2000.0],
        "lot_sqft": [5000.0, None],
        "year_built": [1990, 2005],
        "stories": [2.0, None],
        "latitude": [47.6, 47.7],
        "longitude": [-122.3, -122.4],
        "property_type": ["single_family", "condo"],
    })

    result = engineer_training_features(data)

    assert list(result.columns) == FEATURE_COLUMNS
    assert len(result) == 2
    assert result["lot_sqft"].iloc[1] == 0.0  # filled NaN
    assert result["stories"].iloc[1] == 1.0  # filled NaN
    assert result["type_single_family"].iloc[0] == 1
    assert result["type_condo"].iloc[1] == 1


def test_adjust_price_by_hpi():
    # 10% appreciation
    adjusted = adjust_price_by_hpi(100_000, 200.0, 220.0)
    assert adjusted == pytest.approx(110_000, rel=1e-5)

    # No adjustment when HPI is invalid
    assert adjust_price_by_hpi(100_000, 0.0, 220.0) == 100_000
    assert adjust_price_by_hpi(100_000, 200.0, 0.0) == 100_000
