"""Feature engineering transforms for the ML model."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from housing_estimator.features.property import PropertyFeatures, PropertyType

CURRENT_YEAR = datetime.now().year

# Columns expected by the trained model, in order
FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft",
    "log_sqft",
    "lot_sqft",
    "age",
    "stories",
    "latitude",
    "longitude",
    "zip_median_income",
    "zip_hpi_current",
    "zip_hpi_1yr_change",
    "zip_median_price_per_sqft",
    "type_single_family",
    "type_condo",
    "type_townhouse",
    "type_multi_family",
]


def compute_age(year_built: int) -> int:
    return max(0, CURRENT_YEAR - year_built)


def property_type_one_hot(ptype: PropertyType) -> dict[str, int]:
    mapping = {
        PropertyType.SINGLE_FAMILY: "type_single_family",
        PropertyType.CONDO: "type_condo",
        PropertyType.TOWNHOUSE: "type_townhouse",
        PropertyType.MULTI_FAMILY: "type_multi_family",
    }
    result = {col: 0 for col in mapping.values()}
    if ptype in mapping:
        result[mapping[ptype]] = 1
    return result


def features_to_dataframe(features: PropertyFeatures) -> pd.DataFrame:
    """Convert a PropertyFeatures instance to a single-row DataFrame for prediction."""
    one_hot = property_type_one_hot(features.property_type)

    row = {
        "bedrooms": features.bedrooms,
        "bathrooms": features.bathrooms,
        "sqft": features.sqft,
        "log_sqft": np.log1p(features.sqft),
        "lot_sqft": features.lot_sqft or 0.0,
        "age": compute_age(features.year_built),
        "stories": features.stories or 1.0,
        "latitude": features.latitude or 0.0,
        "longitude": features.longitude or 0.0,
        "zip_median_income": features.zip_median_income or 0.0,
        "zip_hpi_current": features.zip_hpi_current or 0.0,
        "zip_hpi_1yr_change": features.zip_hpi_1yr_change or 0.0,
        "zip_median_price_per_sqft": features.zip_median_price_per_sqft or 0.0,
        **one_hot,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def engineer_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to a training DataFrame.

    Expects columns: bedrooms, bathrooms, sqft, lot_sqft, year_built, stories,
    latitude, longitude, property_type, and optionally market context columns.
    """
    out = df.copy()

    out["log_sqft"] = np.log1p(out["sqft"].fillna(0))
    out["age"] = out["year_built"].apply(lambda y: compute_age(int(y)) if pd.notna(y) else 0)

    out["lot_sqft"] = out["lot_sqft"].fillna(0)
    out["stories"] = out["stories"].fillna(1.0)

    # One-hot encode property_type
    if "property_type" in out.columns:
        for ptype in ["single_family", "condo", "townhouse", "multi_family"]:
            col = f"type_{ptype}"
            out[col] = (out["property_type"] == ptype).astype(int)
    else:
        for ptype in ["single_family", "condo", "townhouse", "multi_family"]:
            out[f"type_{ptype}"] = 0
        out["type_single_family"] = 1

    # Fill missing market context with 0
    for col in ["zip_median_income", "zip_hpi_current", "zip_hpi_1yr_change", "zip_median_price_per_sqft"]:
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(0.0)

    return out[FEATURE_COLUMNS]


def adjust_price_by_hpi(price: float, hpi_at_sale: float, hpi_current: float) -> float:
    """Adjust a historical sale price to current dollars using FHFA HPI."""
    if hpi_at_sale <= 0 or hpi_current <= 0:
        return price
    return price * (hpi_current / hpi_at_sale)
