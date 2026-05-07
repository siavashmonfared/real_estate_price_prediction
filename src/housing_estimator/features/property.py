"""Property features data model."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    SINGLE_FAMILY = "single_family"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"
    MULTI_FAMILY = "multi_family"
    OTHER = "other"


class PropertyFeatures(BaseModel):
    """Core property attributes used for estimation."""

    bedrooms: int = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=20)
    sqft: float = Field(gt=0)
    lot_sqft: Optional[float] = Field(default=None, ge=0)
    year_built: int = Field(ge=1800, le=2030)
    stories: Optional[float] = Field(default=None, ge=1, le=5)
    property_type: PropertyType = PropertyType.SINGLE_FAMILY

    # Location (filled by geocoder)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zip_code: Optional[str] = None

    # Market context (filled by data sources)
    zip_median_income: Optional[float] = None
    zip_hpi_current: Optional[float] = None
    zip_hpi_1yr_change: Optional[float] = None
    zip_median_price_per_sqft: Optional[float] = None
