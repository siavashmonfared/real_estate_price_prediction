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


class Condition(str, Enum):
    """Subjective property condition, mapped to an effective age in the model.

    Comparable sales have unknown condition and are treated as AVERAGE
    (effective age == actual age). Setting the subject to a better/worse
    condition shifts its effective age so a renovated old home prices closer
    to a newer one — the appraiser's "effective age" concept.
    """

    RENOVATED = "renovated"   # recently gut/renovated; reads much newer
    UPDATED = "updated"       # partially updated, good shape
    AVERAGE = "average"       # typical for its age (default)
    DATED = "dated"           # original/needs work; reads older


class PropertyFeatures(BaseModel):
    """Core property attributes used for estimation."""

    bedrooms: int = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=20)
    sqft: float = Field(gt=0)
    lot_sqft: Optional[float] = Field(default=None, ge=0)
    year_built: int = Field(ge=1800, le=2030)
    renovation_year: Optional[int] = Field(default=None, ge=1800, le=2030)
    condition: Condition = Condition.AVERAGE
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
