"""Comparable sales engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from geopy.distance import geodesic

from housing_estimator.config import settings
from housing_estimator.features.engineering import CURRENT_YEAR, adjust_price_by_hpi
from housing_estimator.features.property import PropertyFeatures


@dataclass
class CompSale:
    price: float
    adjusted_price: float
    distance_miles: float
    similarity_score: float
    bedrooms: int
    bathrooms: float
    sqft: float
    year_built: int
    address_hint: str = ""


@dataclass
class CompsResult:
    comps: list[CompSale] = field(default_factory=list)
    comp_estimate: float | None = None
    comp_low: float | None = None
    comp_high: float | None = None


def _load_comp_data() -> pd.DataFrame:
    path = settings.data.processed_path / "training_data_for_comps.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Quick haversine distance in miles."""
    return geodesic((lat1, lon1), (lat2, lon2)).miles


def _compute_similarity(
    subject: PropertyFeatures,
    comp: pd.Series,
    distance: float,
    max_distance: float,
) -> float:
    """Score similarity between subject property and a comp (0-1, higher = more similar)."""
    weights = settings.comps.weights

    # Sqft similarity (0-1)
    sqft_diff = abs(subject.sqft - comp.get("sqft", 0)) / max(subject.sqft, 1)
    sqft_score = max(0, 1 - sqft_diff)

    # Bed/bath similarity
    bed_diff = abs(subject.bedrooms - comp.get("bedrooms", 0))
    bath_diff = abs(subject.bathrooms - comp.get("bathrooms", 0))
    bed_bath_score = max(0, 1 - (bed_diff * 0.3 + bath_diff * 0.2))

    # Age similarity
    subject_age = CURRENT_YEAR - subject.year_built
    comp_age = CURRENT_YEAR - comp.get("year_built", CURRENT_YEAR)
    age_diff = abs(subject_age - comp_age)
    age_score = max(0, 1 - age_diff / 50)

    # Property type match
    comp_type = comp.get("property_type", "single_family")
    type_score = 1.0 if comp_type == subject.property_type.value else 0.3

    # Distance/recency score (closer and more recent = better)
    distance_score = max(0, 1 - distance / max_distance)

    total = (
        weights.get("sqft", 0.3) * sqft_score
        + weights.get("bed_bath", 0.2) * bed_bath_score
        + weights.get("age", 0.15) * age_score
        + weights.get("property_type", 0.15) * type_score
        + weights.get("recency", 0.2) * distance_score
    )

    return total


def _adjust_comp_price(
    subject: PropertyFeatures,
    comp: pd.Series,
    raw_price: float,
) -> float:
    """Adjust a comp's price based on differences from the subject."""
    adjusted = raw_price

    # Sqft adjustment
    comp_sqft = comp.get("sqft", subject.sqft)
    if comp_sqft > 0:
        price_per_sqft = raw_price / comp_sqft
        sqft_diff = subject.sqft - comp_sqft
        adjusted += sqft_diff * price_per_sqft * 0.5  # 50% of $/sqft diff

    # Bedroom adjustment (~$10k per bedroom as a rough heuristic)
    bed_diff = subject.bedrooms - comp.get("bedrooms", subject.bedrooms)
    adjusted += bed_diff * 10_000

    # Age adjustment (~$1k per year newer)
    subject_age = CURRENT_YEAR - subject.year_built
    comp_age = CURRENT_YEAR - comp.get("year_built", subject.year_built)
    age_diff = comp_age - subject_age  # positive = comp is older
    adjusted += age_diff * 1_000

    return max(adjusted, 10_000)


def find_comps(subject: PropertyFeatures) -> CompsResult:
    """Find comparable sales for a subject property."""
    df = _load_comp_data()
    if df.empty or subject.latitude is None or subject.longitude is None:
        return CompsResult()

    # Filter to rows with valid lat/lon
    df = df.dropna(subset=["latitude", "longitude", "price"])

    all_comps: list[tuple[float, float, pd.Series]] = []  # (distance, similarity, row)

    for radius in settings.comps.radii_miles:
        for _, row in df.iterrows():
            dist = _haversine_miles(
                subject.latitude, subject.longitude,
                row["latitude"], row["longitude"],
            )
            if dist <= radius:
                sim = _compute_similarity(subject, row, dist, radius)
                all_comps.append((dist, sim, row))

        # Deduplicate by keeping the best score for each row
        if len(all_comps) >= settings.comps.min_comps:
            break

    if not all_comps:
        return CompsResult()

    # Sort by similarity (descending), take top N
    all_comps.sort(key=lambda x: x[1], reverse=True)

    seen_indices: set[int] = set()
    unique_comps: list[tuple[float, float, pd.Series]] = []
    for dist, sim, row in all_comps:
        row_id = id(row)
        if row_id not in seen_indices:
            seen_indices.add(row_id)
            unique_comps.append((dist, sim, row))
        if len(unique_comps) >= settings.comps.max_comps:
            break

    # Build comp sales with adjustments
    comp_sales: list[CompSale] = []
    for dist, sim, row in unique_comps:
        raw_price = float(row["price"])
        adj_price = _adjust_comp_price(subject, row, raw_price)

        comp_sales.append(CompSale(
            price=raw_price,
            adjusted_price=adj_price,
            distance_miles=round(dist, 2),
            similarity_score=round(sim, 3),
            bedrooms=int(row.get("bedrooms", 0)),
            bathrooms=float(row.get("bathrooms", 0)),
            sqft=float(row.get("sqft", 0)),
            year_built=int(row.get("year_built", 0)),
        ))

    # Weighted average of adjusted prices (weighted by similarity)
    if comp_sales:
        weights = np.array([c.similarity_score for c in comp_sales])
        prices = np.array([c.adjusted_price for c in comp_sales])
        if weights.sum() > 0:
            comp_estimate = float(np.average(prices, weights=weights))
        else:
            comp_estimate = float(np.mean(prices))

        comp_low = float(np.percentile(prices, 10))
        comp_high = float(np.percentile(prices, 90))
    else:
        comp_estimate = None
        comp_low = None
        comp_high = None

    return CompsResult(
        comps=comp_sales,
        comp_estimate=comp_estimate,
        comp_low=comp_low,
        comp_high=comp_high,
    )
