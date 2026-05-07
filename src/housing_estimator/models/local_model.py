"""Local model — train on nearby Redfin sales data and predict for a subject property.

Provides full transparency: train/validation/test splits, metrics on each,
data summary, and feature importances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor

from housing_estimator.datasources.recent_sales import SoldProperty, fetch_recent_sales_redfin
from housing_estimator.features.property import PropertyFeatures

console = Console()

CURRENT_YEAR = datetime.now().year

# Features used by the local model
LOCAL_FEATURES = [
    "sqft",
    "log_sqft",
    "bedrooms",
    "bathrooms",
    "age",
    "lot_sqft",
    "distance_miles",
    "price_per_sqft_neighborhood",
]


@dataclass
class SplitMetrics:
    """Metrics for a single data split."""
    name: str
    n_samples: int
    mae: float
    mape: float
    r2: float
    median_error: float
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)


@dataclass
class DataSummary:
    """Summary statistics of the training data."""
    total_sales: int
    usable_sales: int
    price_min: float
    price_max: float
    price_median: float
    price_mean: float
    sqft_min: float
    sqft_max: float
    sqft_median: float
    bed_range: str
    bath_range: str
    year_built_range: str
    distance_range: str
    search_radius_miles: float
    search_days_back: int


@dataclass
class LocalEstimate:
    point_estimate: float
    low: float
    high: float
    n_training_samples: int
    cv_mae: float
    cv_r2: float
    median_price: float
    median_ppsf: float
    # New: transparent reporting
    train_metrics: SplitMetrics | None = None
    val_metrics: SplitMetrics | None = None
    test_metrics: SplitMetrics | None = None
    data_summary: DataSummary | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)


def _sales_to_dataframe(sales: list[SoldProperty]) -> pd.DataFrame:
    """Convert a list of SoldProperty into a training DataFrame."""
    rows = []
    for s in sales:
        if s.price <= 0 or not s.sqft or s.sqft <= 0:
            continue
        rows.append({
            "price": s.price,
            "sqft": s.sqft,
            "bedrooms": s.bedrooms or 0,
            "bathrooms": s.bathrooms or 0,
            "year_built": s.year_built or CURRENT_YEAR - 30,
            "lot_sqft": s.lot_sqft or 0,
            "distance_miles": s.distance_miles or 0,
            "property_type": s.property_type or "Single Family Residential",
        })
    return pd.DataFrame(rows)


def _engineer_local_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the local model."""
    out = df.copy()
    out["log_sqft"] = np.log1p(out["sqft"])
    out["age"] = CURRENT_YEAR - out["year_built"].clip(upper=CURRENT_YEAR)
    out["lot_sqft"] = out["lot_sqft"].fillna(0)

    # Neighborhood median price/sqft as a feature for each row
    overall_ppsf = (out["price"] / out["sqft"]).median()
    out["price_per_sqft_neighborhood"] = overall_ppsf

    return out


def _subject_to_row(
    features: PropertyFeatures,
    neighborhood_ppsf: float,
) -> pd.DataFrame:
    """Convert subject property features to a single-row DataFrame matching LOCAL_FEATURES."""
    row = {
        "sqft": features.sqft,
        "log_sqft": np.log1p(features.sqft),
        "bedrooms": features.bedrooms,
        "bathrooms": features.bathrooms,
        "age": max(0, CURRENT_YEAR - features.year_built),
        "lot_sqft": features.lot_sqft or 0,
        "distance_miles": 0.0,  # subject is at center
        "price_per_sqft_neighborhood": neighborhood_ppsf,
    }
    return pd.DataFrame([row], columns=LOCAL_FEATURES)


def _compute_split_metrics(
    name: str,
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> SplitMetrics:
    """Compute MAE, MAPE, R², and median error for a given split."""
    preds = model.predict(X)
    actuals = y.values
    errors = np.abs(actuals - preds)
    pct_errors = errors / np.maximum(actuals, 1)

    return SplitMetrics(
        name=name,
        n_samples=len(y),
        mae=float(np.mean(errors)),
        mape=float(np.mean(pct_errors)),
        r2=float(1 - np.sum((actuals - preds) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)),
        median_error=float(np.median(errors)),
        predictions=preds.tolist(),
        actuals=actuals.tolist(),
    )


def _build_data_summary(
    df: pd.DataFrame,
    total_raw: int,
    search_radius: float,
    search_days: int,
) -> DataSummary:
    """Build a summary of the data used for modeling."""
    return DataSummary(
        total_sales=total_raw,
        usable_sales=len(df),
        price_min=float(df["price"].min()),
        price_max=float(df["price"].max()),
        price_median=float(df["price"].median()),
        price_mean=float(df["price"].mean()),
        sqft_min=float(df["sqft"].min()),
        sqft_max=float(df["sqft"].max()),
        sqft_median=float(df["sqft"].median()),
        bed_range=f"{int(df['bedrooms'].min())}-{int(df['bedrooms'].max())}",
        bath_range=f"{df['bathrooms'].min():.0f}-{df['bathrooms'].max():.0f}",
        year_built_range=f"{int(df['year_built'].min())}-{int(df['year_built'].max())}",
        distance_range=f"{df['distance_miles'].min():.1f}-{df['distance_miles'].max():.1f} mi",
        search_radius_miles=search_radius,
        search_days_back=search_days,
    )


def fetch_training_sales(
    lat: float,
    lon: float,
    target_count: int = 200,
) -> tuple[list[SoldProperty], float, int]:
    """Fetch enough sales for training by expanding radius and lookback.

    Returns (sales, radius_used, days_used).
    """
    search_configs = [
        (3.0, 365),
        (5.0, 365),
        (5.0, 730),
        (10.0, 730),
        (10.0, 1095),
    ]

    best_sales: list[SoldProperty] = []
    best_radius = 0.0
    best_days = 0

    for radius, days in search_configs:
        console.print(f"  Searching {radius} mi / {days} days...")
        sales = fetch_recent_sales_redfin(lat, lon, radius_miles=radius, days_back=days)
        # Filter to ones with sqft data (needed for training)
        valid = [s for s in sales if s.sqft and s.sqft > 0 and s.price > 0]
        console.print(f"    Found {len(sales)} sales ({len(valid)} with sqft data)")

        if len(valid) > len(best_sales):
            best_sales = valid
            best_radius = radius
            best_days = days

        if len(valid) >= target_count:
            break

    return best_sales, best_radius, best_days


def train_and_predict(
    features: PropertyFeatures,
    sales: list[SoldProperty],
    search_radius: float = 0.0,
    search_days: int = 0,
) -> LocalEstimate | None:
    """Train a local XGBoost model on nearby sales and predict the subject property.

    Uses a proper 70/15/15 train/validation/test split with metrics on each.
    Returns None if there isn't enough data.
    """
    df_raw = _sales_to_dataframe(sales)
    if len(df_raw) < 20:
        console.print(f"  [yellow]Only {len(df_raw)} usable sales — need at least 20[/yellow]")
        return None

    total_raw = len(sales)
    console.print(f"  {len(df_raw)} usable sales out of {total_raw} total fetched")

    # Engineer features
    df = _engineer_local_features(df_raw)
    neighborhood_ppsf = (df["price"] / df["sqft"]).median()

    # Build data summary BEFORE splitting
    data_summary = _build_data_summary(df, total_raw, search_radius, search_days)

    X = df[LOCAL_FEATURES]
    y = df["price"]

    # --- 70/15/15 Train / Validation / Test Split ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42,
    )
    # Split remaining into train (82.35% of 85% ≈ 70%) and val (17.65% of 85% ≈ 15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42,
    )

    console.print(f"\n  [bold]Data Split:[/bold]")
    console.print(f"    Train:      {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
    console.print(f"    Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.0f}%)")
    console.print(f"    Test:       {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")

    # Tune depth based on training sample size
    n_train = len(X_train)
    max_depth = 3 if n_train < 50 else 4 if n_train < 150 else 5
    n_estimators = 100 if n_train < 50 else 200 if n_train < 150 else 300

    console.print(f"\n  [bold]Model Config:[/bold]")
    console.print(f"    XGBoost: n_estimators={n_estimators}, max_depth={max_depth}, lr=0.08")
    console.print(f"    Features: {', '.join(LOCAL_FEATURES)}")

    # Train point estimate model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Compute metrics on ALL splits ---
    train_metrics = _compute_split_metrics("Train", model, X_train, y_train)
    val_metrics = _compute_split_metrics("Validation", model, X_val, y_val)
    test_metrics = _compute_split_metrics("Test", model, X_test, y_test)

    console.print(f"\n  [bold]Model Performance:[/bold]")
    for m in [train_metrics, val_metrics, test_metrics]:
        console.print(
            f"    {m.name:12s} | MAE: ${m.mae:>10,.0f} | "
            f"MAPE: {m.mape:>6.1%} | R²: {m.r2:>6.3f} | "
            f"Median Error: ${m.median_error:>10,.0f}"
        )

    # Cross-validation on training data for additional robustness check
    n_folds = min(5, max(2, n_train // 10))
    cv_scores_mae = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="neg_mean_absolute_error")
    cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=n_folds, scoring="r2")
    cv_mae = -cv_scores_mae.mean()
    cv_r2 = cv_scores_r2.mean()
    console.print(f"    {'CV (train)':12s} | MAE: ${cv_mae:>10,.0f} | R²: {cv_r2:>6.3f} ({n_folds}-fold)")

    # --- Retrain on ALL data for final prediction ---
    console.print(f"\n  [bold]Retraining on all {len(X)} samples for final prediction...[/bold]")
    model_final = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model_final.fit(X, y)

    # Quantile models for confidence interval (trained on all data)
    model_low = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        objective="reg:quantileerror",
        quantile_alpha=0.10,
        n_jobs=-1,
    )
    model_low.fit(X, y)

    model_high = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        objective="reg:quantileerror",
        quantile_alpha=0.90,
        n_jobs=-1,
    )
    model_high.fit(X, y)

    # Predict subject property
    X_subject = _subject_to_row(features, neighborhood_ppsf)
    point = float(model_final.predict(X_subject)[0])
    low = float(model_low.predict(X_subject)[0])
    high = float(model_high.predict(X_subject)[0])

    low, high = min(low, high), max(low, high)
    point = float(np.clip(point, low, high))
    low = max(low, 10_000)
    point = max(point, 10_000)
    high = max(high, point)

    # Feature importance from final model
    importances = dict(zip(LOCAL_FEATURES, model_final.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    console.print("\n  [bold]Feature Importance (final model):[/bold]")
    for feat, imp in top_features:
        bar = "█" * int(imp * 40)
        console.print(f"    {feat:30s} {imp:.3f} {bar}")

    return LocalEstimate(
        point_estimate=point,
        low=low,
        high=high,
        n_training_samples=len(df),
        cv_mae=cv_mae,
        cv_r2=cv_r2,
        median_price=float(df["price"].median()),
        median_ppsf=float(neighborhood_ppsf),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        data_summary=data_summary,
        feature_importances=importances,
    )
