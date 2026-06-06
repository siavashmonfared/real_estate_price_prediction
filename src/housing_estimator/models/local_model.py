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
from housing_estimator.features.property import Condition, PropertyFeatures

console = Console()

CURRENT_YEAR = datetime.now().year

# Features used by the local model
LOCAL_FEATURES = [
    "sqft",
    "log_sqft",
    "bedrooms",
    "bathrooms",
    "effective_age",
    "lot_sqft",
    "distance_miles",
    "price_per_sqft_neighborhood",
]

# Transparent, appraiser-style condition adjustment applied to the final
# estimate. A model feature can't learn condition because comparable sales
# carry no condition label, so we adjust the prediction directly instead.
_CONDITION_MULTIPLIER = {
    Condition.RENOVATED: 1.08,
    Condition.UPDATED: 1.03,
    Condition.AVERAGE: 1.00,
    Condition.DATED: 0.88,
}


def _effective_age(features: PropertyFeatures) -> float:
    """Effective age of the subject in years.

    Driven only by a *factual* renovation_year (a 2020-renovated home is
    genuinely newer). Condition quality is handled separately via
    _CONDITION_MULTIPLIER, not by distorting age. Comparable sales use actual
    age, so a subject with no renovation_year also uses actual age.
    """
    if features.renovation_year:
        return float(max(0, CURRENT_YEAR - features.renovation_year))
    return float(max(0, CURRENT_YEAR - features.year_built))


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
    # Comparable sales: condition unknown, so effective age == actual age.
    out["effective_age"] = CURRENT_YEAR - out["year_built"].clip(upper=CURRENT_YEAR)
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
        "effective_age": _effective_age(features),
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

    # Complexity scaled to sample size, and regularized to curb the
    # train-R²≈1.0 / low-test-R² overfitting seen on small local datasets.
    n_train = len(X_train)
    max_depth = 2 if n_train < 50 else 3 if n_train < 150 else 4
    min_child_weight = max(3, n_train // 30)
    reg_params = dict(
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=min_child_weight,
        reg_lambda=2.0,
        reg_alpha=0.5,
        gamma=0.5,
        random_state=42,
        n_jobs=-1,
    )

    console.print(f"\n  [bold]Model Config:[/bold]")
    console.print(
        f"    XGBoost: max_depth={max_depth}, lr=0.05, min_child_weight={min_child_weight}, "
        f"reg_lambda=2.0, reg_alpha=0.5, gamma=0.5"
    )
    console.print(f"    Features: {', '.join(LOCAL_FEATURES)}")

    # Pick the number of trees by early stopping on the validation split,
    # so the model stops before it memorizes the training data.
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=max_depth,
        early_stopping_rounds=25,
        eval_metric="mae",
        **reg_params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_n = int(getattr(model, "best_iteration", 0) or 0) + 1
    best_n = max(30, min(best_n, 400))
    console.print(f"    Early stopping selected n_estimators={best_n}")

    # --- Compute metrics on ALL splits (early-stopped model) ---
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

    # Cross-validation (fixed tree count, no early stopping so eval_set isn't needed)
    n_folds = min(5, max(2, n_train // 10))
    cv_model = XGBRegressor(n_estimators=best_n, max_depth=max_depth, **reg_params)
    cv_scores_mae = cross_val_score(cv_model, X_train, y_train, cv=n_folds, scoring="neg_mean_absolute_error")
    cv_scores_r2 = cross_val_score(cv_model, X_train, y_train, cv=n_folds, scoring="r2")
    cv_mae = -cv_scores_mae.mean()
    cv_r2 = cv_scores_r2.mean()
    console.print(f"    {'CV (train)':12s} | MAE: ${cv_mae:>10,.0f} | R²: {cv_r2:>6.3f} ({n_folds}-fold)")

    # --- Retrain on ALL data for final prediction (best_n trees) ---
    console.print(f"\n  [bold]Retraining on all {len(X)} samples ({best_n} trees) for final prediction...[/bold]")
    model_final = XGBRegressor(n_estimators=best_n, max_depth=max_depth, **reg_params)
    model_final.fit(X, y)

    # Quantile models for confidence interval (trained on all data)
    model_low = XGBRegressor(
        n_estimators=best_n, max_depth=max_depth,
        objective="reg:quantileerror", quantile_alpha=0.10, **reg_params,
    )
    model_low.fit(X, y)
    model_high = XGBRegressor(
        n_estimators=best_n, max_depth=max_depth,
        objective="reg:quantileerror", quantile_alpha=0.90, **reg_params,
    )
    model_high.fit(X, y)

    # Predict subject property
    X_subject = _subject_to_row(features, neighborhood_ppsf)
    point = float(model_final.predict(X_subject)[0])
    q_low = float(model_low.predict(X_subject)[0])
    q_high = float(model_high.predict(X_subject)[0])

    # Anchor the band AROUND the point estimate. Previously the point was
    # clipped into [q_low, q_high]; when the quantile models crossed (common
    # on small data) the band collapsed onto the point. Now the point is kept
    # as-is and the band is the union of the quantile predictions and a
    # minimum width derived from real held-out error (val/test MAPE).
    held_out_mape = max(
        (m.mape for m in (val_metrics, test_metrics) if m is not None),
        default=0.15,
    )
    held_out_mape = float(min(max(held_out_mape, 0.05), 0.40))
    low = min(q_low, q_high, point * (1 - held_out_mape))
    high = max(q_low, q_high, point * (1 + held_out_mape))

    # Transparent condition adjustment (shifts the whole distribution).
    cond_mult = _CONDITION_MULTIPLIER.get(features.condition, 1.0)
    if cond_mult != 1.0:
        console.print(
            f"\n  [bold]Condition adjustment:[/bold] {features.condition.value} "
            f"× {cond_mult:.2f}"
        )
    point *= cond_mult
    low *= cond_mult
    high *= cond_mult

    low = max(low, 10_000)
    point = max(point, 10_000)
    high = max(high, point * 1.01)

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
