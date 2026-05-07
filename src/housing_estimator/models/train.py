"""Training pipeline for the XGBoost price estimation model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from housing_estimator.config import settings
from housing_estimator.datasources.census_acs import load_acs_data
from housing_estimator.datasources.fhfa import load_fhfa_data, get_hpi_at_year
from housing_estimator.datasources.redfin_bulk import load_redfin_data
from housing_estimator.features.engineering import (
    FEATURE_COLUMNS,
    engineer_training_features,
    adjust_price_by_hpi,
    CURRENT_YEAR,
)

console = Console()


def _load_king_county() -> pd.DataFrame:
    """Load King County housing dataset."""
    path = settings.data.raw_path / "kc_house_data.csv"
    if not path.exists():
        console.print(f"[red]King County data not found at {path}[/red]")
        console.print("Run 'housing-estimate setup' first.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Standardize columns
    col_map = {
        "price": "price",
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "sqft_living": "sqft",
        "sqft_lot": "lot_sqft",
        "floors": "stories",
        "yr_built": "year_built",
        "lat": "latitude",
        "long": "longitude",
        "zipcode": "zip_code",
        "date": "sale_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["property_type"] = "single_family"
    df["dataset"] = "king_county"

    # Extract sale year from date
    if "sale_date" in df.columns:
        df["sale_year"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.year
    else:
        df["sale_year"] = 2015  # KC dataset is ~2014-2015

    return df


def _load_ames() -> pd.DataFrame:
    """Load Ames, Iowa housing dataset."""
    path = settings.data.raw_path / "ames_housing.csv"
    if not path.exists():
        console.print(f"[yellow]Ames data not found at {path}, skipping.[/yellow]")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Ames dataset column mapping
    col_map = {
        "SalePrice": "price",
        "Gr Liv Area": "sqft",
        "GrLivArea": "sqft",
        "Lot Area": "lot_sqft",
        "LotArea": "lot_sqft",
        "Year Built": "year_built",
        "YearBuilt": "year_built",
        "Bedroom AbvGr": "bedrooms",
        "BedroomAbvGr": "bedrooms",
        "Full Bath": "full_bath",
        "FullBath": "full_bath",
        "Half Bath": "half_bath",
        "HalfBath": "half_bath",
        "Yr Sold": "sale_year",
        "YrSold": "sale_year",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Combine bathrooms
    full = df.get("full_bath", pd.Series(0, index=df.index))
    half = df.get("half_bath", pd.Series(0, index=df.index))
    df["bathrooms"] = full + 0.5 * half

    if "bedrooms" not in df.columns:
        df["bedrooms"] = 3

    df["stories"] = 1.0
    df["property_type"] = "single_family"
    df["dataset"] = "ames"

    # Ames, Iowa approximate coordinates
    df["latitude"] = 42.034
    df["longitude"] = -93.620
    df["zip_code"] = "50010"

    if "sale_year" not in df.columns:
        df["sale_year"] = 2010

    return df


def _enrich_with_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add market context features from FHFA, Redfin, and Census ACS."""
    acs = load_acs_data()
    redfin = load_redfin_data()

    # Merge median income
    if not acs.empty and "zip_code" in df.columns and "zip_code" in acs.columns:
        df = df.merge(
            acs[["zip_code", "median_income"]].drop_duplicates("zip_code"),
            on="zip_code",
            how="left",
        )
        df = df.rename(columns={"median_income": "zip_median_income"})

    # Merge Redfin price/sqft
    if not redfin.empty and "zip_code" in df.columns and "median_price_per_sqft" in redfin.columns:
        df = df.merge(
            redfin[["zip_code", "median_price_per_sqft"]].drop_duplicates("zip_code"),
            on="zip_code",
            how="left",
        )
        df = df.rename(columns={"median_price_per_sqft": "zip_median_price_per_sqft"})

    # FHFA HPI
    fhfa = load_fhfa_data()
    if not fhfa.empty and "zip_code" in df.columns:
        # Get latest HPI for each zip_code
        latest_hpi = fhfa.sort_values("year").groupby("zip_code").last().reset_index()
        df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
        df = df.merge(
            latest_hpi[["zip_code", "hpi"]].rename(columns={"hpi": "zip_hpi_current"}),
            on="zip_code",
            how="left",
        )

    if "zip_hpi_1yr_change" not in df.columns:
        df["zip_hpi_1yr_change"] = 0.0

    return df


def _adjust_prices_to_current(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust historical sale prices to current dollars using FHFA HPI."""
    fhfa = load_fhfa_data()
    if fhfa.empty or "zip_code" not in df.columns or "sale_year" not in df.columns:
        return df

    adjusted = df.copy()
    for idx, row in adjusted.iterrows():
        zip_code = str(row.get("zip_code", ""))
        sale_year = int(row.get("sale_year", CURRENT_YEAR))
        if zip_code and sale_year < CURRENT_YEAR:
            hpi_sale = get_hpi_at_year(zip_code, sale_year)
            hpi_now = get_hpi_at_year(zip_code, CURRENT_YEAR)
            if hpi_now is None:
                # Use latest available year
                hpi_now = get_hpi_at_year(zip_code, CURRENT_YEAR - 1)
            if hpi_sale and hpi_now:
                adjusted.at[idx, "price"] = adjust_price_by_hpi(
                    row["price"], hpi_sale, hpi_now
                )

    return adjusted


def train_model() -> dict[str, float]:
    """Train the XGBoost price estimation model.

    Returns a dict of evaluation metrics.
    """
    console.print("[bold]Loading training data...[/bold]")
    kc = _load_king_county()
    ames = _load_ames()

    dfs = [d for d in [kc, ames] if not d.empty]
    if not dfs:
        console.print("[red]No training data found. Run 'housing-estimate setup' first.[/red]")
        return {}

    df = pd.concat(dfs, ignore_index=True)
    console.print(f"  Loaded {len(df):,} records")

    # Adjust historical prices
    console.print("[bold]Adjusting prices to current dollars...[/bold]")
    df = _adjust_prices_to_current(df)

    # Enrich with market data
    console.print("[bold]Enriching with market context...[/bold]")
    df = _enrich_with_market_data(df)

    # Filter out invalid rows
    df = df.dropna(subset=["price", "sqft", "bedrooms", "bathrooms", "year_built"])
    df = df[(df["price"] > 10_000) & (df["sqft"] > 100)]
    console.print(f"  {len(df):,} records after filtering")

    # Feature engineering
    console.print("[bold]Engineering features...[/bold]")
    X = engineer_training_features(df)
    y = df["price"].values

    # Geographic-stratified split: use dataset source as stratification key
    stratify_col = df["dataset"] if "dataset" in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.model.test_size,
        random_state=settings.model.random_state,
        stratify=stratify_col,
    )
    console.print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train main point-estimate model
    console.print("[bold]Training XGBoost regressor...[/bold]")
    model = XGBRegressor(
        n_estimators=settings.model.n_estimators,
        max_depth=settings.model.max_depth,
        learning_rate=settings.model.learning_rate,
        random_state=settings.model.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Train quantile models for prediction interval
    console.print("[bold]Training quantile models for confidence interval...[/bold]")
    model_low = XGBRegressor(
        n_estimators=settings.model.n_estimators,
        max_depth=settings.model.max_depth,
        learning_rate=settings.model.learning_rate,
        random_state=settings.model.random_state,
        objective="reg:quantileerror",
        quantile_alpha=settings.model.quantile_low,
        n_jobs=-1,
    )
    model_low.fit(X_train, y_train, verbose=False)

    model_high = XGBRegressor(
        n_estimators=settings.model.n_estimators,
        max_depth=settings.model.max_depth,
        learning_rate=settings.model.learning_rate,
        random_state=settings.model.random_state,
        objective="reg:quantileerror",
        quantile_alpha=settings.model.quantile_high,
        n_jobs=-1,
    )
    model_high.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {"mae": mae, "mape": mape, "r2": r2}
    console.print(f"\n[bold green]Model Evaluation:[/bold green]")
    console.print(f"  MAE:  ${mae:,.0f}")
    console.print(f"  MAPE: {mape:.1f}%")
    console.print(f"  R²:   {r2:.4f}")

    # Save models
    models_dir = settings.data.models_path
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "xgb_point.joblib")
    joblib.dump(model_low, models_dir / "xgb_low.joblib")
    joblib.dump(model_high, models_dir / "xgb_high.joblib")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_columns.joblib")

    console.print(f"\n[bold]Models saved to {models_dir}[/bold]")

    # Also save training data for comps engine
    comp_cols = [
        "price", "bedrooms", "bathrooms", "sqft", "lot_sqft",
        "year_built", "stories", "latitude", "longitude",
        "zip_code", "property_type", "sale_year",
    ]
    existing = [c for c in comp_cols if c in df.columns]
    comp_data = df[existing].copy()
    comp_path = settings.data.processed_path
    comp_path.mkdir(parents=True, exist_ok=True)
    comp_data.to_parquet(comp_path / "training_data_for_comps.parquet", index=False)
    console.print(f"[bold]Comp data saved to {comp_path / 'training_data_for_comps.parquet'}[/bold]")

    return metrics
