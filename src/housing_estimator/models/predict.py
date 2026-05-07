"""Prediction module — load trained models and generate estimates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from housing_estimator.config import settings
from housing_estimator.features.engineering import features_to_dataframe
from housing_estimator.features.property import PropertyFeatures


@dataclass
class PredictionResult:
    point_estimate: float
    low: float
    high: float


def _models_dir() -> Path:
    return settings.data.models_path


def _load_model(name: str):
    path = _models_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'housing-estimate train' first."
        )
    return joblib.load(path)


def predict(features: PropertyFeatures) -> PredictionResult:
    """Generate a price prediction with confidence interval."""
    model_point = _load_model("xgb_point.joblib")
    model_low = _load_model("xgb_low.joblib")
    model_high = _load_model("xgb_high.joblib")

    X = features_to_dataframe(features)

    point = float(model_point.predict(X)[0])
    low = float(model_low.predict(X)[0])
    high = float(model_high.predict(X)[0])

    # Ensure sensible ordering
    low, high = min(low, high), max(low, high)
    point = np.clip(point, low, high)

    # Floor at $10k
    point = max(point, 10_000)
    low = max(low, 10_000)
    high = max(high, low)

    return PredictionResult(point_estimate=point, low=low, high=high)
