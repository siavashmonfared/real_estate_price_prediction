"""Configuration loader using Pydantic settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import yaml
from pydantic import BaseModel


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where config/ lives)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "config" / "settings.yaml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Cannot locate config/settings.yaml from project root")


PROJECT_ROOT = _find_project_root()


class GeocodingConfig(BaseModel):
    primary: str = "census"
    fallback: str = "nominatim"
    nominatim_user_agent: str = "housing-estimator/0.1.0"


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "data/models"

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.raw_dir

    @property
    def processed_path(self) -> Path:
        return PROJECT_ROOT / self.processed_dir

    @property
    def models_path(self) -> Path:
        return PROJECT_ROOT / self.models_dir


class ModelConfig(BaseModel):
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    test_size: float = 0.2
    random_state: int = 42
    quantile_low: float = 0.10
    quantile_high: float = 0.90


class CompsConfig(BaseModel):
    radii_miles: list[float] = [0.5, 1.0, 2.0, 5.0]
    min_comps: int = 3
    max_comps: int = 10
    weights: dict[str, float] = {
        "sqft": 0.30,
        "bed_bath": 0.20,
        "age": 0.15,
        "property_type": 0.15,
        "recency": 0.20,
    }


class BlendingConfig(BaseModel):
    high_comp_threshold: int = 5
    mid_comp_threshold: int = 3
    weights: dict[str, list[float]] = {
        "high_comp": [0.60, 0.40],
        "mid_comp": [0.40, 0.60],
        "low_comp": [0.20, 0.80],
        "no_comp": [0.00, 1.00],
    }


class ApiConfig(BaseModel):
    rapidapi_key: str = ""
    zillow_host: str = "zillow-com1.p.rapidapi.com"


class Settings(BaseModel):
    geocoding: GeocodingConfig = GeocodingConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    comps: CompsConfig = CompsConfig()
    blending: BlendingConfig = BlendingConfig()
    api: ApiConfig = ApiConfig()


def load_settings() -> Settings:
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.exists():
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    else:
        raw = {}

    # Allow env var override for API key (don't store secrets in yaml)
    s = Settings(**raw)
    env_key = os.environ.get("RAPIDAPI_KEY", "")
    if env_key:
        s.api.rapidapi_key = env_key
    return s


settings = load_settings()
