# Real Estate Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Residential property price estimation using machine learning, geocoding, public market data, and comparable-sales analysis.

Two complementary approaches are combined:

- **Global model** — XGBoost trained on historical multi-city sales, enriched with FHFA House Price Index, Redfin ZIP-level market data, and Census ACS income. Good for a quick broad estimate.
- **Local model** — XGBoost trained *on demand* from Redfin sold listings within a user-specified radius of the subject property. Better for neighborhood-specific valuation. Reports full train/validation/test diagnostics so you know how much to trust the number.

The two estimates are blended based on the number of comparable sales found: more comps → higher comp weight, fewer comps → fall back to the global model.

All examples use the public institutional address `77 Massachusetts Ave, Cambridge, MA 02139`. For real use, replace the example address and property facts with verified subject-property information.

---

## Architecture

```
housing-estimate price / local-estimate / recent-sales
        │
        ├── geocoder/          Census geocoder → Nominatim fallback
        ├── datasources/       Zillow (property lookup), Redfin (recent sales),
        │                      FHFA HPI, Census ACS income, Redfin ZIP data
        ├── features/          PropertyFeatures schema + engineering transforms
        ├── models/
        │   ├── train.py       Global XGBoost training pipeline
        │   ├── predict.py     Global model inference
        │   └── local_model.py On-demand local model (train → validate → predict)
        ├── comps/             Comparable-sales engine (similarity scoring + price adjustment)
        └── estimator.py       Orchestration: geocode → enrich → ML → comps → blend
```

---

## Repository Layout

```text
.
├── config/
│   └── settings.yaml                 # Data paths, model hyperparameters, comp weights
├── data/
│   ├── models/                       # Pre-trained XGBoost artifacts (joblib)
│   ├── processed/                    # Processed comp training data (parquet)
│   └── raw/                          # Small reference datasets (kc_house_data.csv)
├── scripts/
│   ├── download_training_data.py     # Download King County & Ames training data
│   ├── download_market_data.py       # Download FHFA HPI, Redfin market data
│   ├── estimate_public_example.py    # CLI usage example (public address)
│   └── generate_report.py            # Generate a PDF valuation report via pdflatex
├── src/housing_estimator/
│   ├── cli.py                        # Typer CLI (price, local-estimate, recent-sales, train, setup)
│   ├── config.py                     # Pydantic settings loader
│   ├── estimator.py                  # Main orchestration pipeline
│   ├── output.py                     # Rich console rendering
│   ├── comps/engine.py               # Comparable-sales engine
│   ├── datasources/                  # Zillow, Redfin, FHFA, Census ACS loaders
│   ├── features/
│   │   ├── property.py               # PropertyFeatures schema, PropertyType, Condition
│   │   └── engineering.py            # Feature engineering transforms
│   ├── geocoder/                     # Census and Nominatim geocoders
│   └── models/
│       ├── train.py                  # Global XGBoost training pipeline
│       ├── predict.py                # Global model inference
│       └── local_model.py            # On-demand local model
└── tests/                            # Unit tests with mocked network calls
```

---

## Requirements

- Python 3.10 or newer
- Network access for live geocoding, Redfin recent-sales fetches, and optional Zillow lookup
- `pdflatex` (TeX Live or MacTeX) if you want to generate PDF reports via `generate_report.py`

Core dependencies are declared in `pyproject.toml`:

| Package | Purpose |
|---|---|
| `typer` + `click` | CLI |
| `rich` | Console rendering |
| `httpx` | HTTP client for geocoding and Redfin |
| `pydantic` + `pydantic-settings` | Config and schema validation |
| `xgboost` | Gradient-boosted tree models |
| `scikit-learn` | Train/test split, cross-validation |
| `pandas` + `numpy` + `pyarrow` | Data manipulation |
| `geopy` | Haversine distance |
| `joblib` | Model serialization |
| `pyyaml` | Config file loading |

---

## Installation

```bash
git clone https://github.com/siavashmonfared/real_estate_price_prediction.git
cd real_estate_price_prediction
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Verify:

```bash
housing-estimate --help
```

---

## Quick Start

### One-command estimate

```bash
housing-estimate price "77 Massachusetts Ave, Cambridge, MA 02139" --manual
```

The `price` command geocodes the address, optionally looks up public property details, prompts for property attributes, runs the pre-trained XGBoost model, finds comparable sales, and blends the estimates.

### Local model from recent nearby sales

```bash
housing-estimate local-estimate "77 Massachusetts Ave, Cambridge, MA 02139" --manual
```

Fetches Redfin sold listings near the address, trains an XGBoost model on them, and prints full model diagnostics alongside the estimate. Better for neighborhood-specific valuations.

### Recent sales lookup

```bash
housing-estimate recent-sales "77 Massachusetts Ave, Cambridge, MA 02139" --radius 1.0 --days 365
```

Geocodes the address and returns a formatted table of recently sold properties.

---

## CLI Reference

```
housing-estimate --help

Commands:
  setup           Download training data and market data files
  train           Train the global XGBoost price estimation model
  price           Estimate price using the global model + comparable sales
  local-estimate  Estimate price using a model trained on local recent sales
  recent-sales    Find recently sold properties near an address
```

Key options:

| Command | Option | Description |
|---|---|---|
| `price` | `--manual` / `-m` | Skip Zillow lookup, enter property details manually |
| `local-estimate` | `--manual` / `-m` | Skip Zillow lookup |
| `recent-sales` | `--days` / `-d` | Days to look back (default: 90) |
| `recent-sales` | `--radius` / `-r` | Search radius in miles (default: 1.0) |

---

## Programmatic Usage

### Global model estimate

```python
from housing_estimator.estimator import estimate_price
from housing_estimator.features.property import PropertyFeatures, PropertyType

features = PropertyFeatures(
    bedrooms=3,
    bathrooms=2.0,
    sqft=1_500,
    lot_sqft=5_000,
    year_built=1990,
    property_type=PropertyType.SINGLE_FAMILY,
)

result = estimate_price("77 Massachusetts Ave, Cambridge, MA 02139", features)
print(f"${result.blended_estimate:,.0f}")
print(f"Range: ${result.blended_low:,.0f} — ${result.blended_high:,.0f}")
```

### Local model from Redfin sales

```python
from housing_estimator.datasources.recent_sales import fetch_recent_sales_redfin
from housing_estimator.features.property import Condition, PropertyFeatures, PropertyType
from housing_estimator.models.local_model import train_and_predict

lat, lon = 42.359244, -71.093139

features = PropertyFeatures(
    bedrooms=5,
    bathrooms=3.0,
    sqft=1_995,
    lot_sqft=2_160,
    year_built=1894,
    condition=Condition.RENOVATED,       # transparent appraiser-style adjustment
    property_type=PropertyType.MULTI_FAMILY,
    latitude=lat,
    longitude=lon,
    zip_code="02139",
)

sales = fetch_recent_sales_redfin(lat, lon, radius_miles=1.0, days_back=365)
valid = [s for s in sales if s.sqft and s.sqft > 0 and s.price > 0]

estimate = train_and_predict(features, valid, search_radius=1.0, search_days=365)

print(f"${estimate.point_estimate:,.0f}")
print(f"Range: ${estimate.low:,.0f} — ${estimate.high:,.0f}")
print(f"Test R²: {estimate.test_metrics.r2:.3f}  MAPE: {estimate.test_metrics.mape:.1%}")
```

---

## Property Schema

`PropertyFeatures` (in `features/property.py`) captures all subject-property attributes:

| Field | Type | Description |
|---|---|---|
| `bedrooms` | `int` | Bedroom count |
| `bathrooms` | `float` | Bathroom count (0.5 = half bath) |
| `sqft` | `float` | Finished living area |
| `lot_sqft` | `float \| None` | Lot size |
| `year_built` | `int` | Original construction year |
| `renovation_year` | `int \| None` | If set, drives effective age instead of year_built |
| `condition` | `Condition` | `renovated` / `updated` / `average` / `dated` |
| `stories` | `float \| None` | Number of stories |
| `property_type` | `PropertyType` | `single_family` / `condo` / `townhouse` / `multi_family` / `other` |

The `Condition` field applies a transparent, appraiser-style multiplier to the final estimate:

| Condition | Multiplier | Use when |
|---|---|---|
| `RENOVATED` | ×1.08 | Recent gut renovation |
| `UPDATED` | ×1.03 | Partially updated, good shape |
| `AVERAGE` | ×1.00 | Typical for its age (default) |
| `DATED` | ×0.88 | Original/needs work |

---

## PDF Valuation Report

`scripts/generate_report.py` generates a formatted PDF with:

- Subject property details
- Estimated price range
- Model configuration and train/validation/test metrics
- Tables of nearest and most-recently sold properties
- Full comparable-sales dataset (CSV export)

Edit the constants near the top of `main()` to set the address, features, and search parameters:

```python
ADDRESS = "77 Massachusetts Ave, Cambridge, MA 02139"
SLUG = "example_report"              # output filename prefix
FEATURES = dict(bedrooms=3, bathrooms=2.0, sqft=1_500, year_built=1950, ...)
SEARCH_RADIUS_MILES = 0.5
SEARCH_DAYS_BACK = 365
```

Run:

```bash
python scripts/generate_report.py
```

Output:

```
reports/example_report.pdf
reports/example_report_data.csv
```

Requires `pdflatex` (TeX Live or MacTeX). The `reports/` directory is in `.gitignore`.

---

## Data Pipeline

### Download and train

```bash
housing-estimate setup    # downloads training and market data
housing-estimate train    # trains and saves model artifacts
```

Training sources:

| Dataset | Description |
|---|---|
| King County, WA (2014–2015) | ~21k residential sales with GPS coordinates |
| Ames, IA | Classic ML benchmark (~3k sales, when available) |
| FHFA HPI | Used to inflation-adjust historical prices to current dollars |
| Redfin ZIP data | ZIP-level median price/sqft for market context |
| Census ACS | ZIP-level median household income |

After training, the pipeline writes:

```text
data/models/xgb_point.joblib          # point-estimate model
data/models/xgb_low.joblib            # 10th-percentile quantile model
data/models/xgb_high.joblib           # 90th-percentile quantile model
data/models/feature_columns.joblib    # expected feature column order
data/processed/training_data_for_comps.parquet
```

### Blending logic

The `estimator.py` orchestrator blends the ML estimate with the comparable-sales engine based on how many comps were found:

| Comps found | Comp weight | ML weight |
|---|---|---|
| ≥ 5 | 60% | 40% |
| 3–4 | 40% | 60% |
| 1–2 | 20% | 80% |
| 0 | 0% | 100% |

These thresholds are configurable in `config/settings.yaml`.

---

## Configuration

`config/settings.yaml` controls all major parameters:

```yaml
geocoding:
  primary: census
  fallback: nominatim

model:
  n_estimators: 500
  max_depth: 6
  learning_rate: 0.05
  quantile_low: 0.10
  quantile_high: 0.90

comps:
  radii_miles: [0.5, 1.0, 2.0, 5.0]
  min_comps: 3
  max_comps: 10
  weights:
    sqft: 0.30
    bed_bath: 0.20
    age: 0.15
    property_type: 0.15
    recency: 0.20

blending:
  high_comp_threshold: 5
  mid_comp_threshold: 3
  weights:
    high_comp: [0.60, 0.40]   # [comp, ml]
    mid_comp: [0.40, 0.60]
    low_comp: [0.20, 0.80]
    no_comp: [0.00, 1.00]
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Most tests mock network calls. Live CLI commands that call Census, Nominatim, Zillow, or Redfin still require network access and may fail if a provider changes its response format or rate-limits automated requests.

---

## Troubleshooting

**`housing-estimate` not found** — make sure the virtual environment is activated and the package is installed (`pip install -e .`).

**Zillow lookup blocked** — Zillow returns captcha or rate-limit responses intermittently. Use `--manual` to enter property details directly.

**Redfin returns no sales** — try a larger radius or longer lookback:
```bash
housing-estimate recent-sales "ADDRESS" --radius 3.0 --days 730
```
The local model needs at least 20 usable sales with price and square footage.

**Model artifacts missing** — run `housing-estimate train` or verify these files exist:
```
data/models/xgb_point.joblib
data/models/xgb_low.joblib
data/models/xgb_high.joblib
data/models/feature_columns.joblib
```

---

## Privacy

The repository uses a public institutional address for all examples and does not include private property reports. If you adapt this for a real property, do not commit:

- Full home addresses or owner names
- Assessor account IDs or MLS identifiers
- County records, PDF exports, or private listing data
- Per-property valuation reports (`reports/` is in `.gitignore`)
- Absolute filesystem paths referencing personal directories

---

## Limitations

This is a research and decision-support tool, not a certified appraisal.

- Automated property details (from Zillow, Redfin) can be missing, stale, or incorrect.
- Web endpoints are public and may change format or block automated access without notice.
- Condition, renovation quality, school zones, lot utility, views, and micro-location effects are not captured by the features.
- Small local training sets can overfit even when test R² looks acceptable.
- Confidence intervals are approximate model ranges, not formal appraisal uncertainty.
- Final pricing decisions should be verified against assessor records, MLS listings, recent pending sales, and a human comparable-sales review.

---

## License

MIT
