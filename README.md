# Real Estate Price Prediction

Residential property price estimation using machine learning, geocoding, public market data, and comparable-sales analysis.

This project packages a command-line estimator named `housing-estimate`. It can:

- Estimate a residential property price from an address and property attributes.
- Geocode addresses through the US Census geocoder with a Nominatim fallback.
- Attempt property-detail lookup from public Zillow pages when available.
- Fetch recent sold-property data near a target address from Redfin public endpoints.
- Train a local XGBoost model on nearby sales and report transparent diagnostics.
- Use pre-trained XGBoost models plus comparable-sales blending for a broader estimate.
- Render console reports with model performance, feature importances, estimate ranges, and closest comps.

The examples below use the public MIT main address, `77 Massachusetts Ave, Cambridge, MA 02139`, as a non-private demonstration address. For real residential use, replace the example address and property facts with verified subject-property information.

## Repository Layout

```text
.
├── config/
│   └── settings.yaml                 # Data paths, model settings, comp weights
├── data/
│   ├── models/                       # Pre-trained XGBoost model artifacts
│   ├── processed/                    # Processed comparable-sales training data
│   └── raw/                          # Small raw support datasets
├── scripts/
│   ├── download_training_data.py     # Downloads sample training data
│   ├── download_market_data.py       # Downloads optional market context data
│   └── estimate_public_example.py    # Safe public-address example script
├── src/housing_estimator/
│   ├── cli.py                        # Typer CLI entrypoint
│   ├── estimator.py                  # Main orchestration pipeline
│   ├── comps/                        # Comparable-sales engine
│   ├── datasources/                  # Zillow, Redfin, FHFA, Census ACS loaders
│   ├── features/                     # Property schema and feature engineering
│   ├── geocoder/                     # Census and Nominatim geocoders
│   └── models/                       # Training, prediction, and local model logic
└── tests/                            # Unit tests with mocked network calls
```

## What Is Included

The repository includes code, tests, config, pre-trained model artifacts, and small data assets needed for normal use:

- `data/models/xgb_point.joblib`
- `data/models/xgb_low.joblib`
- `data/models/xgb_high.joblib`
- `data/models/feature_columns.joblib`
- `data/processed/training_data_for_comps.parquet`
- `data/raw/kc_house_data.csv`

Large or region-specific raw market files are intentionally excluded and can be regenerated locally:

- `data/raw/redfin_zip_raw.tsv.gz`
- `data/raw/redfin_zip_market_data.csv`
- `data/raw/fhfa_hpi.csv`
- `data/raw/fhfa_hpi_zip5.xlsx`

Run `housing-estimate setup` or `python scripts/download_market_data.py` if you want those optional market-context files.

## Requirements

- Python 3.10 or newer
- Network access for live geocoding and Redfin recent-sales fetches
- A working Python virtual environment

Core dependencies are declared in `pyproject.toml`, including:

- `typer`
- `click`
- `rich`
- `httpx`
- `pydantic`
- `xgboost`
- `scikit-learn`
- `pandas`
- `numpy`
- `geopy`
- `joblib`
- `pyarrow`
- `pyyaml`

## Installation

Clone the repository:

```bash
git clone https://github.com/siavashmonfared/real_estate_price_prediction.git
cd real_estate_price_prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install --upgrade pip
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

Confirm the CLI is installed:

```bash
housing-estimate --help
```

## Quick Start

### Standard estimate

```bash
housing-estimate price "77 Massachusetts Ave, Cambridge, MA 02139" --manual
```

The `price` command:

1. Optionally tries to look up public property details.
2. Prompts for property details when `--manual` is used or lookup fails.
3. Geocodes the address.
4. Adds available market context.
5. Runs the pre-trained XGBoost model.
6. Finds comparable sales from the processed comp dataset.
7. Blends ML and comp estimates.

Manual mode prompts for:

- Bedrooms
- Bathrooms
- Square footage
- Lot square footage
- Year built
- Property type

### Local model from recent nearby sales

```bash
housing-estimate local-estimate "77 Massachusetts Ave, Cambridge, MA 02139" --manual
```

The `local-estimate` command:

1. Geocodes the address.
2. Gets or prompts for property details.
3. Fetches nearby Redfin sold listings.
4. Expands radius/lookback until enough training sales are found.
5. Trains an XGBoost model on local sales.
6. Reports train, validation, test, and cross-validation metrics.
7. Retrains on all local data and predicts the subject property.
8. Prints the closest comparable sales.

### Recent sales lookup

```bash
housing-estimate recent-sales "77 Massachusetts Ave, Cambridge, MA 02139" --radius 1.0 --days 365
```

This geocodes the address and fetches recently sold Redfin properties in the requested radius and lookback window.

## Example Script

Run the safe public-address example:

```bash
python scripts/estimate_public_example.py
```

The script uses `77 Massachusetts Ave, Cambridge, MA 02139` only as a public geocoding anchor and uses illustrative residential inputs. For a real estimate:

1. Copy `scripts/estimate_public_example.py`.
2. Change `ADDRESS`.
3. Replace `PropertyFeatures` with verified subject-property facts.
4. Adjust search radius, lookback, or target sales count if needed.
5. Run the copied script from the repo root.

## Programmatic Usage

### Standard model estimate

```python
from housing_estimator.estimator import estimate_price
from housing_estimator.features.property import PropertyFeatures, PropertyType

features = PropertyFeatures(
    bedrooms=3,
    bathrooms=2.0,
    sqft=1500,
    lot_sqft=5000,
    year_built=1990,
    property_type=PropertyType.SINGLE_FAMILY,
)

result = estimate_price("77 Massachusetts Ave, Cambridge, MA 02139", features)

print(result.blended_estimate)
print(result.blended_low, result.blended_high)
```

### Local model estimate from Redfin sales

```python
from housing_estimator.datasources.recent_sales import fetch_recent_sales_redfin
from housing_estimator.features.property import PropertyFeatures, PropertyType
from housing_estimator.models.local_model import train_and_predict

lat = 42.359244
lon = -71.093139

features = PropertyFeatures(
    bedrooms=3,
    bathrooms=2.0,
    sqft=1500,
    lot_sqft=5000,
    year_built=1990,
    property_type=PropertyType.SINGLE_FAMILY,
    latitude=lat,
    longitude=lon,
    zip_code="02139",
)

sales = fetch_recent_sales_redfin(lat, lon, radius_miles=1.0, days_back=365)
valid_sales = [s for s in sales if s.sqft and s.sqft > 0 and s.price > 0]

estimate = train_and_predict(
    features,
    valid_sales,
    search_radius=1.0,
    search_days=365,
)

print(estimate.point_estimate)
print(estimate.low, estimate.high)
```

## Data Pipeline

### Download data

```bash
housing-estimate setup
```

This runs:

```bash
python scripts/download_training_data.py
python scripts/download_market_data.py
```

The setup step can download:

- Sample historical home-sales training data
- FHFA HPI data
- Redfin ZIP-level market data
- A large Redfin raw ZIP archive, when available

Large generated raw data files are ignored by Git.

### Train the global model

```bash
housing-estimate train
```

Training reads `data/raw/kc_house_data.csv`, engineers features, and writes:

- `data/models/xgb_point.joblib`
- `data/models/xgb_low.joblib`
- `data/models/xgb_high.joblib`
- `data/models/feature_columns.joblib`
- `data/processed/training_data_for_comps.parquet`

The point model predicts expected sale price. The low/high models provide an approximate 10th/90th percentile range.

## Configuration

Primary settings live in `config/settings.yaml`.

Important sections:

- `data`: paths for raw data, processed data, and model artifacts
- `model`: global XGBoost training parameters
- `comps`: comparable-sale search radii, count limits, and similarity weights
- `blending`: how strongly to weight comps vs. ML based on comp count
- `geocoding`: Census primary geocoder and Nominatim fallback

Example comp weights:

```yaml
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
```

## Model Notes

### Global model

The global model is trained from historical housing data and engineered features. It is useful for a quick broad estimate, especially when paired with the comparable-sales engine.

### Local model

The local model is trained on Redfin sold listings around a specific subject property. It is usually better for neighborhood-specific valuation because it uses current nearby sales, but it depends on the availability and quality of recent sold listings.

Local model features:

- Square footage
- Log square footage
- Bedrooms
- Bathrooms
- Age
- Lot square footage
- Distance from subject property
- Neighborhood median price per square foot

The local model prints:

- Train/validation/test split sizes
- MAE
- MAPE
- R-squared
- Median error
- Cross-validation MAE/R-squared
- Feature importances
- Point estimate
- 80% confidence range
- Closest comparable sales

## Testing

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Most tests mock network calls. Live CLI commands that call Zillow, Census, Nominatim, or Redfin still require network access and may fail if a provider rate limits, changes response format, or blocks automated requests.

## Common Troubleshooting

### `housing-estimate` command not found

Install the package from the repo root:

```bash
pip install -e .
```

Make sure your virtual environment is activated.

### Typer or Click import error

Upgrade package dependencies:

```bash
pip install --upgrade -e .
```

This project declares `click>=8.1` because newer Typer versions depend on modern Click typing behavior.

### Zillow lookup is blocked

Zillow may return captcha or rate-limit responses. Use manual mode:

```bash
housing-estimate price "ADDRESS" --manual
housing-estimate local-estimate "ADDRESS" --manual
```

### Redfin returns no sales

Try a larger radius or longer lookback:

```bash
housing-estimate recent-sales "ADDRESS" --radius 3.0 --days 730
```

For local model training, the code needs at least 20 usable sales with price and square footage.

### Model artifacts missing

Run:

```bash
housing-estimate train
```

or confirm these files exist:

```text
data/models/xgb_point.joblib
data/models/xgb_low.joblib
data/models/xgb_high.joblib
data/models/feature_columns.joblib
```

## Privacy Notes

The repository intentionally uses a public institutional address for examples and does not include private property reports. If you adapt the project for a real property, avoid committing:

- Full home addresses
- Owner names
- Assessor account IDs
- MLS screenshots or private listing exports
- Local PDFs from county records
- Personal file paths
- One-off valuation reports for private properties

## Limitations

This is a research and decision-support estimator, not a certified appraisal.

Important caveats:

- Automated property details can be missing or stale.
- Redfin/Zillow endpoints are public web endpoints and can change or block automated access.
- Assessor data, renovation quality, condition, school zones, lot utility, and micro-location effects may not be captured.
- Small local training sets can overfit even when test R-squared looks strong.
- Confidence intervals are approximate model ranges, not formal appraisal uncertainty.
- Final pricing decisions should be checked against verified assessor records, MLS listings, recent pending sales, and a human comp review.
