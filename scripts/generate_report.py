#!/usr/bin/env python3
"""Generate a PDF valuation report for a subject property.

Usage:
    python scripts/generate_report.py

Edit the constants near the top of main() to set the subject address,
property features, and search parameters. Requires pdflatex on PATH.
The report is written to reports/<slug>_report.pdf.
"""

from __future__ import annotations

import asyncio
import csv
import subprocess
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from housing_estimator.estimator import geocode_address
from housing_estimator.datasources.recent_sales import fetch_recent_sales_redfin
from housing_estimator.features.property import PropertyFeatures, PropertyType
from housing_estimator.models.local_model import train_and_predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


# ── Formatting helpers ────────────────────────────────────────────────────────

def money(value: float | None) -> str:
    if value is None:
        return "--"
    return rf"\${value:,.0f}"


def number(value: float | int | None, digits: int = 0) -> str:
    if value is None:
        return "--"
    return f"{value:,.{digits}f}"


def pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.1%}".replace("%", r"\%")


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def zillow_url(address: str) -> str:
    slug = " ".join(address.replace(",", " ").split())
    return f"https://www.zillow.com/homes/{quote(slug.replace(' ', '-'))}_rb/"


def sale_date_sort_key(sale_date: str) -> pd.Timestamp:
    parsed = pd.to_datetime(sale_date, errors="coerce")
    return parsed if pd.notna(parsed) else pd.Timestamp.min


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_rows(rows: list[dict], include_link: bool = True) -> str:
    out = []
    for i, row in enumerate(rows, 1):
        address = str(row["address"])
        addr_cell = (
            rf"\href{{{zillow_url(address)}}}{{{tex_escape(address)}}}"
            if include_link else tex_escape(address)
        )
        out.append(
            " & ".join([
                str(i),
                addr_cell,
                tex_escape(row["sale_date"]),
                money(float(row["price"])),
                number(row["bedrooms"]),
                number(row["bathrooms"], 1),
                number(row["sqft"]),
                number(row["year_built"]),
                number(row["distance_miles"], 2),
                money(float(row["price_per_sqft"])) if row["price_per_sqft"] else "--",
            ])
            + r" \\"
        )
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Configure subject property here ──────────────────────────────────────
    ADDRESS = "77 Massachusetts Ave, Cambridge, MA 02139"
    GEOCODE_ADDRESS = ADDRESS
    SLUG = "example_report"

    FEATURES = dict(
        bedrooms=3,
        bathrooms=2.0,
        sqft=1_500,
        lot_sqft=None,
        year_built=1950,
        property_type=PropertyType.SINGLE_FAMILY,
    )

    SEARCH_RADIUS_MILES = 0.5
    SEARCH_DAYS_BACK = 365
    # ─────────────────────────────────────────────────────────────────────────

    REPORTS_DIR.mkdir(exist_ok=True)

    geo = asyncio.run(geocode_address(GEOCODE_ADDRESS))
    if geo is None:
        raise RuntimeError(f"Could not geocode {GEOCODE_ADDRESS}")

    features = PropertyFeatures(
        **FEATURES,
        latitude=geo.lat,
        longitude=geo.lon,
        zip_code=geo.zip_code,
    )

    sales = [
        sale
        for sale in fetch_recent_sales_redfin(
            geo.lat, geo.lon,
            radius_miles=SEARCH_RADIUS_MILES,
            days_back=SEARCH_DAYS_BACK,
        )
        if sale.sqft and sale.sqft > 0 and sale.price > 0
    ]

    estimate = train_and_predict(features, sales, SEARCH_RADIUS_MILES, SEARCH_DAYS_BACK)
    if estimate is None:
        raise RuntimeError("Local model could not produce an estimate — need more nearby sales.")

    rows: list[dict] = [
        {
            "address": s.address,
            "sale_date": s.sale_date or "",
            "price": s.price,
            "bedrooms": s.bedrooms,
            "bathrooms": s.bathrooms,
            "sqft": s.sqft,
            "year_built": s.year_built,
            "distance_miles": s.distance_miles,
            "price_per_sqft": s.price_per_sqft,
            "redfin_url": s.url or "",
            "zillow_url": zillow_url(s.address),
        }
        for s in sorted(sales, key=lambda s: s.distance_miles or 999)
    ]

    csv_path = REPORTS_DIR / f"{SLUG}_data.csv"
    write_csv(csv_path, rows)

    nearest = rows[:15]
    recent = sorted(rows, key=lambda r: sale_date_sort_key(str(r["sale_date"])), reverse=True)[:15]
    ds = estimate.data_summary
    assert ds is not None

    tex_path = REPORTS_DIR / f"{SLUG}.tex"
    pdf_path = REPORTS_DIR / f"{SLUG}.pdf"

    tex = rf"""
\documentclass[9pt]{{article}}
\usepackage[margin=0.55in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{titlesec}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.12}}
\titleformat{{\section}}{{\large\bfseries}}{{}}{{0em}}{{}}

\begin{{document}}

{{\LARGE \textbf{{Residential Valuation Report}}}}\\
{{\large {tex_escape(ADDRESS)}}}\\
Generated {date.today().isoformat()}

\section*{{Property Information}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Address & {tex_escape(ADDRESS)} \\
Matched geocode & {tex_escape(geo.matched_address)} \\
Coordinates & {geo.lat:.6f}, {geo.lon:.6f} \\
ZIP & {tex_escape(geo.zip_code)} \\
Property type & {tex_escape(features.property_type.value.replace("_", " ").title())} \\
Bedrooms & {features.bedrooms} \\
Bathrooms & {features.bathrooms:.1f} \\
Square feet & {features.sqft:,.0f} \\
Lot square feet & {f"{features.lot_sqft:,.0f}" if features.lot_sqft else "Not available"} \\
Year built & {features.year_built} \\
\bottomrule
\end{{tabular}}

\section*{{Estimated Price Range}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Point estimate & {money(estimate.point_estimate)} \\
Low estimate & {money(estimate.low)} \\
High estimate & {money(estimate.high)} \\
Median sale price (training data) & {money(estimate.median_price)} \\
Median price per sqft (training data) & {money(estimate.median_ppsf)} \\
\bottomrule
\end{{tabular}}

\section*{{Data Used by the Model}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Data source & Redfin public sold-property GIS CSV endpoint \\
Raw sales returned & {ds.total_sales} \\
Usable model rows & {ds.usable_sales} \\
Search radius & {ds.search_radius_miles:.1f} miles \\
Lookback period & {ds.search_days_back} days \\
Distance range & {tex_escape(ds.distance_range)} \\
Price range & {money(ds.price_min)}--{money(ds.price_max)} \\
Square-footage range & {number(ds.sqft_min)}--{number(ds.sqft_max)} sqft \\
Bedroom range & {tex_escape(ds.bed_range)} \\
Bathroom range & {tex_escape(ds.bath_range)} \\
Year-built range & {tex_escape(ds.year_built_range)} \\
CSV export & \texttt{{reports/{SLUG}\_data.csv}} \\
\bottomrule
\end{{tabular}}

\section*{{Model Configuration and Validation}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Model & XGBoost local model trained on nearby sold properties \\
Features & sqft, log\_sqft, bedrooms, bathrooms, age, lot\_sqft, distance\_miles, neighborhood median price/sqft \\
Train/validation/test split & 70\% / 15\% / 15\% \\
Final model & Retrained on all usable rows after validation \\
Cross-validation MAE & {money(estimate.cv_mae)} \\
Cross-validation R\textsuperscript{{2}} & {estimate.cv_r2:.3f} \\
Validation MAE & {money(estimate.val_metrics.mae if estimate.val_metrics else None)} \\
Validation MAPE & {pct(estimate.val_metrics.mape if estimate.val_metrics else None)} \\
Test MAE & {money(estimate.test_metrics.mae if estimate.test_metrics else None)} \\
Test MAPE & {pct(estimate.test_metrics.mape if estimate.test_metrics else None)} \\
\bottomrule
\end{{tabular}}

\section*{{Nearest Sold Properties}}
Each address links to a generated Zillow search URL for that sold property.
\scriptsize
\begin{{longtable}}{{r p{{2.45in}} p{{0.75in}} r r r r r r r}}
\toprule
\# & Address & Sold & Price & Bed & Bath & Sqft & Year & Mi & \$/Sqft \\
\midrule
\endhead
{table_rows(nearest)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Most Recently Sold}}
\scriptsize
\begin{{longtable}}{{r p{{2.45in}} p{{0.75in}} r r r r r r r}}
\toprule
\# & Address & Sold & Price & Bed & Bath & Sqft & Year & Mi & \$/Sqft \\
\midrule
\endhead
{table_rows(recent)}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Appendix: Full Dataset}}
All {len(rows)} usable sold-property rows, sorted by distance from subject.
\scriptsize
\begin{{longtable}}{{r p{{2.45in}} p{{0.75in}} r r r r r r r}}
\toprule
\# & Address & Sold & Price & Bed & Bath & Sqft & Year & Mi & \$/Sqft \\
\midrule
\endhead
{table_rows(rows)}
\bottomrule
\end{{longtable}}
\normalsize

\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")

    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=REPORTS_DIR,
            check=True,
        )

    print(f"PDF:  {pdf_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
