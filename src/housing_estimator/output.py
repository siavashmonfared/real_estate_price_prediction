"""Rich console output rendering for estimation results."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from housing_estimator.datasources.recent_sales import SoldProperty
from housing_estimator.estimator import EstimateResult
from housing_estimator.models.local_model import CURRENT_YEAR, LocalEstimate, _effective_age

console = Console()


def render_estimate(result: EstimateResult) -> None:
    """Render the full estimation result to the console."""
    console.print()

    # Header
    console.rule("[bold blue]Property Price Estimate[/bold blue]")
    console.print()

    # Address
    if result.geocoding:
        console.print(f"  [bold]Address:[/bold] {result.geocoding.matched_address}")
        console.print(f"  [bold]ZIP:[/bold] {result.geocoding.zip_code}")
        console.print(
            f"  [bold]Coordinates:[/bold] {result.geocoding.lat:.6f}, "
            f"{result.geocoding.lon:.6f}"
        )
    else:
        console.print(f"  [bold]Address:[/bold] {result.address}")
    console.print()

    # Property details
    if result.features:
        f = result.features
        console.print(f"  [bold]Property:[/bold] {f.bedrooms} bed / {f.bathrooms} bath / "
                       f"{f.sqft:,.0f} sqft / Built {f.year_built}")
        if f.lot_sqft:
            console.print(f"  [bold]Lot:[/bold] {f.lot_sqft:,.0f} sqft")
        console.print(f"  [bold]Type:[/bold] {f.property_type.value.replace('_', ' ').title()}")
        console.print()

    # Main estimate
    if result.blended_estimate:
        estimate_text = f"${result.blended_estimate:,.0f}"
        range_text = ""
        if result.blended_low and result.blended_high:
            range_text = (
                f"\n  Range: ${result.blended_low:,.0f} — ${result.blended_high:,.0f}"
            )

        panel = Panel(
            Text.from_markup(
                f"[bold green]{estimate_text}[/bold green]{range_text}"
            ),
            title="[bold]Estimated Value[/bold]",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)
    else:
        console.print("[red]Could not generate an estimate.[/red]")
        if result.errors:
            for err in result.errors:
                console.print(f"  [red]- {err}[/red]")
        return

    # Blending breakdown
    console.print()
    console.print("[bold]Estimate Breakdown:[/bold]")
    breakdown = Table(show_header=True, header_style="bold")
    breakdown.add_column("Method")
    breakdown.add_column("Estimate", justify="right")
    breakdown.add_column("Weight", justify="right")

    if result.ml_prediction:
        breakdown.add_row(
            "ML Model (XGBoost)",
            f"${result.ml_prediction.point_estimate:,.0f}",
            f"{result.ml_weight:.0%}",
        )

    if result.comps_result and result.comps_result.comp_estimate:
        breakdown.add_row(
            f"Comparable Sales ({len(result.comps_result.comps)} comps)",
            f"${result.comps_result.comp_estimate:,.0f}",
            f"{result.comp_weight:.0%}",
        )

    console.print(breakdown)

    # Market context
    if result.features:
        f = result.features
        ctx_items = []
        if f.zip_median_income:
            ctx_items.append(f"Median Income: ${f.zip_median_income:,.0f}")
        if f.zip_hpi_current:
            ctx_items.append(f"HPI: {f.zip_hpi_current:.1f}")
        if f.zip_hpi_1yr_change is not None:
            ctx_items.append(f"HPI 1yr Change: {f.zip_hpi_1yr_change:+.1%}")
        if f.zip_median_price_per_sqft:
            ctx_items.append(f"Median $/sqft: ${f.zip_median_price_per_sqft:,.0f}")

        if ctx_items:
            console.print()
            console.print("[bold]Market Context:[/bold]")
            for item in ctx_items:
                console.print(f"  {item}")

    # Comparable sales table
    if result.comps_result and result.comps_result.comps:
        console.print()
        console.print("[bold]Comparable Sales:[/bold]")
        comps_table = Table(show_header=True, header_style="bold")
        comps_table.add_column("#", justify="right")
        comps_table.add_column("Price", justify="right")
        comps_table.add_column("Adj. Price", justify="right")
        comps_table.add_column("Beds")
        comps_table.add_column("Baths")
        comps_table.add_column("Sqft", justify="right")
        comps_table.add_column("Year")
        comps_table.add_column("Dist (mi)", justify="right")
        comps_table.add_column("Score", justify="right")

        for i, comp in enumerate(result.comps_result.comps, 1):
            comps_table.add_row(
                str(i),
                f"${comp.price:,.0f}",
                f"${comp.adjusted_price:,.0f}",
                str(comp.bedrooms),
                f"{comp.bathrooms:.1f}",
                f"{comp.sqft:,.0f}",
                str(comp.year_built),
                f"{comp.distance_miles:.1f}",
                f"{comp.similarity_score:.2f}",
            )

        console.print(comps_table)

    console.print()

    # Errors/warnings
    if result.errors:
        console.print("[yellow]Warnings:[/yellow]")
        for err in result.errors:
            console.print(f"  [yellow]- {err}[/yellow]")
        console.print()


def render_recent_sales(
    address: str,
    sales: list[SoldProperty],
    radius_miles: float,
    days_back: int,
) -> None:
    """Render recent sales results to the console."""
    console.print()
    console.rule("[bold blue]Recent Sales[/bold blue]")
    console.print()
    console.print(f"  [bold]Near:[/bold] {address}")
    console.print(f"  [bold]Radius:[/bold] {radius_miles} miles")
    console.print(f"  [bold]Period:[/bold] Last {days_back} days")
    console.print(f"  [bold]Results:[/bold] {len(sales)} properties")
    console.print()

    if not sales:
        console.print("  [yellow]No recent sales found in this area.[/yellow]")
        console.print()
        return

    # Summary stats
    prices = [s.price for s in sales]
    sqft_prices = [s.price_per_sqft for s in sales if s.price_per_sqft]

    summary = Panel(
        Text.from_markup(
            f"[bold]Median:[/bold] ${np.median(prices):,.0f}    "
            f"[bold]Mean:[/bold] ${np.mean(prices):,.0f}    "
            f"[bold]Range:[/bold] ${min(prices):,.0f} — ${max(prices):,.0f}"
            + (f"\n[bold]Median $/sqft:[/bold] ${np.median(sqft_prices):,.0f}" if sqft_prices else "")
        ),
        title="[bold]Summary[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(summary)
    console.print()

    # Sales table
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Address", max_width=40)
    table.add_column("Price", justify="right")
    table.add_column("$/sqft", justify="right")
    table.add_column("Beds")
    table.add_column("Baths")
    table.add_column("Sqft", justify="right")
    table.add_column("Year")
    table.add_column("Sold", max_width=12)
    table.add_column("Dist (mi)", justify="right")

    for i, sale in enumerate(sales, 1):
        table.add_row(
            str(i),
            sale.address[:40] if sale.address else "",
            f"${sale.price:,.0f}",
            f"${sale.price_per_sqft:,.0f}" if sale.price_per_sqft else "-",
            str(sale.bedrooms) if sale.bedrooms else "-",
            f"{sale.bathrooms:.1f}" if sale.bathrooms else "-",
            f"{sale.sqft:,.0f}" if sale.sqft else "-",
            str(sale.year_built) if sale.year_built else "-",
            sale.sale_date or "-",
            f"{sale.distance_miles:.2f}" if sale.distance_miles is not None else "-",
        )

    console.print(table)
    console.print()


def render_local_estimate(
    address: str,
    features: "PropertyFeatures",
    estimate: LocalEstimate,
    closest_sales: list[SoldProperty],
) -> None:
    """Render a local-model estimation result with full transparency."""
    from housing_estimator.features.property import PropertyFeatures

    console.print()
    console.rule("[bold blue]Local Market Estimate[/bold blue]")
    console.print()

    # === SECTION 1: Subject Property ===
    console.print("[bold cyan]1. SUBJECT PROPERTY[/bold cyan]")
    console.print(f"  Address:    {address}")
    console.print(f"  Bedrooms:   {features.bedrooms}")
    console.print(f"  Bathrooms:  {features.bathrooms}")
    console.print(f"  Sqft:       {features.sqft:,.0f}")
    console.print(f"  Year Built: {features.year_built}")
    if features.lot_sqft:
        console.print(f"  Lot Sqft:   {features.lot_sqft:,.0f}")
    console.print()

    # === SECTION 2: Training Data Summary ===
    if estimate.data_summary:
        ds = estimate.data_summary
        console.print("[bold cyan]2. TRAINING DATA[/bold cyan]")
        console.print(f"  Source:           Redfin recently sold properties")
        console.print(f"  Search area:      {ds.search_radius_miles} mile radius, last {ds.search_days_back} days")
        console.print(f"  Total fetched:    {ds.total_sales}")
        console.print(f"  Usable (w/ sqft): {ds.usable_sales}")
        console.print()

        data_table = Table(show_header=True, header_style="bold", title="Training Data Distribution", box=None)
        data_table.add_column("Stat", style="bold")
        data_table.add_column("Min", justify="right")
        data_table.add_column("Median", justify="right")
        data_table.add_column("Max", justify="right")
        data_table.add_column("Mean", justify="right")

        data_table.add_row(
            "Sale Price",
            f"${ds.price_min:,.0f}",
            f"${ds.price_median:,.0f}",
            f"${ds.price_max:,.0f}",
            f"${ds.price_mean:,.0f}",
        )
        data_table.add_row(
            "Sqft",
            f"{ds.sqft_min:,.0f}",
            f"{ds.sqft_median:,.0f}",
            f"{ds.sqft_max:,.0f}",
            "",
        )
        data_table.add_row("Bedrooms", ds.bed_range, "", "", "")
        data_table.add_row("Bathrooms", ds.bath_range, "", "", "")
        data_table.add_row("Year Built", ds.year_built_range, "", "", "")
        data_table.add_row("Distance", ds.distance_range, "", "", "")

        console.print(data_table)
        console.print()

    # === SECTION 3: Model Performance ===
    console.print("[bold cyan]3. MODEL PERFORMANCE (Train / Validation / Test Split)[/bold cyan]")

    if estimate.train_metrics and estimate.val_metrics and estimate.test_metrics:
        perf_table = Table(show_header=True, header_style="bold", box=None)
        perf_table.add_column("Split", style="bold")
        perf_table.add_column("Samples", justify="right")
        perf_table.add_column("MAE ($)", justify="right")
        perf_table.add_column("MAPE", justify="right")
        perf_table.add_column("R²", justify="right")
        perf_table.add_column("Median Error ($)", justify="right")

        for m in [estimate.train_metrics, estimate.val_metrics, estimate.test_metrics]:
            # Color R² based on quality
            r2_color = "green" if m.r2 > 0.7 else "yellow" if m.r2 > 0.5 else "red"
            perf_table.add_row(
                m.name,
                str(m.n_samples),
                f"${m.mae:,.0f}",
                f"{m.mape:.1%}",
                f"[{r2_color}]{m.r2:.3f}[/{r2_color}]",
                f"${m.median_error:,.0f}",
            )

        console.print(perf_table)
        console.print()

        # Interpretation
        test_r2 = estimate.test_metrics.r2
        test_mape = estimate.test_metrics.mape
        train_r2 = estimate.train_metrics.r2

        if train_r2 - test_r2 > 0.15:
            console.print("  [yellow]Note: Gap between train and test R² suggests some overfitting.[/yellow]")
        if test_r2 > 0.7:
            console.print(f"  [green]Test R² of {test_r2:.3f} indicates good predictive power.[/green]")
        elif test_r2 > 0.5:
            console.print(f"  [yellow]Test R² of {test_r2:.3f} indicates moderate predictive power.[/yellow]")
        else:
            console.print(f"  [red]Test R² of {test_r2:.3f} indicates limited predictive power — treat estimate with caution.[/red]")

        console.print(f"  On average, predictions are off by {test_mape:.1%} (MAPE) on held-out test data.")
        console.print()

    # === SECTION 4: Feature Importance ===
    if estimate.feature_importances:
        console.print("[bold cyan]4. FEATURE IMPORTANCE[/bold cyan]")
        fi_table = Table(show_header=True, header_style="bold", box=None)
        fi_table.add_column("Feature", style="bold")
        fi_table.add_column("Importance", justify="right")
        fi_table.add_column("Bar")

        sorted_fi = sorted(estimate.feature_importances.items(), key=lambda x: x[1], reverse=True)
        max_imp = sorted_fi[0][1] if sorted_fi else 1
        for feat, imp in sorted_fi:
            bar_len = int((imp / max_imp) * 25)
            bar = "█" * bar_len
            fi_table.add_row(feat, f"{imp:.3f}", f"[cyan]{bar}[/cyan]")

        console.print(fi_table)
        console.print()

    # === SECTION 5: Final Estimate ===
    console.print("[bold cyan]5. ESTIMATE FOR SUBJECT PROPERTY[/bold cyan]")

    panel = Panel(
        Text.from_markup(
            f"[bold green]${estimate.point_estimate:,.0f}[/bold green]\n"
            f"  80% Confidence Range: ${estimate.low:,.0f} — ${estimate.high:,.0f}"
        ),
        title="[bold]Estimated Value[/bold]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)

    # Context
    console.print()
    ctx_table = Table(show_header=False, box=None, padding=(0, 2))
    ctx_table.add_column("Key", style="bold")
    ctx_table.add_column("Value")
    ctx_table.add_row("Neighborhood median price", f"${estimate.median_price:,.0f}")
    ctx_table.add_row("Neighborhood median $/sqft", f"${estimate.median_ppsf:,.0f}")
    ctx_table.add_row("Implied $/sqft for subject", f"${estimate.point_estimate / features.sqft:,.0f}")

    # Condition / effective age context (subject-specific, no hardcoded values)
    cond = getattr(features, "condition", None)
    if cond is not None:
        actual_age = max(0, CURRENT_YEAR - features.year_built)
        if getattr(features, "renovation_year", None):
            eff_age = max(0, CURRENT_YEAR - features.renovation_year)
            cond_label = f"renovated {features.renovation_year}"
        else:
            eff_age = _effective_age(features)
            cond_label = cond.value
        ctx_table.add_row("Condition (model input)", cond_label)
        ctx_table.add_row("Actual / effective age", f"{actual_age} yr / {eff_age:.0f} yr")
    console.print(ctx_table)

    # === SECTION 6: Closest Comparable Sales ===
    if closest_sales:
        console.print()
        console.print(f"[bold cyan]6. CLOSEST COMPARABLE SALES ({len(closest_sales)})[/bold cyan]")
        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Address", max_width=35)
        table.add_column("Price", justify="right")
        table.add_column("$/sqft", justify="right")
        table.add_column("Beds")
        table.add_column("Baths")
        table.add_column("Sqft", justify="right")
        table.add_column("Year")
        table.add_column("Sold")
        table.add_column("Dist", justify="right")

        for i, s in enumerate(closest_sales, 1):
            table.add_row(
                str(i),
                (s.address or "")[:35],
                f"${s.price:,.0f}",
                f"${s.price_per_sqft:,.0f}" if s.price_per_sqft else "-",
                str(s.bedrooms) if s.bedrooms else "-",
                f"{s.bathrooms:.1f}" if s.bathrooms else "-",
                f"{s.sqft:,.0f}" if s.sqft else "-",
                str(s.year_built) if s.year_built else "-",
                s.sale_date or "-",
                f"{s.distance_miles:.2f} mi" if s.distance_miles is not None else "-",
            )

        console.print(table)

    console.print()
