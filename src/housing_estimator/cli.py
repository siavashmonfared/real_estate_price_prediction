"""CLI entry point using Typer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import FloatPrompt, IntPrompt, Prompt

from housing_estimator.config import PROJECT_ROOT

app = typer.Typer(
    name="housing-estimate",
    help="Estimate residential property prices using ML and comparable sales.",
)
console = Console()


@app.command()
def setup():
    """Download training data and market data files."""
    console.print("[bold]Setting up housing estimator...[/bold]\n")

    scripts_dir = PROJECT_ROOT / "scripts"

    console.print("[bold]Step 1: Downloading training data...[/bold]")
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "download_training_data.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        console.print("[red]Training data download failed.[/red]")
        raise typer.Exit(1)

    console.print("\n[bold]Step 2: Downloading market data...[/bold]")
    result = subprocess.run(
        [sys.executable, str(scripts_dir / "download_market_data.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        console.print("[yellow]Market data download had issues (non-critical).[/yellow]")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("Next: run [bold]housing-estimate train[/bold] to train the model.")


@app.command()
def train():
    """Train the XGBoost price estimation model."""
    from housing_estimator.models.train import train_model

    console.print("[bold]Training price estimation model...[/bold]\n")
    metrics = train_model()

    if not metrics:
        console.print("[red]Training failed. Check data availability.[/red]")
        raise typer.Exit(1)

    console.print("\n[bold green]Training complete![/bold green]")


@app.command()
def price(
    address: str = typer.Argument(help="Street address to estimate"),
    manual: bool = typer.Option(False, "--manual", "-m", help="Skip Zillow lookup, enter details manually"),
):
    """Estimate the price of a property at the given address."""
    from housing_estimator.datasources.zillow import lookup_property, zillow_type_to_property_type
    from housing_estimator.estimator import estimate_price
    from housing_estimator.features.property import PropertyFeatures, PropertyType
    from housing_estimator.output import render_estimate

    console.print(f"\n[bold]Estimating price for:[/bold] {address}\n")

    zillow_data = None
    if not manual:
        console.print("[bold]Looking up property details on Zillow...[/bold]")
        zillow_data = lookup_property(address)

    if zillow_data and zillow_data.sqft:
        console.print(f"  [green]Found property data![/green]")
        if zillow_data.bedrooms:
            console.print(f"    Bedrooms:  {zillow_data.bedrooms}")
        if zillow_data.bathrooms:
            console.print(f"    Bathrooms: {zillow_data.bathrooms}")
        console.print(f"    Sqft:      {zillow_data.sqft:,.0f}")
        if zillow_data.lot_sqft:
            console.print(f"    Lot sqft:  {zillow_data.lot_sqft:,.0f}")
        if zillow_data.year_built:
            console.print(f"    Year built: {zillow_data.year_built}")
        if zillow_data.property_type:
            console.print(f"    Type:      {zillow_data.property_type}")
        if zillow_data.zestimate:
            console.print(f"    Zestimate: ${zillow_data.zestimate:,.0f}")
        console.print()

        # Let user confirm or override
        use_zillow = Prompt.ask(
            "  Use these details? (y to accept, n to enter manually)",
            choices=["y", "n"],
            default="y",
        )
        if use_zillow == "y":
            ptype_str = zillow_type_to_property_type(zillow_data.property_type)
            features = PropertyFeatures(
                bedrooms=zillow_data.bedrooms or IntPrompt.ask("  Bedrooms", default=3),
                bathrooms=zillow_data.bathrooms or FloatPrompt.ask("  Bathrooms", default=2.0),
                sqft=zillow_data.sqft,
                lot_sqft=zillow_data.lot_sqft,
                year_built=zillow_data.year_built or IntPrompt.ask("  Year built", default=1990),
                property_type=PropertyType(ptype_str),
            )
            result = estimate_price(address, features)
            render_estimate(result)
            return

    if not manual:
        console.print("  [yellow]Could not fetch property details automatically.[/yellow]")
        console.print()

    # Manual input fallback
    console.print("[bold]Enter property details:[/bold]")
    bedrooms = IntPrompt.ask("  Bedrooms", default=3)
    bathrooms = FloatPrompt.ask("  Bathrooms", default=2.0)
    sqft = FloatPrompt.ask("  Square footage", default=1500.0)
    lot_sqft_str = Prompt.ask("  Lot sqft (press Enter to skip)", default="")
    lot_sqft = float(lot_sqft_str) if lot_sqft_str else None
    year_built = IntPrompt.ask("  Year built", default=1990)

    type_choices = {
        "1": PropertyType.SINGLE_FAMILY,
        "2": PropertyType.CONDO,
        "3": PropertyType.TOWNHOUSE,
        "4": PropertyType.MULTI_FAMILY,
        "5": PropertyType.OTHER,
    }
    console.print("  Property type:")
    for k, v in type_choices.items():
        console.print(f"    {k}. {v.value.replace('_', ' ').title()}")
    type_choice = Prompt.ask("  Select", choices=list(type_choices.keys()), default="1")
    property_type = type_choices[type_choice]

    features = PropertyFeatures(
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        sqft=sqft,
        lot_sqft=lot_sqft,
        year_built=year_built,
        property_type=property_type,
    )

    result = estimate_price(address, features)
    render_estimate(result)


@app.command(name="recent-sales")
def recent_sales(
    address: str = typer.Argument(help="Street address to search near"),
    days: int = typer.Option(90, "--days", "-d", help="Number of days to look back"),
    radius: float = typer.Option(1.0, "--radius", "-r", help="Search radius in miles"),
):
    """Find recently sold properties near an address."""
    import asyncio
    from housing_estimator.datasources.recent_sales import fetch_recent_sales_redfin
    from housing_estimator.estimator import geocode_address
    from housing_estimator.output import render_recent_sales

    console.print(f"\n[bold]Searching recent sales near:[/bold] {address}")
    console.print(f"  Radius: {radius} miles | Last {days} days\n")

    # Geocode
    console.print("[bold]Geocoding address...[/bold]")
    geo = asyncio.run(geocode_address(address))
    if not geo:
        console.print("[red]Could not geocode address.[/red]")
        raise typer.Exit(1)

    console.print(f"  Matched: {geo.matched_address}")
    console.print(f"  Lat/Lon: {geo.lat:.6f}, {geo.lon:.6f}\n")

    # Fetch sales
    console.print("[bold]Fetching recent sales from Redfin...[/bold]")
    sales = fetch_recent_sales_redfin(
        lat=geo.lat,
        lon=geo.lon,
        radius_miles=radius,
        days_back=days,
    )

    render_recent_sales(
        address=geo.matched_address,
        sales=sales,
        radius_miles=radius,
        days_back=days,
    )


@app.command(name="local-estimate")
def local_estimate(
    address: str = typer.Argument(help="Street address to estimate"),
    manual: bool = typer.Option(False, "--manual", "-m", help="Skip Zillow lookup, enter details manually"),
):
    """Estimate price using a model trained on local recent sales data."""
    import asyncio
    from housing_estimator.datasources.zillow import lookup_property, zillow_type_to_property_type
    from housing_estimator.estimator import geocode_address
    from housing_estimator.features.property import PropertyFeatures, PropertyType
    from housing_estimator.models.local_model import fetch_training_sales, train_and_predict
    from housing_estimator.output import render_local_estimate

    console.print(f"\n[bold]Local Market Estimate for:[/bold] {address}\n")

    # Step 1: Geocode
    console.print("[bold]Geocoding address...[/bold]")
    geo = asyncio.run(geocode_address(address))
    if not geo:
        console.print("[red]Could not geocode address.[/red]")
        raise typer.Exit(1)

    console.print(f"  Matched: {geo.matched_address}")
    console.print(f"  Lat/Lon: {geo.lat:.6f}, {geo.lon:.6f}\n")

    # Step 2: Get property details (Zillow or manual)
    zillow_data = None
    if not manual:
        console.print("[bold]Looking up property details on Zillow...[/bold]")
        zillow_data = lookup_property(address)

    features = None
    if zillow_data and zillow_data.sqft:
        console.print(f"  [green]Found property data![/green]")
        if zillow_data.bedrooms:
            console.print(f"    Bedrooms:  {zillow_data.bedrooms}")
        if zillow_data.bathrooms:
            console.print(f"    Bathrooms: {zillow_data.bathrooms}")
        console.print(f"    Sqft:      {zillow_data.sqft:,.0f}")
        if zillow_data.lot_sqft:
            console.print(f"    Lot sqft:  {zillow_data.lot_sqft:,.0f}")
        if zillow_data.year_built:
            console.print(f"    Year built: {zillow_data.year_built}")
        console.print()

        use_zillow = Prompt.ask(
            "  Use these details? (y/n)",
            choices=["y", "n"],
            default="y",
        )
        if use_zillow == "y":
            ptype_str = zillow_type_to_property_type(zillow_data.property_type)
            features = PropertyFeatures(
                bedrooms=zillow_data.bedrooms or IntPrompt.ask("  Bedrooms", default=3),
                bathrooms=zillow_data.bathrooms or FloatPrompt.ask("  Bathrooms", default=2.0),
                sqft=zillow_data.sqft,
                lot_sqft=zillow_data.lot_sqft,
                year_built=zillow_data.year_built or IntPrompt.ask("  Year built", default=1990),
                property_type=PropertyType(ptype_str),
                latitude=geo.lat,
                longitude=geo.lon,
                zip_code=geo.zip_code,
            )

    if features is None:
        if not manual:
            console.print("  [yellow]Could not fetch property details automatically.[/yellow]")
        console.print("\n[bold]Enter property details:[/bold]")
        bedrooms = IntPrompt.ask("  Bedrooms", default=3)
        bathrooms = FloatPrompt.ask("  Bathrooms", default=2.0)
        sqft = FloatPrompt.ask("  Square footage", default=1500.0)
        lot_sqft_str = Prompt.ask("  Lot sqft (press Enter to skip)", default="")
        lot_sqft = float(lot_sqft_str) if lot_sqft_str else None
        year_built = IntPrompt.ask("  Year built", default=1990)
        features = PropertyFeatures(
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            sqft=sqft,
            lot_sqft=lot_sqft,
            year_built=year_built,
            latitude=geo.lat,
            longitude=geo.lon,
            zip_code=geo.zip_code,
        )

    # Step 3: Fetch local sales for training
    console.print("\n[bold]Fetching nearby sales for local model training...[/bold]")
    sales, search_radius, search_days = fetch_training_sales(lat=geo.lat, lon=geo.lon, target_count=200)

    if len(sales) < 20:
        console.print("[red]Not enough sales data to build a local model (need ≥20).[/red]")
        raise typer.Exit(1)

    # Step 4: Train and predict
    console.print(f"\n[bold]Training local model on {len(sales)} sales...[/bold]")
    estimate = train_and_predict(features, sales, search_radius, search_days)

    if estimate is None:
        console.print("[red]Could not generate estimate.[/red]")
        raise typer.Exit(1)

    # Show closest sales as comps (top 10 by distance)
    closest = sorted(
        [s for s in sales if s.sqft and s.sqft > 0],
        key=lambda s: s.distance_miles or 999,
    )[:10]

    render_local_estimate(
        address=geo.matched_address,
        features=features,
        estimate=estimate,
        closest_sales=closest,
    )


if __name__ == "__main__":
    app()
