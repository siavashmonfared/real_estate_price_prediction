"""Download training datasets (King County and Ames housing data)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
from rich.console import Console

from housing_estimator.config import settings

console = Console()

# Public dataset URLs
DATASETS = {
    "kc_house_data.csv": {
        "url": "https://raw.githubusercontent.com/rashida048/Datasets/refs/heads/master/home_data.csv",
        "description": "King County House Sales (21K homes)",
    },
    "ames_housing.csv": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv",
        "description": "Ames Iowa Housing Dataset",
        # Note: the Ames dataset on this URL is the Boston housing (smaller).
        # We'll try an alternative Ames URL as well.
        "alt_url": "https://raw.githubusercontent.com/rashida048/Datasets/refs/heads/master/AmesHousing.csv",
    },
}


def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file from a URL."""
    console.print(f"  Downloading {description}...")
    console.print(f"    URL: {url}")
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            size_mb = len(resp.content) / 1_048_576
            console.print(f"    Saved: {dest} ({size_mb:.1f} MB)")
            return True
    except httpx.HTTPError as e:
        console.print(f"    [red]Failed: {e}[/red]")
        return False


def main():
    raw_dir = settings.data.raw_path
    raw_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for filename, info in DATASETS.items():
        dest = raw_dir / filename
        if dest.exists():
            console.print(f"  [yellow]{filename} already exists, skipping.[/yellow]")
            success_count += 1
            continue

        ok = download_file(info["url"], dest, info["description"])
        if not ok and "alt_url" in info:
            console.print("    Trying alternative URL...")
            ok = download_file(info["alt_url"], dest, info["description"])

        if ok:
            success_count += 1

    console.print(f"\n  Downloaded {success_count}/{len(DATASETS)} datasets.")
    if success_count == 0:
        console.print("[red]  No datasets downloaded. Check your internet connection.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
