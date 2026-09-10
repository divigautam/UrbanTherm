"""
UrbanTherm Step 3: historical weather for Ghaziabad (Open-Meteo).

Uses the existing 500 m grid only to pick one representative city coordinate
(mean of cell centroids). ERA5 reanalysis is ~9 km, so one point is enough;
this script does not rebuild the grid or boundary.

Fetches hourly:
  - temperature_2m
  - relative_humidity_2m
  - wind_speed_10m

Saves a clean CSV under data/processed/.

Run from the project root:
    python scripts/fetch_weather.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRID_CSV_PATH = PROCESSED_DIR / "ghaziabad_grid_500m.csv"
WEATHER_CSV_PATH = PROCESSED_DIR / "ghaziabad_weather.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
)
TIMEZONE = "Asia/Kolkata"
USER_AGENT = "UrbanTherm/0.1 (college project; Ghaziabad historical weather)"

# Archive reanalysis usually lags a few days behind "today".
ARCHIVE_LAG_DAYS = 5
EXPECTED_GRID_CELLS = 3881


def default_date_range() -> tuple[str, str]:
    """Previous complete calendar year, or last 365 available days if that year is incomplete."""
    latest = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    start = date(latest.year - 1, 1, 1)
    end = date(latest.year - 1, 12, 31)
    if end > latest:
        end = latest
        start = end - timedelta(days=364)
    return start.isoformat(), end.isoformat()


def parse_args() -> argparse.Namespace:
    default_start, default_end = default_date_range()
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo historical weather for Ghaziabad and write a CSV."
    )
    parser.add_argument(
        "--start-date",
        default=default_start,
        help=f"ISO date YYYY-MM-DD (default: {default_start})",
    )
    parser.add_argument(
        "--end-date",
        default=default_end,
        help=f"ISO date YYYY-MM-DD (default: {default_end})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WEATHER_CSV_PATH,
        help="Output CSV path (default: data/processed/ghaziabad_weather.csv)",
    )
    return parser.parse_args()


def city_coordinate_from_existing_grid() -> tuple[float, float]:
    """
    Read centroid_lat / centroid_lon from the existing grid CSV and return
    their arithmetic mean. Does not create a new grid.
    """
    if not GRID_CSV_PATH.exists():
        print(
            f"Existing grid not found: {GRID_CSV_PATH}\n"
            "Run  python scripts/create_grid.py  first (do not recreate the grid here).",
            file=sys.stderr,
        )
        sys.exit(1)

    lats: list[float] = []
    lons: list[float] = []
    with GRID_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lats.append(float(row["centroid_lat"]))
            lons.append(float(row["centroid_lon"]))

    n = len(lats)
    if n == 0:
        print(f"Grid CSV has no rows: {GRID_CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    if n != EXPECTED_GRID_CELLS:
        print(
            f"Note: expected {EXPECTED_GRID_CELLS} grid cells, found {n} in {GRID_CSV_PATH}",
            file=sys.stderr,
        )

    lat = sum(lats) / n
    lon = sum(lons) / n
    return lat, lon


def fetch_archive(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict:
    query = urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.6f}",
            "longitude": f"{longitude:.6f}",
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": TIMEZONE,
        }
    )
    url = f"{ARCHIVE_URL}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Requesting Open-Meteo Historical Weather API...\n  {url}")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Open-Meteo HTTP {exc.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Could not reach Open-Meteo: {exc}", file=sys.stderr)
        sys.exit(1)

    if "hourly" not in payload:
        print(f"Unexpected API response (no hourly block): {payload}", file=sys.stderr)
        sys.exit(1)
    return payload


def write_csv(path: Path, payload: dict) -> int:
    hourly = payload["hourly"]
    times = hourly.get("time") or []
    for name in HOURLY_VARS:
        if name not in hourly:
            print(f"API response missing hourly variable: {name}", file=sys.stderr)
            sys.exit(1)
        if len(hourly[name]) != len(times):
            print(
                f"Length mismatch for {name}: {len(hourly[name])} vs {len(times)} times",
                file=sys.stderr,
            )
            sys.exit(1)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["time", *HOURLY_VARS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, timestamp in enumerate(times):
            writer.writerow(
                {
                    "time": timestamp,
                    **{name: hourly[name][i] for name in HOURLY_VARS},
                }
            )
    return len(times)


def main() -> None:
    args = parse_args()
    latitude, longitude = city_coordinate_from_existing_grid()
    print(
        f"Using existing grid ({GRID_CSV_PATH.name}) mean centroid: "
        f"{latitude:.6f}, {longitude:.6f}"
    )

    payload = fetch_archive(latitude, longitude, args.start_date, args.end_date)
    n_rows = write_csv(args.output, payload)
    units = payload.get("hourly_units", {})

    print("\n=== UrbanTherm weather summary ===")
    print(f"API:                     {ARCHIVE_URL}")
    print(f"Latitude, longitude:     {payload.get('latitude')}, {payload.get('longitude')}")
    print(f"Timezone:                {payload.get('timezone', TIMEZONE)}")
    print(f"Date range:              {args.start_date} to {args.end_date}")
    print(f"Hourly rows:             {n_rows}")
    units_ascii = {key: str(val).replace("\N{DEGREE SIGN}", "") for key, val in units.items()}
    print(f"Units:                   {units_ascii}")
    print(f"CSV:                     {args.output.resolve()}")


if __name__ == "__main__":
    main()
