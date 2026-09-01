"""
UrbanTherm Step 2: Ghaziabad study area and 500 m spatial grid.

This script:
  1. Downloads the real Ghaziabad boundary from OpenStreetMap (Nominatim)
     using OSMnx — no hand-drawn or fake coordinates.
  2. Saves that boundary as GeoJSON.
  3. Reprojects to a meter-based CRS (UTM zone 43N).
  4. Builds a 500 m × 500 m fishnet, then clips it to the city polygon.
  5. Writes GeoJSON + CSV outputs and a preview map.

Run from the project root:
    python scripts/create_grid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

# ---------------------------------------------------------------------------
# Paths: resolve from this file so the script works from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

BOUNDARY_PATH = RAW_DIR / "ghaziabad_boundary.geojson"
GRID_GEOJSON_PATH = PROCESSED_DIR / "ghaziabad_grid_500m.geojson"
GRID_CSV_PATH = PROCESSED_DIR / "ghaziabad_grid_500m.csv"
PREVIEW_PATH = DOCS_DIR / "ghaziabad_grid_preview.png"

# Nominatim / OSM place query. Keep it specific so we get the Indian city,
# not a similarly named place elsewhere.
PLACE_QUERY = "Ghaziabad, Uttar Pradesh, India"

# 500 m cells must be created in a projected CRS (metres), not lat/lon degrees.
# Ghaziabad is near 77.4°E, 28.7°N → UTM zone 43 North.
GRID_CRS = "EPSG:32643"
CELL_SIZE_M = 500
WGS84 = "EPSG:4326"

# Nominatim can be slow or rate-limit. Retry a few times before giving up.
DOWNLOAD_ATTEMPTS = 3


def download_ghaziabad_boundary() -> gpd.GeoDataFrame:
    """
    Ask Nominatim (via OSMnx) for the official OSM polygon of Ghaziabad.

    OSMnx wraps the Nominatim geocoder. The result is a GeoDataFrame in
    WGS84 (EPSG:4326) with a Polygon or MultiPolygon geometry — that
    geometry is the study-area boundary we clip the grid to later.
    """
    try:
        import osmnx as ox
    except ImportError:
        print(
            "OSMnx is not installed. Install geospatial packages with:\n"
            "  pip install geopandas shapely osmnx matplotlib",
            file=sys.stderr,
        )
        sys.exit(1)

    # Identify this project to Nominatim (their usage policy asks for a
    # descriptive user agent, not a generic default).
    ox.settings.http_user_agent = "UrbanTherm/0.1 (college project; Ghaziabad heat grid)"
    ox.settings.timeout = 180

    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            print(f"Downloading '{PLACE_QUERY}' from OpenStreetMap/Nominatim "
                  f"(attempt {attempt}/{DOWNLOAD_ATTEMPTS})...")
            # geocode_to_gdf returns the place polygon, not a street network.
            boundary = ox.geocode_to_gdf(PLACE_QUERY)
            if boundary.empty:
                raise RuntimeError("Nominatim returned no geometry for Ghaziabad.")
            return boundary
        except Exception as exc:  # network errors, HTTP 429, empty results, etc.
            last_error = exc
            print(f"  Nominatim request failed: {exc}", file=sys.stderr)

    print(
        "Could not retrieve the Ghaziabad boundary from OpenStreetMap/Nominatim.\n"
        "Check your internet connection and try again later "
        "(Nominatim also rate-limits frequent requests).\n"
        f"Last error: {last_error}",
        file=sys.stderr,
    )
    sys.exit(1)


def keep_polygon_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop non-area features if Nominatim ever returns a point or line."""
    polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polygons.empty:
        raise RuntimeError(
            "The OSM result for Ghaziabad was not a polygon. "
            "Cannot build an area grid from a point or line."
        )
    return polygons


def validate_and_fix_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    OSM polygons are usually valid, but a tiny self-intersection can break
    clip/overlay. buffer(0) is a common Shapely trick to repair those.
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        raise RuntimeError(
            f"{int(invalid.sum())} boundary geometry/geometries are still invalid "
            "after repair. Stopped so we do not build a grid on a broken polygon."
        )
    return gdf


def dissolve_to_one_polygon(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge all parts into a single study-area feature for clipping."""
    dissolved = gdf.dissolve().reset_index(drop=True)
    return dissolved[["geometry"]]


def make_fishnet(boundary_proj: gpd.GeoDataFrame, cell_size: float) -> gpd.GeoDataFrame:
    """
    Build a regular square grid that covers the projected bounding box.

    Steps:
      1. Read min/max X and Y in metres from the projected boundary.
      2. Walk across that box in `cell_size` steps and create Shapely boxes.
      3. Return a GeoDataFrame in the same projected CRS.

    We do NOT use latitude/longitude here. One degree of longitude is not
    500 m, and it changes with latitude, so a "square" in degrees would
    not be 500 m × 500 m on the ground.
    """
    minx, miny, maxx, maxy = boundary_proj.total_bounds

    # arange stops before max, so add one extra step to cover the far edge.
    xs = np.arange(minx, maxx + cell_size, cell_size)
    ys = np.arange(miny, maxy + cell_size, cell_size)

    cells = [
        box(x, y, x + cell_size, y + cell_size)
        for x in xs[:-1]
        for y in ys[:-1]
    ]
    return gpd.GeoDataFrame({"geometry": cells}, crs=boundary_proj.crs)


def clip_grid_to_city(grid: gpd.GeoDataFrame, city: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only the parts of each square that fall inside Ghaziabad.

    Interior cells stay 500 m × 500 m. Edge cells become smaller leftover
    polygons after intersection with the city boundary.
    """
    clipped = gpd.overlay(grid, city, how="intersection", keep_geom_type=True)
    clipped = clipped[~clipped.geometry.is_empty].copy()
    # Drop numerical slivers that are not real study cells.
    clipped = clipped[clipped.geometry.area > 1.0].copy()
    clipped = clipped.reset_index(drop=True)
    if clipped.empty:
        raise RuntimeError("Clipping produced no grid cells. Check the boundary CRS.")
    return clipped


def add_attributes(grid_proj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add grid_id, area, and WGS84 centroids.

    Centroids are computed in the projected CRS (accurate metres), then
    transformed to lat/lon for mapping and CSV export.
    """
    n = len(grid_proj)
    width = max(3, len(str(n)))
    grid_proj = grid_proj.copy()
    grid_proj["grid_id"] = [f"G{i:0{width}d}" for i in range(1, n + 1)]
    grid_proj["area_m2"] = grid_proj.geometry.area

    # Centroid in metres, then convert that POINT to WGS84 for lat/lon columns.
    centroids_wgs84 = gpd.GeoSeries(grid_proj.geometry.centroid, crs=grid_proj.crs).to_crs(WGS84)
    grid_proj["centroid_lon"] = centroids_wgs84.x
    grid_proj["centroid_lat"] = centroids_wgs84.y
    return grid_proj


def _json_ready(value):
    """Turn pandas/NumPy values into types json.dumps can serialize."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_geojson_feature_collection(
    path: Path,
    gdf: gpd.GeoDataFrame,
    property_columns: list[str] | None = None,
) -> None:
    """
    Write a GeoJSON FeatureCollection with the standard library.

    GeoPandas `to_file()` uses Pyogrio/GDAL. On some Windows setups the GDAL
    DLL is blocked, so we convert each Shapely geometry with `mapping()` and
    dump JSON ourselves. Coordinates must already be WGS84 (lon, lat).
    """
    features: list[dict] = []
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        if property_columns is None:
            props = {
                key: _json_ready(val)
                for key, val in row.items()
                if key != "geometry"
            }
        else:
            props = {key: _json_ready(row[key]) for key in property_columns}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(row.geometry),
            }
        )

    collection = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(collection), encoding="utf-8")


def save_outputs(boundary_wgs84: gpd.GeoDataFrame, grid_proj: gpd.GeoDataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # GeoJSON only — no GeoDataFrame.to_file() / Pyogrio / GDAL.
    write_geojson_feature_collection(BOUNDARY_PATH, boundary_wgs84)

    grid_wgs84 = grid_proj.to_crs(WGS84)
    write_geojson_feature_collection(
        GRID_GEOJSON_PATH,
        grid_wgs84,
        property_columns=["grid_id", "centroid_lat", "centroid_lon", "area_m2"],
    )

    pd.DataFrame(
        grid_proj[["grid_id", "centroid_lat", "centroid_lon", "area_m2"]]
    ).to_csv(GRID_CSV_PATH, index=False)


def save_preview(boundary_proj: gpd.GeoDataFrame, grid_proj: gpd.GeoDataFrame) -> None:
    """Draw the city outline, the clipped 500 m cells, and IDs when they fit."""
    fig, ax = plt.subplots(figsize=(12, 12))
    boundary_proj.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=3)
    grid_proj.plot(ax=ax, facecolor="#f4a261", edgecolor="#9a3412", linewidth=0.25, alpha=0.55)

    n = len(grid_proj)
    # Hundreds of labels on a dense grid become unreadable; skip IDs if too many.
    if n <= 400:
        for _, row in grid_proj.iterrows():
            c = row.geometry.centroid
            ax.text(
                c.x,
                c.y,
                row["grid_id"],
                fontsize=3.5,
                ha="center",
                va="center",
                color="#1f2937",
            )
        id_note = "all grid IDs labeled"
    else:
        id_note = f"IDs omitted ({n} cells — too dense to read)"

    ax.set_title(
        f"UrbanTherm — Ghaziabad 500 m grid\n"
        f"{n} cells | {GRID_CRS} | {id_note}"
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PREVIEW_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    boundary = download_ghaziabad_boundary()
    downloaded_crs = boundary.crs

    boundary = keep_polygon_geometries(boundary)
    boundary = validate_and_fix_geometry(boundary)
    # Keep the real OSM polygon(s) in WGS84 for the boundary GeoJSON file.
    boundary_downloaded_wgs84 = boundary.to_crs(WGS84)
    boundary_wgs84 = dissolve_to_one_polygon(boundary_downloaded_wgs84)

    # --- Project to metres, then build the fishnet ---
    boundary_proj = boundary_wgs84.to_crs(GRID_CRS)
    if not boundary_proj.crs:
        raise RuntimeError("Failed to assign a projected CRS to the boundary.")

    grid = make_fishnet(boundary_proj, CELL_SIZE_M)
    grid = clip_grid_to_city(grid, boundary_proj)
    grid = add_attributes(grid)

    save_outputs(boundary_downloaded_wgs84, grid)
    save_preview(boundary_proj, grid)

    print("\n=== UrbanTherm grid summary ===")
    print(f"Place query:              {PLACE_QUERY}")
    print(f"CRS of downloaded boundary: {downloaded_crs}")
    print(f"CRS used for grid creation: {GRID_CRS} (UTM 43N, units = metres)")
    print(f"Number of grid cells:     {len(grid)}")
    print(f"Minimum cell area (m²):   {grid['area_m2'].min():.2f}")
    print(f"Maximum cell area (m²):   {grid['area_m2'].max():.2f}")
    print(f"Boundary GeoJSON:         {BOUNDARY_PATH}")
    print(f"Grid GeoJSON:             {GRID_GEOJSON_PATH}")
    print(f"Grid CSV:                 {GRID_CSV_PATH}")
    print(f"Preview PNG:              {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
