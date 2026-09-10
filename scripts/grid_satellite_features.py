import rasterio
import pandas as pd
import numpy as np
from pyproj import Transformer

grid_file = "data/processed/ghaziabad_grid_500m.csv"

red_file = "data/raw/satellite/red.TIF"
nir_file = "data/raw/satellite/nir08.TIF"
swir_file = "data/raw/satellite/swir16.TIF"
thermal_file = "data/raw/satellite/lwir11.TIF"

output_file = "data/processed/ghaziabad_satellite_features.csv"

print("Loading grid...", flush=True)

grid = pd.read_csv(grid_file)

print("Grid cells:", len(grid), flush=True)

# -----------------------------
# LOAD SATELLITE BANDS
# -----------------------------

print("Loading Red...", flush=True)

with rasterio.open(red_file) as src:
    red = src.read(1).astype(np.float32)
    transform = src.transform
    raster_crs = src.crs
    height = src.height
    width = src.width

print("Red loaded", flush=True)

print("Loading NIR...", flush=True)

with rasterio.open(nir_file) as src:
    nir = src.read(1).astype(np.float32)

print("NIR loaded", flush=True)

print("Loading SWIR...", flush=True)

with rasterio.open(swir_file) as src:
    swir = src.read(1).astype(np.float32)

print("SWIR loaded", flush=True)

print("Loading Thermal...", flush=True)

with rasterio.open(thermal_file) as src:
    thermal = src.read(1).astype(np.float32)

print("Thermal loaded", flush=True)

# -----------------------------
# COORDINATE TRANSFORMATION
# -----------------------------

print("Converting grid coordinates...", flush=True)

transformer = Transformer.from_crs(
    "EPSG:4326",
    raster_crs,
    always_xy=True
)

x, y = transformer.transform(
    grid["centroid_lon"].to_numpy(),
    grid["centroid_lat"].to_numpy()
)

rows, cols = rasterio.transform.rowcol(
    transform,
    x,
    y
)

rows = np.asarray(rows)
cols = np.asarray(cols)

valid = (
    (rows >= 0) &
    (rows < height) &
    (cols >= 0) &
    (cols < width)
)

print("Valid grid cells:", valid.sum(), flush=True)

# -----------------------------
# EXTRACT PIXELS
# -----------------------------

red_values = red[rows[valid], cols[valid]]
nir_values = nir[rows[valid], cols[valid]]
swir_values = swir[rows[valid], cols[valid]]
thermal_values = thermal[rows[valid], cols[valid]]

# -----------------------------
# REFLECTANCE SCALING
# -----------------------------

red_reflectance = red_values * 0.0000275 - 0.2
nir_reflectance = nir_values * 0.0000275 - 0.2
swir_reflectance = swir_values * 0.0000275 - 0.2

# -----------------------------
# NDVI
# -----------------------------

ndvi_denominator = nir_reflectance + red_reflectance

ndvi = np.where(
    ndvi_denominator != 0,
    (nir_reflectance - red_reflectance)
    / ndvi_denominator,
    np.nan
)

# -----------------------------
# NDBI
# -----------------------------

ndbi_denominator = swir_reflectance + nir_reflectance

ndbi = np.where(
    ndbi_denominator != 0,
    (swir_reflectance - nir_reflectance)
    / ndbi_denominator,
    np.nan
)

# -----------------------------
# LAND SURFACE TEMPERATURE
# -----------------------------

lst_celsius = (
    thermal_values * 0.00341802
    + 149.0
    - 273.15
)

# -----------------------------
# CREATE DATASET
# -----------------------------

result = grid.loc[
    valid,
    [
        "grid_id",
        "centroid_lat",
        "centroid_lon"
    ]
].copy()

result["ndvi"] = ndvi
result["ndbi"] = ndbi
result["lst_celsius"] = lst_celsius

# Remove invalid thermal pixels
result.loc[thermal_values == 0, "lst_celsius"] = np.nan

# Save
result.to_csv(output_file, index=False)

# -----------------------------
# RESULTS
# -----------------------------

print("\n================================")
print("FINAL SATELLITE DATASET CREATED")
print("================================")

print("Cells:", len(result))

print("\nNDVI")
print("Mean:", np.nanmean(result["ndvi"]))
print("Min:", np.nanmin(result["ndvi"]))
print("Max:", np.nanmax(result["ndvi"]))

print("\nNDBI")
print("Mean:", np.nanmean(result["ndbi"]))
print("Min:", np.nanmin(result["ndbi"]))
print("Max:", np.nanmax(result["ndbi"]))

print("\nLST")
print("Mean:", np.nanmean(result["lst_celsius"]), "°C")
print("Min:", np.nanmin(result["lst_celsius"]), "°C")
print("Max:", np.nanmax(result["lst_celsius"]), "°C")

print("\nSaved:", output_file)
print("DONE!")