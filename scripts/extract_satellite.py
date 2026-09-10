from pystac_client import Client
import planetary_computer
import xarray as xr

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=[77.35, 28.55, 77.55, 28.85],
    datetime="2025-05-01/2025-06-30",
    query={"eo:cloud_cover": {"lt": 20}}
)

item = list(search.items())[0]

print("Selected:", item.id)

planetary_computer.sign_inplace(item)

# -------------------------
# NDVI
# -------------------------

red = xr.open_dataarray(
    item.assets["red"].href,
    engine="rasterio"
)

nir = xr.open_dataarray(
    item.assets["nir08"].href,
    engine="rasterio"
)

ndvi = (nir - red) / (nir + red)

print("\nNDVI")
print("Mean:", float(ndvi.mean()))

# -------------------------
# LAND SURFACE TEMPERATURE
# -------------------------

thermal = xr.open_dataarray(
    item.assets["lwir11"].href,
    engine="rasterio"
)

thermal = thermal.where(thermal != 0)

lst_kelvin = thermal * 0.00341802 + 149.0
lst_celsius = lst_kelvin - 273.15

print("\nLand Surface Temperature")
print("Minimum:", float(lst_celsius.min()), "°C")
print("Maximum:", float(lst_celsius.max()), "°C")
print("Mean:", float(lst_celsius.mean()), "°C")

print("\nReal satellite LST + NDVI extraction successful!")