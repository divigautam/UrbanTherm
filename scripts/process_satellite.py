import rasterio
import numpy as np

files = {
    "red": "data/raw/satellite/red.TIF",
    "nir": "data/raw/satellite/nir08.TIF",
    "thermal": "data/raw/satellite/lwir11.TIF"
}

data = {}

for name, file in files.items():
    print("Reading", name + "...")

    with rasterio.open(file) as src:
        data[name] = src.read(
            1,
            window=((0, 100), (0, 100))
        ).astype(float)

    print(name, "done")

red = data["red"]
nir = data["nir"]
thermal = data["thermal"]

# NDVI
ndvi = (nir - red) / (nir + red)

# LST
lst_celsius = (
    thermal * 0.00341802
    + 149.0
    - 273.15
)

valid = (
    np.isfinite(ndvi)
    & np.isfinite(lst_celsius)
    & (thermal != 0)
)

print("\nREAL SATELLITE TEST")
print("===================")
print("Valid pixels:", np.sum(valid))

print("\nNDVI")
print("Mean:", np.mean(ndvi[valid]))

print("\nLAND SURFACE TEMPERATURE")
print("Mean:", np.mean(lst_celsius[valid]), "°C")

print("\nSUCCESS!")