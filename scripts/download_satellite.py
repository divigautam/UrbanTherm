from pystac_client import Client
import planetary_computer
import requests
import os

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

os.makedirs("data/raw/satellite", exist_ok=True)

bands = ["swir16"]

for band in bands:

    url = item.assets[band].href

    print(f"Downloading {band}...")

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    filename = f"data/raw/satellite/{band}.TIF"

    with open(filename, "wb") as f:
        f.write(response.content)

    print(f"Saved: {filename}")

print("\nSatellite bands downloaded successfully!")