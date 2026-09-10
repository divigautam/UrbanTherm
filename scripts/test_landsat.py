from pystac_client import Client

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=[77.35, 28.55, 77.55, 28.85],
    datetime="2025-05-01/2025-06-30",
    query={"eo:cloud_cover": {"lt": 20}}
)

items = list(search.items())

print("Images found:", len(items))

if items:
    item = items[0]

    print("\nSelected image:")
    print(item.id)

    print("\nAvailable bands/assets:")
    for name in item.assets:
        print("-", name)