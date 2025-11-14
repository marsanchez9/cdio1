import requests
import os
from PIL import Image
from io import BytesIO

# --- AOI basado en tu punto: (17.034261, 78.183078) ---
aoi_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [78.173078, 17.044261],
                    [78.173078, 17.024261],
                    [78.193078, 17.024261],
                    [78.193078, 17.044261],
                    [78.173078, 17.044261]
                ]]
            }
        }
    ]
}

stac_url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
date_range = ["2025-01-01", "2025-07-01"]

payload = {
    "collections": [collection],
    "intersects": aoi_geojson["features"][0]["geometry"],
    "datetime": f"{date_range[0]}T00:00:00Z/{date_range[1]}T00:00:00Z",
    "limit": 10,
}

print("Consultando STAC API…")
resp = requests.post(f"{stac_url}/search", json=payload)
resp.raise_for_status()
data = resp.json()

save_folder = "./sentinel2_india_images"
os.makedirs(save_folder, exist_ok=True)
print(f"Las imágenes se guardarán en: {os.path.abspath(save_folder)}")

total_saved = 0

for item in data["features"]:
    item_id = item["id"]
    assets = item.get("assets", {})
    
    print(f"\nProcesando item: {item_id}")
    
    bands = {
        "red": assets.get("red"),
        "green": assets.get("green"),
        "blue": assets.get("blue"),
        "preview": assets.get("thumbnail")
    }

    item_folder = os.path.join(save_folder, item_id)
    os.makedirs(item_folder, exist_ok=True)
    
    def download_and_save(url, filepath):
        print(f"  Bajando {url}…")
        r = requests.get(url)
        r.raise_for_status()
        content = r.content
        
        if url.endswith(".tif") or url.endswith(".tiff"):
            img = Image.open(BytesIO(content))
            img.convert("RGB").save(filepath, "JPEG")
        else:
            with open(filepath, "wb") as f:
                f.write(content)

    # Descargar bandas RGB
    for color in ["red", "green", "blue"]:
        asset = bands[color]
        if asset:
            filepath = os.path.join(item_folder, f"{color}.jpg")
            download_and_save(asset["href"], filepath)
        else:
            print(f"  Aviso: no hay banda {color}")

    # Descargar vista previa
    if bands.get("preview"):
        download_and_save(bands["preview"]["href"], os.path.join(item_folder, "preview.jpg"))

    total_saved += 1

print(f"\nTotal items descargados: {total_saved}")
