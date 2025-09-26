import requests
import json

# 1. URL de la STAC API
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# 2. Cargar el AOI desde tu GeoJSON
with open("polygon.geojson", "r") as f:
    aoi = json.load(f)
geometry = aoi["features"][0]["geometry"]

# 3. Construir el payload de búsqueda
payload = {
    "collections": ["sentinel-2-l2a"],
    "datetime": "2025-01-01T00:00:00Z/2025-07-01T23:59:59Z",
    "intersects": geometry,
    "limit": 5,  # máximo 5 resultados para no saturar
    "fields": {
        "include": [
            "id",
            "properties.datetime",
            "properties.eo:cloud_cover",
            "assets.B02",
            "assets.B03",
            "assets.B04",
            "assets.B08"
        ]
    }
}

# 4. Hacer la petición POST
response = requests.post(STAC_URL, json=payload)
response.raise_for_status()  # lanza error si falla

data = response.json()

# 5. Procesar y mostrar resultados
features = data.get("features", [])
print(f"Encontradas {len(features)} escenas\n")

for feat in features:
    props = feat["properties"]
    cc = props.get("eo:cloud_cover", "N/A")
    dt = props.get("datetime", "N/A")
    assets = feat.get("assets", {})

    print(f"🛰️ {feat['id']} | fecha={dt} | nubes={cc}%")
    print("  B02 (blue):", assets.get("B02", {}).get("href", "N/A"))
    print("  B03 (green):", assets.get("B03", {}).get("href", "N/A"))
    print("  B04 (red):", assets.get("B04", {}).get("href", "N/A"))
    print("  B08 (nir):", assets.get("B08", {}).get("href", "N/A"))
    print()
