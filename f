import requests
import json

# 1. URL de la STAC API
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# 2. Geometria del teu AOI (Castelldefels)
geometry = {
    "type": "Polygon",
    "coordinates": [
        [
            [1.9462356856159317, 41.264465013042155],
            [1.9462356856159317, 41.26341613943984],
            [2.0028501773516894, 41.26341613943984],
            [2.0028501773516894, 41.264465013042155],
            [1.9462356856159317, 41.264465013042155]
        ]
    ]
}

# 3. Construir el payload de la consulta
payload = {
    "collections": ["sentinel-2-l2a"],
    "datetime": "2025-01-01T00:00:00Z/2025-07-01T23:59:59Z",
    "intersects": geometry,
    "limit": 5  # mostra només 5 resultats
}

# 4. Fer la petició POST a la STAC API
response = requests.post(STAC_URL, json=payload)
response.raise_for_status()
data = response.json()

# 5. Processar resultats
features = data.get("features", [])
print(f"✅ Escenes trobades: {len(features)}\n")

for feat in features:
    props = feat["properties"]
    cc = props.get("eo:cloud_cover", "N/A")
    dt = props.get("datetime", "N/A")
    assets = feat.get("assets", {})

    print(f"🛰️ {feat['id']} | data={dt} | núvols={cc}%")
    print("  Bands disponibles:", list(assets.keys()))
    print("  B02 (blue):", assets.get("B02", {}).get("href", "N/A"))
    print("  B03 (green):", assets.get("B03", {}).get("href", "N/A"))
    print("  B04 (red):", assets.get("B04", {}).get("href", "N/A"))
    print("  B08 (nir):", assets.get("B08", {}).get("href", "N/A"))
    print()
