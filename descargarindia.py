import requests
import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
import time
import json

# ==============================
# PARÀMETRES DEL PROJECTE
# ==============================
aoi_path = "map.geojson"
stac_url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
date_range = ["2025-01-01", "2025-07-01"]

# ==============================
# DESCÀRREGA I CERCA STAC
# ==============================
print("Baixant AOI...")
with open(aoi_path, "r", encoding="utf-8") as f:
    aoi_geojson = json.load(f)


payload = {
    "collections": [collection],
    "intersects": aoi_geojson["features"][0]["geometry"],
    "datetime": f"{date_range[0]}T00:00:00Z/{date_range[1]}T00:00:00Z",
    "limit": 20, #limit d'imatges per a optimitzar el temps de descàrrega
}

print("Consultant STAC API...")
resp = requests.post(f"{stac_url}/search", json=payload)
resp.raise_for_status()
data = resp.json()

save_folder = "./sentinel2_india"
os.makedirs(save_folder, exist_ok=True)
print(f"Les imatges NDWI es guardaran a: {os.path.abspath(save_folder)}")


def download_tiff_as_array(url):
    """Descarrega un TIFF i retorna el contingut com a array numpy."""
    start_t = time.time()
    r = requests.get(url)
    r.raise_for_status()
    duration = time.time() - start_t
    size_bytes = len(r.content)
    with MemoryFile(r.content) as memfile:
        with memfile.open() as dataset:
            array = dataset.read(1).astype(np.float32)
    return array, size_bytes, duration


# DESCÀRREGA I CÀLCUL NDWI
total_saved = 0
total_size = 0  # bytes
total_time_download = 0.0

start_all = time.time()

for item in data["features"]:
    item_id = item["id"]
    properties = item.get("properties", {})
    cloud_cover = properties.get("eo:cloud_cover", None)

    # 🔹 Filtre per nuvolositat
    if cloud_cover is not None and cloud_cover > 90:
        print(f"\n⛅ Saltant {item_id}: massa núvols ({cloud_cover:.1f}%)")
        continue

    assets = item.get("assets", {})
    print(f"\nProcessem item: {item_id} (nuvolositat={cloud_cover}%)")

    # Només necessitem bandes green i nir
    green_asset = assets.get("green")
    nir_asset = assets.get("nir")

    if not (green_asset and nir_asset):
        print("⚠️  Falta banda green o nir — no es pot calcular NDWI.")
        continue

    # Descarregar bandes
    print(" Baixant bandes per NDWI...")
    green_arr, green_size, green_time = download_tiff_as_array(green_asset["href"])
    nir_arr, nir_size, nir_time = download_tiff_as_array(nir_asset["href"])

    total_size += green_size + nir_size
    total_time_download += green_time + nir_time

    # Calcular NDWI
    print(" Calculant NDWI...")
    ndwi = (green_arr - nir_arr) / (green_arr + nir_arr + 1e-10)

    # Guardar imatge NDWI
    item_folder = os.path.join(save_folder, item_id)
    os.makedirs(item_folder, exist_ok=True)

    ndwi_path = os.path.join(item_folder, "ndwi_color.png")
    plt.figure(figsize=(8, 8))
    plt.imshow(ndwi, cmap='RdYlGn')
    plt.colorbar(label='NDWI')
    plt.axis('off')
    plt.savefig(ndwi_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f" ✅ NDWI guardat a {ndwi_path}")

    total_saved += 1


# ESTADÍSTIQUES GLOBALS

end_all = time.time()
total_time = end_all - start_all
total_size_mb = total_size / (1024 ** 2)
bandwidth = total_size_mb / total_time if total_time > 0 else 0

# Extrapolació per N mesos
N = 6  # pots canviar-ho
estimated_time_N_months = total_time * N


# RESULTATS

print("\n===== RESULTATS GLOBALS =====")
print(f"Total imatges NDWI processades: {total_saved}")
print(f"Mida total descarregada: {total_size_mb:.2f} MB")
print(f"Temps total descàrrega: {total_time_download:.2f} s")
print(f"Temps total execució: {total_time:.2f} s")
print(f"Ample de banda mitjà: {bandwidth:.2f} MB/s")
print(f"Temps estimat per {N} mesos: {estimated_time_N_months/60:.2f} min")

print("\nProcés completat correctament ✅")
