import requests
import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import rowcol
from affine import Affine
from shapely.geometry import shape
import matplotlib.pyplot as plt
import time
import json
from shapely.ops import transform
from pyproj import Transformer

def reproject_geom(geom, src_crs, dst_crs):
    """Reprojecta un geom de src_crs a dst_crs"""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return [transform(transformer.transform, g) for g in geom]


# ==============================
# PARÀMETRES DEL PROJECTE
# ==============================
aoi_path = "map.geojson"  # Posa aquí el camí correcte
stac_url = "https://earth-search.aws.element84.com/v1"
collection = "sentinel-2-l2a"
date_range = ["2025-01-01", "2025-07-01"]

save_folder = "./sentinel2_india"
os.makedirs(save_folder, exist_ok=True)

# ==============================
# FUNCIONS AUXILIARS
# ==============================
def download_tiff(url):
    """Descarrega un TIFF i retorna array, transform i CRS."""
    r = requests.get(url)
    r.raise_for_status()
    with MemoryFile(r.content) as memfile:
        with memfile.open() as dataset:
            arr = dataset.read(1).astype(np.float32)
            transform = dataset.transform
            crs = dataset.crs
    return arr, transform, crs


def clip_to_aoi(array, transform, crs, geom):
    """Retalla un array raster segons un polígon AOI."""
    height, width = array.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": array.dtype,
        "crs": crs,
        "transform": transform,
    }
    with MemoryFile() as mem:
        with mem.open(**profile) as dataset:
            dataset.write(array, 1)
            # Prepare geometries for rasterio.mask.mask: accept shapely geometries or GeoJSON mappings
            from shapely.geometry import mapping
            geom_for_mask = []
            for g in geom:
                if hasattr(g, 'geom_type'):
                    geom_for_mask.append(mapping(g))
                else:
                    geom_for_mask.append(g)
            out_img, out_transform = mask(dataset, geom_for_mask, crop=True)
    return out_img[0], out_transform

def crop_100m(array, transform):
    """Retalla un raster a 100x100 metres al voltant del centre."""
    res_x = transform.a
    res_y = -transform.e
    px = int(100 / res_x)
    py = int(100 / res_y)
    h, w = array.shape
    cx, cy = w//2, h//2
    x1 = max(0, cx - px//2)
    x2 = min(w, cx + px//2)
    y1 = max(0, cy - py//2)
    y2 = min(h, cy + py//2)
    return array[y1:y2, x1:x2]


def crop_around_geom_100m(array, transform, crs, geom):
    """Recorta el array para que el AOI (geom) quede dentro de un área de 100x100 m.

    - Si el AOI ya es mayor a 100 m en alguna dimensión, se recorta a la caja mínima que contiene el AOI.
    - Maneja CRS geográfico estimando grados⇄metros alrededor del centro si es necesario.
    """
    # Obtener bounds del AOI en el CRS del raster
    # geom es lista de shapely geometrías
    from shapely.ops import unary_union
    union = unary_union(geom)
    minx, miny, maxx, maxy = union.bounds

    # Centrado del AOI (en unidades del CRS del raster)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    # Dimensiones del AOI en unidades del CRS
    aoi_w = maxx - minx
    aoi_h = maxy - miny

    # Convertir 100 m a unidades del CRS. Si CRS es geográfico (grados) aproximamos.
    try:
        is_geographic = getattr(crs, "is_geographic", False)
    except Exception:
        is_geographic = False

    half_m = 50.0
    if is_geographic:
        # Aproximación: metros por grado en latitud/longitud alrededor del centro
        lat = cy
        lat_rad = np.deg2rad(lat)
        meters_per_deg_lat = 111132.0
        meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
        half_x = half_m / meters_per_deg_lon
        half_y = half_m / meters_per_deg_lat
        aoi_w_m = aoi_w * meters_per_deg_lon
        aoi_h_m = aoi_h * meters_per_deg_lat
    else:
        # Asumimos unidades en metros
        half_x = half_m
        half_y = half_m
        aoi_w_m = aoi_w
        aoi_h_m = aoi_h

    # Si el AOI es más grande que 100 m, avisamos y recortamos la caja que contiene el AOI
    if aoi_w_m > 100.0 or aoi_h_m > 100.0:
        print("⚠️ El AOI es mayor que 100×100 m en al menos una dimensión; se recortará a la caja del AOI.")
        bx_min, by_min, bx_max, by_max = minx, miny, maxx, maxy
    else:
        bx_min = cx - half_x
        bx_max = cx + half_x
        by_min = cy - half_y
        by_max = cy + half_y

    # Convertir caja en coordenadas espaciales a filas/columnas de la matriz
    # rowcol devuelve (row, col)
    # top-left: (bx_min, by_max), bottom-right: (bx_max, by_min)
    try:
        r1, c1 = rowcol(transform, bx_min, by_max)
        r2, c2 = rowcol(transform, bx_max, by_min)
    except Exception:
        # Fallback manual si rowcol no funciona
        a = transform.a
        c = transform.c
        f = transform.f
        # col = (x - c) / a
        c1 = int((bx_min - c) / a)
        c2 = int((bx_max - c) / a)
        # row = (y - f) / e
        e = transform.e
        r1 = int((by_max - f) / e)
        r2 = int((by_min - f) / e)

    row_start = max(0, min(r1, r2))
    row_stop = min(array.shape[0], max(r1, r2) + 1)
    col_start = max(0, min(c1, c2))
    col_stop = min(array.shape[1], max(c1, c2) + 1)

    if row_start >= row_stop or col_start >= col_stop:
        print("⚠️ Ventana de recorte inválida; se devuelve el array original.")
        return array

    cropped = array[row_start:row_stop, col_start:col_stop]

    # Calcular nuevo transform para la ventana recortada
    try:
        new_transform = transform * Affine.translation(col_start, row_start)
    except Exception:
        # Fallback: recomponer transform manualmente
        a = transform.a
        b = transform.b
        c = transform.c
        d = transform.d
        e = transform.e
        f = transform.f
        new_c = c + col_start * a + row_start * b
        new_f = f + col_start * d + row_start * e
        new_transform = Affine(a, b, new_c, d, e, new_f)

    return cropped, new_transform

# ==============================
# CARREGAR AOI
# ==============================
print("Baixant AOI...")
with open(aoi_path, "r", encoding="utf-8") as f:
    aoi_geojson = json.load(f)
geometry = aoi_geojson["features"][0]["geometry"]
geom = [shape(geometry)]

# ==============================
# CONSULTA STAC
# ==============================
payload = {
    "collections": [collection],
    "intersects": geometry,
    "datetime": f"{date_range[0]}T00:00:00Z/{date_range[1]}T00:00:00Z",
    "limit": 20,
}

print("Consultant STAC API...")
resp = requests.post(f"{stac_url}/search", json=payload)
resp.raise_for_status()
data = resp.json()

print(f"Les imatges NDWI es guardaran a: {os.path.abspath(save_folder)}")

# ==============================
# DESCÀRREGA I PROCESSAMENT
# ==============================
total_saved = 0
total_size = 0
total_time_download = 0.0
start_all = time.time()

for item in data["features"]:
    item_id = item["id"]
    properties = item.get("properties", {})
    cloud_cover = properties.get("eo:cloud_cover", None)

    if cloud_cover is not None and cloud_cover > 10:
        print(f"\n⛅ Saltant {item_id}: massa núvols ({cloud_cover:.1f}%)")
        continue

    assets = item.get("assets", {})
    green_asset = assets.get("green")
    nir_asset = assets.get("nir")

    if not (green_asset and nir_asset):
        print(f"⚠️ Falta banda green o nir per {item_id}")
        continue

    print(f"\nProcessem {item_id} (nuvolositat={cloud_cover}%)")
    green_arr, green_transform, green_crs = download_tiff(green_asset["href"])
    nir_arr, nir_transform, nir_crs = download_tiff(nir_asset["href"])

    # NDWI
    ndwi = (green_arr - nir_arr) / (green_arr + nir_arr + 1e-10)

    # Reprojecta l'AOI al CRS del raster
    geom_proj = reproject_geom(geom, "EPSG:4326", green_crs)


    # Retallar AOI (usant la geometria reprojetada)
    ndwi_clip, ndwi_clip_transform = clip_to_aoi(ndwi, green_transform, green_crs, geom_proj)
    # Retallar dins d'una finestra de 100x100 metres al voltant de l'AOI (o la caixa de l'AOI si és més gran)
    ndwi_100m, ndwi_100m_transform = crop_around_geom_100m(ndwi_clip, ndwi_clip_transform, green_crs, geom_proj)

    # Crear carpeta
    item_folder = os.path.join(save_folder, item_id)
    os.makedirs(item_folder, exist_ok=True)

    # Guardar resultats (GeoTIFFs georreferenciats)
    clip_tif = os.path.join(item_folder, "ndwi_clip.tif")
    profile = {
        "driver": "GTiff",
        "height": ndwi_clip.shape[0],
        "width": ndwi_clip.shape[1],
        "count": 1,
        "dtype": ndwi_clip.dtype,
        "crs": green_crs,
        "transform": ndwi_clip_transform,
    }
    with rasterio.open(clip_tif, 'w', **profile) as dst:
        dst.write(ndwi_clip, 1)

    clip100_tif = os.path.join(item_folder, "ndwi_100m.tif")
    profile100 = profile.copy()
    profile100.update({
        "height": ndwi_100m.shape[0],
        "width": ndwi_100m.shape[1],
        "transform": ndwi_100m_transform,
    })
    with rasterio.open(clip100_tif, 'w', **profile100) as dst:
        dst.write(ndwi_100m, 1)

    # Also save PNG previews for quick inspection
    plt.imsave(os.path.join(item_folder, "ndwi_clip.png"), ndwi_clip, cmap="RdYlGn")
    plt.imsave(os.path.join(item_folder, "ndwi_100m.png"), ndwi_100m, cmap="RdYlGn")

    ndwi_path = os.path.join(item_folder, "ndwi_color.png")
    plt.figure(figsize=(8,8))
    plt.imshow(ndwi, cmap='RdYlGn')
    plt.colorbar(label='NDWI')
    plt.axis('off')
    plt.savefig(ndwi_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"✅ NDWI guardat a {ndwi_path}")
    total_saved += 1

end_all = time.time()
total_time = end_all - start_all

# ==============================
# ESTADÍSTIQUES
# ==============================
print("\n===== RESULTATS GLOBALS =====")
print(f"Total imatges NDWI processades: {total_saved}")
print(f"Temps total execució: {total_time:.2f} s")
print(f"Ample de banda aproximat: {total_size/1024**2/total_time:.2f} MB/s" if total_time>0 else "N/A")
print("Procés completat correctament ✅")
