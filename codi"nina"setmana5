#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CDIO1 - Session 5: Download Sentinel-2 L2A via STAC, cloud-filter, RGB viz, NDWI, metrics.
- STAC: https://earth-search.aws.element84.com/v1
- AOI: Castelledefels (del repo), o usa cualquier GeoJSON local
"""

import os
import io
import json
import math
import time
import pickle
import shutil
import zipfile
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterable

import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# CONFIGURACIÓN PRINCIPAL
# =========================

CFG = {
    # AOI: usa la URL del enunciado o un path local a tu GeoJSON
    "aoi_geojson_url": "https://raw.githubusercontent.com/thoyo-upc/bdse-cdio1/main/coastline_estimator/projects/castelldefels_h1_2025/input/polygon.geojson",
    # Si prefieres local, pon la ruta y deja aoi_geojson_url = None:
    "aoi_geojson_local": None,

    "stac_url": "https://earth-search.aws.element84.com/v1",
    "collection": "sentinel-2-l2a",
    "date_range": ["2025-01-01", "2025-07-01"],  # [inicio, fin)
    "bands": ["red", "green", "blue", "nir"],

    # Filtro de nubes (None para “sin filtro”)
    "max_cloud_percent": 10.0,

    # Descargas y salidas
    "out_dir": "outputs_session5",           # raíz de trabajo
    "download_dir": "downloads",             # dentro de out_dir
    "ndwi_dir": "ndwi",                      # dentro de out_dir
    "fig_dir": "figs",                       # dentro de out_dir
    "metrics_path": "metrics.pkl",           # pickle con métricas

    # Concurrencia
    "download_threads_baseline": 1,          # para comparar
    "download_threads_fast": 4,              # requerido por enunciado
    "ndwi_processes": 4,                     # multiprocessing

    # Para extrapolaciones
    "months_for_extrapolation": 6,           # 1º semestre 2025
}


# =========================
# UTILIDADES
# =========================

def ensure_dirs(*paths: str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_aoi(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Carga AOI desde URL o fichero local (GeoJSON)."""
    if cfg["aoi_geojson_local"]:
        with open(cfg["aoi_geojson_local"], "r", encoding="utf-8") as f:
            return json.load(f)
    # desde URL
    r = requests.get(cfg["aoi_geojson_url"], timeout=30)
    r.raise_for_status()
    return r.json()


def stac_search(
    stac_url: str,
    collection: str,
    aoi_geojson: Dict[str, Any],
    date_range: List[str],
    bands: List[str],
    limit: int = 200
) -> List[Dict[str, Any]]:
    """
    Busca ítems Sentinel-2 L2A intersectando el AOI y rango de fechas.
    Devuelve la lista completa (paginando).
    """
    endpoint = f"{stac_url}/search"
    payload = {
        "collections": [collection],
        "intersects": aoi_geojson["features"][0]["geometry"] if "features" in aoi_geojson else aoi_geojson["geometry"],
        "datetime": f"{date_range[0]}T00:00:00Z/{date_range[1]}T00:00:00Z",
        "limit": min(limit, 200),
        "fields": {
            "include": [
                "id", "assets", "properties.platform", "properties.eo:cloud_cover",
                "properties.s2:cloud_cover", "properties.datetime"
            ],
            "exclude": ["links"]
        }
    }

    items: List[Dict[str, Any]] = []
    next_token = None

    while True:
        if next_token:
            payload["next"] = next_token
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("features", [])
        items.extend(batch)
        next_token = data.get("links", [{}])[-1].get("next", None)
        # earth-search usa campo 'links' con rel=next; fallback:
        if "links" in data:
            for lk in data["links"]:
                if lk.get("rel") == "next":
                    next_token = lk.get("body", {}).get("next", None)
        if not next_token or not batch:
            break

    # Filtra solo los que realmente tienen las bandas requeridas
    def has_bands(it):
        a = it.get("assets", {})
        return all(b in a for b in bands)

    items = [it for it in items if has_bands(it)]
    return items


def get_cloud_percent(props: Dict[str, Any]) -> Optional[float]:
    """Extrae cloud cover tolerantemente de propiedades STAC."""
    for key in ("eo:cloud_cover", "s2:cloud_cover"):
        if key in props and props[key] is not None:
            try:
                return float(props[key])
            except Exception:
                pass
    return None


def split_cloudy(items: List[Dict[str, Any]], max_cloud: Optional[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separa en (sin_nube, con_nube) según el umbral. Si max_cloud es None, no filtra."""
    if max_cloud is None:
        return (items, items)  # sin filtro vs “con nubes” no tiene sentido, pero mantenemos interfaz
    ok, ko = [], []
    for it in items:
        c = get_cloud_percent(it.get("properties", {}))
        if c is not None and c <= max_cloud:
            ok.append(it)
        else:
            ko.append(it)
    return ok, ko


def summarize_by_platform(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """Cuenta ítems por plataforma (A/B/C)."""
    counts: Dict[str, int] = {}
    for it in items:
        p = it.get("properties", {}).get("platform", "")
        # Normaliza: 'sentinel-2a' -> 'S2A'
        tag = ""
        if isinstance(p, str) and "2" in p:
            if "a" in p.lower():
                tag = "S2A"
            elif "b" in p.lower():
                tag = "S2B"
            elif "c" in p.lower():
                tag = "S2C"
        counts[tag or (p or "unknown")] = counts.get(tag or (p or "unknown"), 0) + 1
    return counts


def _download_one(url: str, out_path: Path) -> int:
    """Descarga un archivo y devuelve su tamaño en bytes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path.stat().st_size


def download_items_bands(
    items: List[Dict[str, Any]],
    bands: List[str],
    out_dir: Path,
    n_threads: int = 4
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Descarga concurrente de bandas para cada ítem.
    Devuelve (lista de descriptores con rutas locales, total_bytes).
    """
    tasks = []
    total_bytes = 0
    results: List[Dict[str, Any]] = []

    def plan_item(item: Dict[str, Any]) -> Dict[str, Any]:
        iid = item["id"]
        assets = item["assets"]
        local_paths = {}
        for b in bands:
            href = assets[b]["href"]
            out_path = out_dir / iid / f"{iid}_{b}.tif"
            local_paths[b] = {"url": href, "path": out_path}
        return {"id": iid, "paths": local_paths, "properties": item.get("properties", {})}

    planned = [plan_item(it) for it in items]

    with ThreadPoolExecutor(max_workers=n_threads) as exe:
        fut2meta = {}
        for meta in planned:
            for b, info in meta["paths"].items():
                fut = exe.submit(_download_one, info["url"], info["path"])
                fut2meta[fut] = (meta, b)

        for fut in tqdm(as_completed(fut2meta), total=len(fut2meta), desc=f"Descargando ({n_threads} hilos)"):
            size = fut.result()
            total_bytes += size

    # devolver estructuras con rutas listas
    for meta in planned:
        results.append(meta)
    return results, total_bytes


def open_rgb_stack(item_desc: Dict[str, Any]) -> np.ndarray:
    """Abre y apila R,G,B normalizados para visualización rápida."""
    def _read_norm(pth: Path) -> np.ndarray:
        with rasterio.open(pth) as src:
            arr = src.read(1).astype("float32")
        mn, mx = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
        if mx <= mn:
            mx = mn + 1e-6
        arr = np.clip((arr - mn) / (mx - mn), 0, 1)
        return arr

    r = _read_norm(item_desc["paths"]["red"]["path"])
    g = _read_norm(item_desc["paths"]["green"]["path"])
    b = _read_norm(item_desc["paths"]["blue"]["path"])
    return np.dstack([r, g, b])


def show_and_save_rgb(title: str, rgb: np.ndarray, out_png: Path):
    plt.figure(figsize=(7, 7))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.show()


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = green + nir
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / np.where(denom == 0, np.nan, denom)
    ndwi = np.where(np.isnan(ndwi), 0.0, ndwi).astype("float32")
    return ndwi


def _ndwi_worker(green_path: str, nir_path: str, out_tif: str):
    with rasterio.open(green_path) as gsrc:
        g = gsrc.read(1).astype("float32")
        prof = gsrc.profile.copy()
    with rasterio.open(nir_path) as nsrc:
        n = nsrc.read(1).astype("float32")
    ndwi = calculate_ndwi(g, n)
    prof.update(count=1, dtype="float32", compress="DEFLATE")
    with rasterio.open(out_tif, "w", **prof) as dst:
        dst.write(ndwi, 1)


def process_ndwi(items_desc: List[Dict[str, Any]], out_dir: Path, n_procs: int = 4) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for meta in items_desc:
        iid = meta["id"]
        green_path = str(meta["paths"]["green"]["path"])
        nir_path = str(meta["paths"]["nir"]["path"])
        out_tif = str(out_dir / f"{iid}_ndwi.tif")
        jobs.append((green_path, nir_path, out_tif))

    outs = []
    with ProcessPoolExecutor(max_workers=n_procs) as exe:
        futs = [exe.submit(_ndwi_worker, *j) for j in jobs]
        for fut, j in tqdm(zip(futs, jobs), total=len(jobs), desc=f"NDWI ({n_procs} procesos)"):
            fut.result()
            outs.append(Path(j[2]))
    return outs


def human_size(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    num = float(nbytes)
    while num >= 1024 and i < len(units)-1:
        num /= 1024.0
        i += 1
    return f"{num:.2f} {units[i]}"


def measure_download(items: List[Dict[str, Any]], bands: List[str], base_dir: Path, threads: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Descarga midiendo tiempo y tamaño totales."""
    t0 = time.time()
    descs, total_bytes = download_items_bands(items, bands, base_dir, n_threads=threads)
    t1 = time.time()
    dt = t1 - t0
    bandwidth = (total_bytes / dt) if dt > 0 else 0
    metrics = {
        "threads": threads,
        "elapsed_s": dt,
        "total_bytes": total_bytes,
        "bandwidth_Bps": bandwidth
    }
    return descs, metrics


# =========================
# MAIN PIPELINE
# =========================

def main():
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    out_root = Path(CFG["out_dir"])
    dldir_base = out_root / CFG["download_dir"]
    ndwi_dir = out_root / CFG["ndwi_dir"]
    fig_dir = out_root / CFG["fig_dir"]
    ensure_dirs(out_root, dldir_base, ndwi_dir, fig_dir)

    print("1) Cargando AOI…")
    aoi = load_aoi(CFG)

    print("2) Buscando ítems en STAC…")
    items_all = stac_search(
        CFG["stac_url"], CFG["collection"], aoi, CFG["date_range"], CFG["bands"]
    )
    print(f"   Ítems encontrados con bandas {CFG['bands']}: {len(items_all)}")
    platforms_all = summarize_by_platform(items_all)
    print("   Por satélite (sin filtrar nubes):", platforms_all)

    # Filtrado por nubes
    items_ok, items_cloudy = split_cloudy(items_all, CFG["max_cloud_percent"])
    print(f"   Ítems con nubes <= {CFG['max_cloud_percent']}%: {len(items_ok)}")
    print("   Por satélite (filtrados):", summarize_by_platform(items_ok))

    # 3) Descarga: comparar 1 hilo vs 4 hilos (mismo conjunto filtrado)
    print("\n3) Descarga baseline (1 hilo)…")
    dldir_1 = dldir_base / "threads_1"
    descs_1, m1 = measure_download(items_ok, CFG["bands"], dldir_1, CFG["download_threads_baseline"])
    print(f"   Tiempo: {m1['elapsed_s']:.1f}s | Tamaño: {human_size(m1['total_bytes'])} | BW: {human_size(m1['bandwidth_Bps'])}/s")

    print("   Descarga rápida (4 hilos)…")
    dldir_4 = dldir_base / "threads_4"
    descs_4, m4 = measure_download(items_ok, CFG["bands"], dldir_4, CFG["download_threads_fast"])
    print(f"   Tiempo: {m4['elapsed_s']:.1f}s | Tamaño: {human_size(m4['total_bytes'])} | BW: {human_size(m4['bandwidth_Bps'])}/s")

    # 4) Visualización RGB: coger un ejemplo cloudless y uno cloudy (si existen)
    print("\n4) Visualizaciones RGB…")
    if descs_4:
        try:
            rgb_ok = open_rgb_stack(descs_4[0])
            show_and_save_rgb("RGB (cloud-filtered)", rgb_ok, fig_dir / "rgb_cloudless.png")
        except Exception as e:
            print("   Aviso RGB cloudless:", e)
    if items_cloudy:
        # descargar 1 escena nublada para comparar (si no se descargó antes)
        print("   Descargando una escena nublada para comparación…")
        desc_cloudy, _ = measure_download(items_cloudy[:1], CFG["bands"], dldir_base / "cloudy_example", CFG["download_threads_fast"])
        try:
            rgb_cloudy = open_rgb_stack(desc_cloudy[0])
            show_and_save_rgb("RGB (cloudy)", rgb_cloudy, fig_dir / "rgb_cloudy.png")
        except Exception as e:
            print("   Aviso RGB cloudy:", e)

    # 5) NDWI con multiprocessing sobre las descargas de 4 hilos
    print("\n5) NDWI (multiprocessing)…")
    t0 = time.time()
    ndwi_paths = process_ndwi(descs_4, ndwi_dir, n_procs=CFG["ndwi_processes"])
    t1 = time.time()
    print(f"   NDWI calculado para {len(ndwi_paths)} escenas en {t1 - t0:.1f}s (proc={CFG['ndwi_processes']})")

    # 6) Extrapolaciones (muy simples): tamaño/tiempo a N meses
    print("\n6) Métricas y extrapolaciones…")
    months = CFG["months_for_extrapolation"]
    # Tomamos como base lo filtrado y descargado; si el rango ya es de 6 meses, esto simplemente reporta.
    total_size_6m = m4["total_bytes"]
    time_6m = m4["elapsed_s"]
    # Por mes (promedio)
    size_per_month = total_size_6m / months if months else total_size_6m
    time_per_month = time_6m / months if months else time_6m
    # “Extrapolar a N meses” → aquí N = months (ya lo es); puedes cambiarlo si el profe pide otro N
    metrics = {
        "items_total": len(items_all),
        "items_filtered_cloud": len(items_ok),
        "platform_counts_all": platforms_all,
        "platform_counts_filtered": summarize_by_platform(items_ok),
        "download_metrics_threads_1": m1,
        "download_metrics_threads_4": m4,
        "ndwi_count": len(ndwi_paths),
        "months": months,
        "size_total_bytes_range": total_size_6m,
        "time_total_seconds_range": time_6m,
        "size_per_month_bytes": size_per_month,
        "time_per_month_seconds": time_per_month,
    }

    # 7) Guardar métricas y ejemplo de NDWI colormap
    with open(out_root / CFG["metrics_path"], "wb") as f:
        pickle.dump(metrics, f)

    print("   Guardado metrics.pkl y figuras en:", out_root.resolve())

    # Visualizar un NDWI
    if ndwi_paths:
        try:
            with rasterio.open(ndwi_paths[0]) as src:
                ndwi = src.read(1)
            plt.figure(figsize=(7, 7))
            plt.imshow(ndwi, cmap="RdYlGn", vmin=-1, vmax=1)
            plt.colorbar(label="NDWI")
            plt.title("NDWI (ejemplo)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fig_dir / "ndwi_example.png", dpi=180)
            plt.show()
        except Exception as e:
            print("   Aviso NDWI preview:", e)

    print("\n✅ Pipeline completado. Sube /outputs_session5 a tu repo GitHub.")


if __name__ == "__main__":
    main()
