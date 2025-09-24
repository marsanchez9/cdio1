import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pytest

def compute_ndwi(green, nir):
    """
    Compute NDWI = (Green - NIR) / (Green + NIR).
    Handles division by zero and NaNs.
    """
    ndwi = np.where(
        (green + nir) == 0,
        0,
        (green - nir) / (green + nir)
    )
    # Tractar valors NaN resultants
    ndwi = np.nan_to_num(ndwi, nan=0.0)  #cambia los np.nan por 0
    return ndwi

def main():
    # --- RUTES ALS FITXERS ---
    nir_path = r"C:\Users\ninac\Desktop\test_data\S2C_31TDF_20250630_0_L2A_nir.tif"
    green_path = r"C:\Users\ninac\Desktop\test_data\S2C_31TDF_20250630_0_L2A_green.tif"

    # --- CARREGAR LES BANDES ---
    with rasterio.open(nir_path) as nir_src, rasterio.open(green_path) as green_src:
        nir = nir_src.read(1).astype("float32")
        green = green_src.read(1).astype("float32")

    # --- NDWI ---
    ndwi = compute_ndwi(green, nir)

    # --- MOSTRAR ---
    plt.figure(figsize=(8, 6))
    ndwi_plot = plt.imshow(ndwi, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(ndwi_plot, label="NDWI")
    plt.title("NDWI (Normalized Difference Water Index)")
    plt.axis("off")
    plt.show()

# -----------------
# TESTS
# -----------------
@pytest.mark.parametrize("green, nir, expected", [
    (np.array([10.0]), np.array([5.0]), np.array([0.3333])),  # cas normal
    (np.array([0.0]), np.array([0.0]), np.array([0.0])),      # divisió per zero
    (np.array([5.0]), np.array([5.0]), np.array([0.0])),      # green = nir
    (np.array([np.nan]), np.array([5.0]), np.array([0.0])),   # valor NaN
])                                          #Hay más errores potenciales, pero de rastrerio, memoria de RAM... no tiene mucho que ver con la imagen
def test_ndwi(green, nir, expected):
    result = compute_ndwi(green, nir)
    np.testing.assert_allclose(result, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    main() 
