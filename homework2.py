import matplotlib.pyplot as plt
import rasterio
import numpy as np

nir_path = r"C:\Users\ninac\Desktop\test_data\S2C_31TDF_20250630_0_L2A_nir.tif"
green_path = r"C:\Users\ninac\Desktop\test_data\S2C_31TDF_20250630_0_L2A_green.tif"

# Obrim les bandes
with rasterio.open(nir_path) as nir_src, rasterio.open(green_path) as green_src:
    nir = nir_src.read(1).astype("float32")
    green = green_src.read(1).astype("float32")

# Càlcul NDWI, controlant divisions per zero
ndwi = np.where(
    (green + nir) == 0,
    0,
    (green - nir) / (green + nir)
)

# Mostrem la imatge NDWI
plt.figure(figsize=(8, 6))
ndwi_plot = plt.imshow(ndwi, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(ndwi_plot, label="NDWI")
plt.title("NDWI (Normalized Difference Water Index)")
plt.axis("off")
plt.show()
