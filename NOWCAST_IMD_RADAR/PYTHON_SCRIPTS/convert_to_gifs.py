import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import imageio.v2 as imageio
import cv2

# === USER CONFIGURATION ===
input_files = [
    "interpolated_forcasted_outputs_only_2/caz_vsk_25Jun2025_1530_input.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1540_interpolated.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1550_interpolated.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1600_interpolated.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1610_interpolated.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1620_interpolated.tif",
    "interpolated_forcasted_outputs_only_2/caz_vsk_25Jun2025_1630_input.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1640_forecast.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1700_forecast.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1650_forecast.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1710_forecast.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1720_forecast.tif",
    "interpolated_mean_outputs/mean_interpolation/caz_vsk_25Jun2025_1730_forecast.tif"
]

vmin = 0
vmax = 60
output_gif = "mean.gif"
frame_duration = 0.8  # seconds per frame

# === Add timestamp and label text overlays ===
def add_overlay(image, timestamp, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = image.copy()
    # Timestamp (bottom-left)
    cv2.putText(img, timestamp, (10, 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # Label (top-left)
    cv2.putText(img, label, (10, 55), font, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return img

# === Load and Colorize with Label ===
def load_colorize_label(tif_path, index):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("turbo")  # Use a perceptually uniform colormap
    rgb = cmap(norm(arr))[:, :, :3]
    rgb = (rgb * 255).astype(np.uint8)

    # Extract timestamp
    timestamp = os.path.basename(tif_path).split("_")[-2]

    # Determine label
    if index in [0, 6]:
        label = "Original"
    elif 1 <= index <= 5:
        label = "Interpolated"
    else:
        label = "Forecast"

    return add_overlay(rgb, timestamp, label)

# === Main Process ===
print("🔄 Creating GIF with labels...")

frames = []
for idx, tif_file in enumerate(input_files):
    print(f"▶️ Processing: {tif_file}")
    frame = load_colorize_label(tif_file, idx)
    frames.append(frame)

# Save the final GIF
imageio.mimsave(output_gif, frames, duration=frame_duration)
print(f"✅ GIF saved to: {output_gif}")



