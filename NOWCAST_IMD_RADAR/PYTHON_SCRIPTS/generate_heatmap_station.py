import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import re
from multiprocessing import Pool
import time
# === CONFIG ===
forecast_dir = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/STACKED_MOSAICS/forecast_mosaics"
observed_dir = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/STACKED_MOSAICS/observed_mosaics"
radar_gap_mask_path = "radar_gap_mask.tif"

bbox = (65.025, 100.025, 5.025, 39.975)
dpi = 700
proj = ccrs.Mercator.GOOGLE

colors = ["#e6eeff", "#80aaff", "#3377ff", "#003cb3", "#002266",
          "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]
brk = [0.5, 2.5, 5, 10, 20, 30, 40, 50, 75, 100, 1000]

new_colors = ['#A0A0A0', '#ffffff', '#fffebd', '#fffe02', '#02c3ff',
              '#30982f', '#f95c05', '#720101']
new_breaks = [-1.5, -0.5, 0.1, 2.4, 15.6, 64.4, 115.5, 204.4, 1000]

# === OUTPUT DIRS ===
forecast_out = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/forecast_png"
observed_out = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/observed_png"
aware_out = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/aware"

os.makedirs(forecast_out, exist_ok=True)
os.makedirs(observed_out, exist_ok=True)
os.makedirs(aware_out, exist_ok=True)


def extract_datetime_from_filename(tif_path):
    basename = os.path.basename(tif_path)
    match = re.search(r'(\d{2}[A-Za-z]{3}\d{4})_(\d{4})', basename)
    if match:
        date_part, time_part = match.groups()
        try:
            return pd.to_datetime(date_part + time_part, format="%d%b%Y%H%M")
        except Exception as e:
            print(f"⚠️ Error parsing datetime from {basename}: {e}")
    else:
        print(f"⚠️ No datetime found in {basename}")
    return pd.Timestamp.now()


def plot_tiff(tif_path, sufix, output_dir):
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            width = src.width
            height = src.height

        lon = np.linspace(transform[2], transform[2] + transform[0] * width, width)
        lat = np.linspace(transform[5], transform[5] + transform[4] * height, height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        dt = extract_datetime_from_filename(tif_path)
        time_str = dt.strftime('%Y%m%d%H%M_IST')

        # Plot original
        fig = plt.figure(frameon=False, dpi=dpi)
        ax = plt.axes(projection=proj)
        ax.set_extent(bbox, ccrs.PlateCarree())
        ax.set_frame_on(False)
        ax.contourf(lon_grid, lat_grid, data, levels=brk, colors=colors,
                    alpha=1, corner_mask=True, extend="neither", transform=ccrs.PlateCarree())
        out_path = os.path.join(output_dir, f"{sufix}_{time_str}.png")
        fig.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
        plt.close()
        print(f"✅ Saved: {out_path}")

        # Plot masked AWARE
        if radar_gap_mask_path and os.path.exists(radar_gap_mask_path):
            with rasterio.open(radar_gap_mask_path) as src:
                radar_mask = src.read(1)

            if radar_mask.shape != data.shape:
                pad_rows = data.shape[0] - radar_mask.shape[0]
                pad_cols = data.shape[1] - radar_mask.shape[1]
                radar_mask = np.pad(radar_mask, ((0, pad_rows), (0, pad_cols)), constant_values=0)

            masked_data = data.astype(np.float32).copy()
            nan_mask = np.isnan(masked_data)
            masked_data[(masked_data < 0.04) & (radar_mask == 1) & (~nan_mask)] = -1
            masked_data[nan_mask & (radar_mask == 0)] = 0
            masked_data[nan_mask & (radar_mask == 1)] = np.nan

            fig2 = plt.figure(frameon=False, dpi=dpi)
            ax2 = plt.axes(projection=proj)
            ax2.set_extent(bbox, ccrs.PlateCarree())
            ax2.set_frame_on(False)
            ax2.contourf(lon_grid, lat_grid, np.ma.masked_invalid(masked_data),
                         levels=new_breaks, colors=new_colors, alpha=1,
                         extend='max', transform=ccrs.PlateCarree())

            aware_path = os.path.join(aware_out, f"{sufix}_{time_str}.png")
            fig2.savefig(aware_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
            plt.close()
            print(f"🎯 Aware Masked: {aware_path}")
    except Exception as e:
        print(f"❌ Failed processing {tif_path}: {e}")


def run_parallel_tiff_processing(directory, sufix, output_dir, num_processes=4):
    tif_list = sorted([
        (os.path.join(directory, tif), sufix, output_dir)
        for tif in os.listdir(directory) if tif.endswith(".tif")
    ])
    with Pool(processes=num_processes) as pool:
        pool.starmap(plot_tiff, tif_list)


if __name__ == "__main__":
    st = time.time()
    print("📡 Processing Forecast TIFFs in parallel...")
    run_parallel_tiff_processing(forecast_dir, "forecast", forecast_out, num_processes=4)

    print("🌧️ Processing Observed TIFFs in parallel...")
    run_parallel_tiff_processing(observed_dir, "observed", observed_out, num_processes=4)
    et = time.time()
    print(f"⏱️ Total processing time: {et - st:.2f} seconds")
    print("✅ All tasks completed successfully!")
