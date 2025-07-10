import os

# === Paths ===
gif_dir = "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1130"     # folder with .gif files
tif_dir = "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1130/Processing_results/clipped_tifs"     # folder with .tif files

# === Step 1: Get unique prefixes from .gif files (e.g., 'caz_lkn') ===
gif_prefixes = set()
for gif_file in os.listdir(gif_dir):
    if gif_file.endswith(".gif"):
        prefix = "_".join(gif_file.split("_")[:2])  # 'caz_lkn_25Jun2025_1042.gif' -> 'caz_lkn'
        gif_prefixes.add(prefix)

# === Step 2: Loop through .tif files and check if prefix matches ===
for tif_file in os.listdir(tif_dir):
    if tif_file.endswith(".tif"):
        prefix = "_".join(tif_file.split("_")[:2])  # 'caz_lkn_25Jun2025_1042_extract_...tif' -> 'caz_lkn'
        if prefix not in gif_prefixes:
            # Delete unmatched .tif file
            full_path = os.path.join(tif_dir, tif_file)
            os.remove(full_path)
            print(f"❌ Deleted unmatched: {tif_file}")
        else:
            print(f"✅ Kept: {tif_file}")
