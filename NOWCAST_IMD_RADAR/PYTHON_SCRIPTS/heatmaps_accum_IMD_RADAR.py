import os
import shutil
from datetime import datetime, timedelta, time
import time

# import time
import pandas as pd
#from datetime import datetime, timedelta, time
# from pysteps import nowcasts
import xarray as xr
import numpy as np
# import time
import matplotlib.pyplot as plt
import rioxarray
import rasterio
# from rasterio.enums import Resampling
# from cdo import *
import cartopy.crs as ccrs
import glob
# import pysteps
# from pysteps.datasets import load_dataset
# from pysteps import motion
from PIL import Image
import imageio
import matplotlib.colors as mcolors
import imageio
import os
from rasterio.enums import Resampling
from shapely.errors import TopologicalError
from shapely.geometry import Polygon
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import sys
from multiprocessing import Pool


start = time.time()
# Step - 4 - use env met_work3

#To convert .tiff to .png (accum tiff files)

#system_time = datetime.now()

#python heatmaps_accum_IMD_RADAR.py "$(date +'%d%b%Y_%H%M')"

if len(sys.argv) != 2:
    print("Usage: python heatmaps_MLP.py <system_time>")
    sys.exit(1)

system_time_str = sys.argv[1]
#print('current', current_time_str)

try:
    system_time = datetime.strptime(system_time_str, "%d%b%Y_%H%M")
    print("Received time:", system_time)
except ValueError as e:
    print("Error parsing date:", e)
    sys.exit(1)

#system_time = datetime(2024, 12, 22, 9, 25, 00)

def pixel_to_coords(row, col, transform):
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y

def handle_repeated_values(tiff_data, threshold):
    unique, counts = np.unique(tiff_data, return_counts=True)
    value_counts = dict(zip(unique, counts))
    repeated_values = [value for value, count in value_counts.items() if count > threshold]
    for value in repeated_values:
        tiff_data[tiff_data == value] = np.nan
    return tiff_data

#coll = ["#e6eeff", "#80aaff", "#3377ff", "#003cb3", "#002266", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]

#no of colors = no of commas or (no of breaks - 1) 
def get_heatmap_params(filename):
    if "1HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"], [0, 1, 4, 16, 30, 50, 100, 1000], 'Accumulated map(1HR) in mm'
    elif "2HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"], [0, 1.5, 6, 24, 44.9, 75.1, 147.5, 1000], 'Accumulated map(2HR) in mm'
    elif "3HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"], [0, 1.7, 7, 28, 52.2, 87.8, 168.8, 1000], 'Accumulated map(3HR) in mm'
    elif "6HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"],  [0.1, 2, 8.9, 36.1, 66.7, 113.3, 206.3, 1000], 'Accumulated map(6HR) in mm'
    elif "12HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"], [0.2, 2.4, 12.2, 49.8, 91, 157, 265.5, 1000], 'Accumulated map(12HR) in mm'
    elif "24HR" in filename:
        return ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"], [0.3, 2.5, 15.6, 64.5, 115.6, 204.5, 300, 1000], 'Accumulated map(24HR) in mm'
    else:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

def heatmap_tiff(input_tiff, output_folder, coll, brk, sufix, img_extent, units, threshold=1000):
    tiff_image = imageio.imread(input_tiff)
    tiff_data = np.array(tiff_image)
    tiff_data = np.ma.masked_invalid(tiff_data)
    tiff_data = np.ma.masked_equal(tiff_data, 255)
    tiff_data = handle_repeated_values(tiff_data, threshold)
    with rasterio.open(input_tiff) as src:
        transform = src.transform
    lats, lons = [], []
    rows, cols = tiff_data.shape[:2]
    for col in range(cols):
        lon, _ = pixel_to_coords(0, col, transform)
        lons.append(lon)
    for row in range(rows):
        _, lat = pixel_to_coords(row, 0, transform)
        lats.append(lat)
    dpi = 700
    proj = ccrs.Mercator.GOOGLE
    fig = plt.figure(frameon=False, dpi=dpi, facecolor='white')
    ax = plt.axes(projection=proj, facecolor='white')
    ax.set_extent(img_extent, ccrs.PlateCarree())
    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)
    np_array = tiff_data.data
    np_array[np.isnan(np_array)] = -999
    contour = ax.contourf(lons, lats, np_array, levels=brk, colors=coll, alpha=1, corner_mask=True, extend="neither",
                          transform=ccrs.PlateCarree())
    output_filename = os.path.join(output_folder, sufix + '.png')
    fig.savefig(output_filename, bbox_inches='tight', facecolor='white', edgecolor='white', pad_inches=0.0)
    plt.close()
    tiff_image = Image.open(output_filename)
    tiff_image.save(output_filename, 'PNG')
    print(f"Heatmap saved as {output_filename}")

def process_tiff_folder(folder_path, output_folder, current_time):
    tasks = []
    for tiff_file in os.listdir(folder_path):
        if tiff_file.endswith('.tif'):
            tiff_path = os.path.join(folder_path, tiff_file)
            try:
                coll, brk, units = get_heatmap_params(tiff_file)
                print(f"tiff_file {tiff_file}")
                sufix = os.path.splitext(tiff_file)[0].split('_')[2]
                print(f"suffix {sufix}")
                img_extent = [67.4, 97.1, 6.3, 34.8]
                tasks.append((tiff_path, output_folder, coll, brk, sufix, img_extent, units))
            except ValueError as e:
                print(e)
    with Pool(processes=len(tasks)) as pool:
        pool.starmap(heatmap_tiff, tasks)

search_directory = os.path.expanduser("/data/NOWCAST_IMD_RADAR/accum_radar_data/")
output_directory = os.path.expanduser("/data/NOWCAST_IMD_RADAR/accum_heatmaps/")

current_time = system_time - timedelta(minutes=5)
minute = (current_time.minute // 5) * 5
current_time = current_time.replace(minute=minute, second=0, microsecond=0)
current_time = current_time.strftime("%d%b%Y_%H%M")
print(current_time)

for folder_name in os.listdir(search_directory):
    folder_path = os.path.join(search_directory, folder_name)
    if os.path.isdir(folder_path) and current_time in folder_name:
        output_folder = os.path.join(output_directory, folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        process_tiff_folder(folder_path, output_folder, current_time)

print('done') 
end = time.time()
print('time taken for heatmaps:', (end-start)/60, 'minutes')
