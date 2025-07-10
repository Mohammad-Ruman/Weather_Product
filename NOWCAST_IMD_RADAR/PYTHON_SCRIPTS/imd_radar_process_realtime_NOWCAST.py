import imageio
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import os
import csv
import rioxarray
from sklearn.svm import SVC
from scipy.spatial import distance
from rasterio.mask import mask
from shapely.geometry import shape
import fiona
import matplotlib.pyplot as plt
import pyproj 
from numpy import *
import pytesseract
from datetime import datetime,timedelta
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image, ImageSequence, UnidentifiedImageError
from io import BytesIO
import time
from shutil import move


import rasterio
from rasterio.transform import from_origin
#from osgeo import ogr
import fiona
#import geopandas as gpd
import os
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.plot import show
from pathlib import Path
import requests
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import shutil
import urllib
import glob
from rasterio.warp import calculate_default_transform, reproject
import imageio
import numpy as np
import rioxarray
import pandas as pd
from sklearn.svm import SVC
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import time
from rasterio.features import geometry_mask
from rasterio import Affine
from datetime import datetime, timedelta
import logging
import psutil
from rasterio import warp
from rasterio.warp import reproject, Resampling
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor

from os import cpu_count
from constants import REGION_DATE_FORMAT_DICT, REGION_NAME_POSSIBLE_DATE_FORMATS


if len(sys.argv) != 2:
    print("Usage: python imd_radar_process_realtime_NOWCAST.py <system_time>")
    sys.exit(1)

system_time_str = sys.argv[1]

try:
    system_time = datetime.strptime(system_time_str, "%d%b%Y_%H%M")
    print("Received time:", system_time)
except ValueError as e:
    print("Error parsing date:", e)
    sys.exit(1)


#FOR ALL REGIONS AT A TIME - DOWNLOAD AND RENAME
start2 = time.time()
excel_path = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/29regions_info.xlsx")
df_regions = pd.read_excel(excel_path)
parent_dir = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/")
#parent_dir = os.path.expanduser("~/downloads/RADAR_DATA_MLP_NEW_MODEL")

#system_time = datetime.now()
print(system_time)

nearest_hour = (system_time.minute // 10) * 10
current_time = system_time.replace(minute=nearest_hour, second=0, microsecond=0)
print("Adjusted datetime:", current_time)

big_folder = current_time.strftime("%d%b%Y_%H%M")
filename_datetime = str(big_folder)
folder_name = 'Realtime_radar_INDIA_' + filename_datetime
#folder_name = 'Realtime_radar_INDIA_Allreg' + filename_datetime

# text_file_name = f"Time_{big_folder}.txt"
# file_path = os.path.expanduser("~/gifs/{text_file_name}")

# with open(file_path, "w") as file:
#     file.write(filename_datetime)


# folder_name = 'realtime_gifs_FINAL_trial_25reg_01Apr2240' 
output_folder = os.path.join(parent_dir, folder_name)
os.makedirs(output_folder, exist_ok=True)

def download_gif(url, output_folder):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Extract the filename from the URL
        filename = url.split("/")[-1]

        # Save the GIF to the specified output folder
        output_path = os.path.join(output_folder, filename)
        with open(output_path, 'wb') as f:
            f.write(response.content)

        return output_path
    except Exception as e:
        print(f"Error downloading GIF from {url}: {e}")
        return None

# Function to check if a GIF contains problem keywords
def is_problem_gif(gif_path, keywords):
    try:
        gif = Image.open(gif_path)
        text = pytesseract.image_to_string(gif)
        print(f"Extracted text from {gif_path}: {text}")
        for keyword in keywords:
            if keyword in text.lower():
                return True
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error: Unable to identify image file {gif_path}: {e}")
    return False

def rename_gif(text, string_format, gif_url, downloaded_path):
    try:
        datetime_object = datetime.strptime(text.lower(), string_format)
        datetime_str = (datetime_object + timedelta(hours=5.5)).strftime("%d%b%Y_%H%M")
        if datetime_str:
           gif_name = gif_url.split("/")[-1].split(".")[0]
           new_filename = f"{gif_name}_{datetime_str}.gif"
           new_path = os.path.join(output_folder, new_filename)
           os.rename(downloaded_path, new_path)
           return True
    except ValueError as e:
        return False

def move_to_problem_gifs(gif_file_path, output_folder):
    gif_name = gif_file_path.split("/")[-1].split(".")[0]
    problem_gifs_folder = os.path.join(output_folder, "Problem_gifs")
    os.makedirs(problem_gifs_folder, exist_ok=True)
    move(gif_file_path, os.path.join(problem_gifs_folder, f"{gif_name}.gif"))

def download_and_rename_gif(row, output_folder):
    gif_url = row['gif_url']
    bbox_crop = (row['x1'], row['y1'], row['x2'], row['y2'])
    region_name = row['Region_name']
    downloaded_path = download_gif(gif_url, output_folder)
    if downloaded_path is None:
        return None
    try:
        gif = Image.open(downloaded_path)
        frame = gif.copy().convert('LA')
        cropped_img = frame.crop(bbox_crop)
        possible_formats = []
        string_format = REGION_DATE_FORMAT_DICT[region_name]
        if region_name in REGION_NAME_POSSIBLE_DATE_FORMATS:
            possible_formats = REGION_NAME_POSSIBLE_DATE_FORMATS[region_name]
        possible_formats.insert(0, string_format)
        is_renaming_success = False
        try:
            text = pytesseract.image_to_string(cropped_img)
            text = text.replace('NOY', 'NOV')
            text = text.replace('noy', 'nov')
            text = text.replace('no¥', 'nov')
            text = text.replace('NO¥', 'NOV')
            if "45 nov" or "45 NOV" in text:
                text = text.replace("45 nov", "15 nov")
                text = text.replace("45 NOV", "15 NOV")

            if "1 dec" or "1 DEC" in text: 
                text = text.replace("1 dec", "21 dec") 
                text = text.replace("1 DEC", "21 DEC")
            for date_format in possible_formats:
                is_renaming_success = rename_gif(text, date_format, gif_url, downloaded_path)
                if is_renaming_success:
                    break
        except Exception as e:
            pass

        if not is_renaming_success:
            print(f"Unable to extract datetime with any of the provided formats for region {region_name}")
            move_to_problem_gifs(downloaded_path, output_folder)

    except (UnidentifiedImageError, OSError) as e:
        print(f"Error: Unable to identify image file {downloaded_path}: {e}")
        move_to_problem_gifs(downloaded_path, output_folder)

max_workers = min(8, 2)

def parallel_download_gifs(df_regions, output_folder):
    
    non_folder = os.path.join(output_folder, "Problem_gifs")
    os.makedirs(non_folder, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = []
        for index, row in df_regions.iterrows():
            futures.append(
                executor.submit(download_and_rename_gif, row, output_folder)
            )

        for future in futures:
            try:
                future.result()  
            except Exception as e:
                print(f"Error occurred: {e}")

for index, row in df_regions.iterrows():
    non_folder = os.path.join(output_folder, "Problem_gifs")
    os.makedirs(non_folder, exist_ok=True)
    #download_and_rename_gif(row, output_folder)

download_start_time = time.time()  
parallel_download_gifs(df_regions, output_folder)
download_complete_time = time.time()
print('file download time(in sec) : ', (download_complete_time - download_start_time))

print("checkpoint1")

downloaded_files_folder = output_folder
notmatched_files_folder = os.path.join(downloaded_files_folder, "Not_match")
os.makedirs(notmatched_files_folder, exist_ok=True)

time_buffer = timedelta(minutes=20) # 20 minutes buffer
time_30_minutes_ago = current_time - time_buffer

for filename in os.listdir(downloaded_files_folder):
    if filename.endswith('.gif'):
        filename_parts = filename.split('_')
        date_str = filename_parts[2]  
        time_str = filename_parts[3].split('.')[0] 
        try:
            gif_time = datetime.strptime(date_str + time_str, '%d%b%Y%H%M')  
        except ValueError:
            # to be done, what if there is an error in parsing the time and that gif is left
            print("Skipping file:", filename, "- Unable to parse time from filename.")
            continue
#        if gif_time > time_30_minutes_ago and gif_time < current_time + timedelta(minutes=5):
#            print('Matched')
#        elif gif_time == time_30_minutes_ago:
#            print('Equal to T-30, Matched')
#        elif gif_time == current_time:
#            print('Equal to T, Matched')  
#        else:
#            print('Not matched')
#            src_path = os.path.join(downloaded_files_folder, filename)
#            dest_path = os.path.join(notmatched_files_folder, filename)
#            shutil.move(src_path, dest_path)
#            print("Moved:", filename)
        if gif_time < time_30_minutes_ago or gif_time >= current_time + timedelta(minutes=5):
            src_path = os.path.join(downloaded_files_folder, filename)
            dest_path = os.path.join(notmatched_files_folder, filename)
            shutil.move(src_path, dest_path)



print('done')
# import sys
print("checkpoint2")
#sys.exit(1)


# excel_path = os.path.expanduser("~/gifs/5regions_info_3.xlsx")
# df_regions = pd.read_excel(excel_path)
# parent_dir = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/")
shp_csv_folder_path = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/SHAPE_FILES_CSVS_ALL_REGIONS/")

input_real_folder = output_folder
parent_dir = output_folder

# #change parent_dir to output_folder - in the end, input_real_folder to ouput_folder
# input_real_folder = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/realtime_gifs_FINAL_trial5/trial1")
# parent_dir = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/realtime_gifs_FINAL_trial5")

process_folder = os.path.join(parent_dir, "Processing_results")
os.makedirs(process_folder, exist_ok = True)

png_folder = os.path.join(process_folder, 'only_pngs')
os.makedirs(png_folder, exist_ok = True)

extract_folder = os.path.join(process_folder, "extract_gifs")
os.makedirs(extract_folder, exist_ok = True)

fill_folder = os.path.join(process_folder, 'single_fill_pngs')
os.makedirs(fill_folder, exist_ok = True)

georef_folder = os.path.join(process_folder, "georef_tifs")
os.makedirs(georef_folder, exist_ok=True)

clip_folder = os.path.join(process_folder, "clipped_tifs")
os.makedirs(clip_folder, exist_ok=True)

def convert_gif_to_png(input_path, output_path):
    with Image.open(input_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(output_path, format='PNG')

def extract_image(png_path, csv_file, output_png, region_name):
    with rasterio.open(png_path) as src:
        image = src.read()
        profile = src.profile
        transform = src.transform

    rgb_data = pd.read_csv(csv_file)

    output_image = np.zeros_like(image)
    print(image.shape)

    buffer = 3  # Define the buffer range

    # ref_png = os.path.expanduser("~/downloads/shafeer_/shafeer_/shafeer_recode_binary.img")

    # with rasterio.open(ref_png) as ref_src:
    #     ref_image = ref_src.read(1)  # Assuming the reference PNG is single band

    # # Resize the input image to match the reference image
    # ref_height, ref_width = ref_image.shape
    # input_image_resized = np.array([np.array(Image.fromarray(image_band).resize((ref_width, ref_height), Image.NEAREST)) for image_band in image])

    # # Update output_image with the resized input image
    # output_image = np.zeros_like(input_image_resized)

    # Iterate through each row in the CSV file
    for _, row in rgb_data.iterrows():
        # Extract RGB values
        r, g, b = row['R'], row['G'], row['B']

        # Boolean mask with buffer
        mask = (
            (image[0, :, :] >= r - buffer) & (image[0, :, :] <= r + buffer) &
            (image[1, :, :] >= g - buffer) & (image[1, :, :] <= g + buffer) &
            (image[2, :, :] >= b - buffer) & (image[2, :, :] <= b + buffer)
        )

        # Set the matched pixels to their original values
        output_image[:, mask] = image[:, mask]

        if region_name == 'Kolkata':

            try:

                ref_png = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/KOLKATA_MASK/recode_binary.img")

                with rasterio.open(ref_png) as ref_src:
                    ref_image = ref_src.read(1) 

                output_image[:, ref_image == 1] = 0  # Set to (0, 0, 0)

            except Exception as e:
                print(f"Error processing reference image: {e}")

                try:

                    ref_png = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/KOLKATA_MASK2/radar_new.img")
                    with rasterio.open(ref_png) as ref_src:
                        ref_image = ref_src.read(1)  

                    output_image[:, ref_image == 1] = 0  

                except Exception as e:

                    print('Error processing - try 3rd mask')

                    ref_png = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/KOLKATA_MASK3/kolkata_recode.img")

                    with rasterio.open(ref_png) as ref_src:
                        ref_image = ref_src.read(1) 

                    output_image[:, ref_image == 1] = 0 

        if region_name == 'Vishakapatnam':

            try:

                ref_png = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/VISHAKAPATNAM_MASK/radar_mask_2.img")

                # Read the reference PNG
                with rasterio.open(ref_png) as ref_src:
                    ref_image = ref_src.read(1)  # Assuming the reference PNG is single band

                # Mask the extracted image
                output_image[:, ref_image == 1] = 0  # Set to (0, 0, 0)

            except Exception as e:
                print(f"Error processing reference image: {e}")

    png_transform = from_origin(0, 0, transform.a, transform.e)
    profile.update(transform=png_transform)

    with rasterio.open(output_png, 'w', **profile) as dst:
        dst.write(output_image)

    print(f"Extracted and masked image saved as {output_png}")

def save_numpy_as_png(rgb_array, output_file, origin):
    # Flip the array vertically to handle the origin difference
    rgb_array_flipped = np.flipud(rgb_array)

    # Convert the RGB array to unsigned 8-bit integer format
    rgb_uint8 = rgb_array_flipped.astype(np.uint8)

    # Save the RGB array as PNG using imageio with explicit origin setting
    imageio.imwrite(output_file, rgb_uint8, origin=origin)

def fill_missing_values(grid):
    filled_grid = grid.copy() 

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 0:  # Check if the value is missing (0)
                surrounding_values = []

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip the center pixel
                        ni, nj = i + dx, j + dy

                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            neighbor_value = grid[ni, nj]
                            if neighbor_value != 0:
                                surrounding_values.append(neighbor_value)

                if len(surrounding_values) <= 1:
                    continue
                else:
                    filled_grid[i, j] = max(surrounding_values)

    return filled_grid

def empirical_relation(value):
    return ((10 ** (value / 10)) / 200) ** (5 / 8)

def rgb_to_single_fill(extract_png, csv_file, output_fill_refl_png, output_fill_rainrate_png):
    rds = rioxarray.open_rasterio(extract_png)
    df = rds.rename("dem").to_dataframe().reset_index().dropna()
    df = df.pivot_table(columns="band", index=['y', 'x'], values='dem').reset_index()

    # Load the SVM model
    dataset = pd.read_csv(csv_file)
    X = dataset[['R', 'G', 'B']].values
    y = dataset["Value_int"].values
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X, y)

    # Predict values using the SVM model
    X_pred = df.iloc[:, 2:5].values
    df['val'] = classifier.predict(X_pred)

    grid = df.pivot(index='y', columns='x', values='val').values

    filled_grid = fill_missing_values(grid)

    #To convert reflectivity (dbZ) to rainfall rate (mm/hr)
    # Calculating rainfall rate using updated values
    filled_df = pd.DataFrame({'y': df['y'], 'x': df['x'], 'val': filled_grid.flatten()})
    #filled_df['rainfall_rate'] = filled_df['val'].apply(empirical_relation)

    filled_df['rainfall_rate'] = filled_df['val'].apply(empirical_relation).round().astype(int)


    # Reshape the rainfall rate array back to the grid shape
    filled_rainmap = filled_df.pivot(index='y', columns='x', values='rainfall_rate').values

    # df['rainfall_rate'] = df['val'].apply(empirical_relation)
    # #rainmap = df.pivot('y', 'x', 'rainfall_rate')
    # rainmap = df.pivot(index='y', columns='x', values='rainfall_rate').values

    png_origin = (rds.x.min(), rds.y.max())

    #filled_grid = fill_missing_values(grid)
    #filled_rainmap = fill_missing_values(rainmap)

    save_numpy_as_png(filled_grid, output_fill_refl_png, origin=png_origin)
    save_numpy_as_png(filled_rainmap, output_fill_rainrate_png, origin=png_origin)

def georeference_pngs(input_image_path, output_image_path, final_image_path, gcp):
    gdal_translate_cmd = f'gdal_translate -of GTiff {gcp} {input_image_path} {output_image_path}'
    os.system(gdal_translate_cmd)

    gdalwarp_cmd = f'gdalwarp -r bilinear -order 1 -co COMPRESS=None -t_srs EPSG:4326 "{output_image_path}" "{final_image_path}"'
    os.system(gdalwarp_cmd)

    # gdal_cmd = f'gdal_translate -projwin 74.15 21.5 83 13.15 -of GTiff "{final_image_path}" "{op_image_path}"'
    # os.system(gdal_cmd)

def clip_raster_to_shapefile(shapefile_path, raster_path, output_path):
    with fiona.open(shapefile_path, "r") as shapefile:
        geometry = [feature["geometry"] for feature in shapefile]

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)

    out_meta = src.meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': out_image.shape[1],
                     'width': out_image.shape[2],
                     'transform': out_transform})

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

csv_files = {
    'Bhopal': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Bhopal.csv'),
    'Patiala': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Patiala.csv'),
    'Nagpur': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Nagpur.csv'),
    'Gopalpur': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Gopalpur.csv'),
    'Agartala': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Agartala.csv'),
    'Sohra': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Sohra.csv'),
    'Goa': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Goa.csv'),
    'Mumbai': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Mumbai_old.csv'),
    'Bhuj': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Bhuj.csv'),
    'Hyderabad': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Hyderabad.csv'),
    'Karaikal': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Karaikal.csv'),
    'Kochi': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Kochi.csv'),
    'Thiruvananthapuram': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Thiruvananthapuram.csv'),
    'Paradip': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Paradip.csv'),
    'Kolkata': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Kolkata.csv'),
    'Mohanbari': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Mohanbari.csv'),
    'Patna': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Patna.csv'),
    'Lucknow': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Lucknow.csv'),
    'Delhi': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Delhi.csv'),
    'Chennai': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Chennai_old.csv'),
    'Vishakapatnam': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Vishakapatnam.csv'),
    'Machilipatnam': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Machilipatnam.csv'),
    # 'Sriharikota': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Sriharikota.csv'),
    'Jaipur': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Jaipur.csv'),
    'Srinagar': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Srinagar.csv'),
    'Jot': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Jot.csv'),
    'Mukteshwar': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Mukteshwar.csv'),
    'Lansdowne': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Lansdowne.csv'),
    'Surkandaji': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Surkandaji.csv'),
    'Banihal': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Banihal.csv'),
    'Jammu': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Jammu.csv'),
    'Kufri': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Kufri.csv'),
    'Murari': os.path.join(shp_csv_folder_path, 'RGB_single_refl_Murari.csv'),
}

shapefiles = {
    'Bhopal': os.path.join(shp_csv_folder_path, 'Bhopal_buffer.shp'),
    'Patiala': os.path.join(shp_csv_folder_path, 'Patiala_buffer.shp'),
    'Nagpur': os.path.join(shp_csv_folder_path, 'Nagpur_buffer.shp'),
    'Gopalpur': os.path.join(shp_csv_folder_path, 'Gopalpur_buffer.shp'),
    'Agartala': os.path.join(shp_csv_folder_path, 'Agartala_buffer.shp'),
    'Sohra': os.path.join(shp_csv_folder_path, 'Sohra_buffer.shp'),
    'Goa': os.path.join(shp_csv_folder_path, 'Goa_buffer.shp'),
    'Mumbai': os.path.join(shp_csv_folder_path, 'Mumbai_buffer.shp'),
    'Bhuj': os.path.join(shp_csv_folder_path, 'Bhuj_buffer.shp'),
    'Hyderabad': os.path.join(shp_csv_folder_path, 'Hyderabad_buffer.shp'),
    'Karaikal': os.path.join(shp_csv_folder_path, 'Karaikal_buffer.shp'),
    'Kochi': os.path.join(shp_csv_folder_path, 'Kochi_buffer.shp'),
    'Thiruvananthapuram': os.path.join(shp_csv_folder_path, 'Thiruvananthapuram_buffer.shp'),
    'Paradip': os.path.join(shp_csv_folder_path, 'Paradip_buffer.shp'),
    'Kolkata': os.path.join(shp_csv_folder_path, 'Kolkata_buffer.shp'),
    'Mohanbari': os.path.join(shp_csv_folder_path, 'Mohanbari_buffer.shp'),
    'Patna': os.path.join(shp_csv_folder_path, 'Patna_buffer.shp'),
    'Lucknow': os.path.join(shp_csv_folder_path, 'Lucknow_buffer.shp'),
    #'Delhi': os.path.join(shp_csv_folder_path, 'Delhi_buffer.shp'),
    'Delhi': os.path.join(shp_csv_folder_path, 'Delhi_buffer_new.shp'),
    #'Chennai': os.path.join(shp_csv_folder_path, 'Chennai_buffer.shp'),
    'Chennai': os.path.join(shp_csv_folder_path, 'Chennai_buffer_old.shp'),
    'Vishakapatnam': os.path.join(shp_csv_folder_path, 'Vishakapatnam_buffer.shp'),
    'Machilipatnam': os.path.join(shp_csv_folder_path, 'Machilipatnam_buffer.shp'),
    # 'Sriharikota': os.path.join(shp_csv_folder_path, 'Sriharikota_buffer.shp'),
    'Jaipur': os.path.join(shp_csv_folder_path, 'Jaipur_buffer.shp'),
    'Srinagar': os.path.join(shp_csv_folder_path, 'Srinagar_buffer.shp'),
    'Jot': os.path.join(shp_csv_folder_path, 'Jot_buffer.shp'),
    'Mukteshwar': os.path.join(shp_csv_folder_path, 'Mukteshwar_buffer.shp'),
    'Lansdowne': os.path.join(shp_csv_folder_path, 'Lansdowne_buffer.shp'),
    'Surkandaji': os.path.join(shp_csv_folder_path, 'Surkandaji_buffer.shp'),
    'Banihal': os.path.join(shp_csv_folder_path, 'Banihal_buffer.shp'),
    'Jammu': os.path.join(shp_csv_folder_path, 'Jammu_buffer.shp'),
    'Kufri': os.path.join(shp_csv_folder_path, 'Kufri_buffer.shp'),
    'Murari': os.path.join(shp_csv_folder_path, 'Murari_buffer.shp'),
}

gcp_dict = {
    'Bhopal': "-gcp 260.367 459.543 77.4239 23.2416 -gcp 363.471 459.36 78.3973 23.2422 -gcp 467.859 459.8 79.3708 23.2422 -gcp 260.184 563.528 77.4222 22.3399 -gcp 260.184 667.916 77.4222 21.436 -gcp 364.572 639.727 78.3385 21.6538 -gcp 351.138 511.774 78.2491 22.7646 -gcp 156.291 639.727 76.5035 21.6515 -gcp 170.165 512.215 76.5961 22.7654 -gcp 156.126 459.36 76.4467 23.2419 -gcp 52.0677 459.029 75.4697 23.2422 -gcp 80.4772 355.632 75.7611 24.201 -gcp 207.824 369.507 76.9644 24.0392 -gcp 260.431 355.302 77.4226 24.145 -gcp 260.349 251.079 77.4234 25.0457 -gcp 350.532 407.166 78.2529 23.7202 -gcp 363.911 279.075 78.3435 24.8344 ",
    'Patiala': "-gcp 260.392 459.592 76.4495 30.3573 -gcp 86.7047 460.082 75.6693 30.3569 -gcp 259.364 285.856 76.449 31.0343 -gcp 433.197 458.907 77.2294 30.3569 -gcp 260.147 632.741 76.449 29.6822 ",
    'Nagpur': "-gcp 260.44 459.69 79.0595 21.0975 -gcp 364.241 459.47 80.0191 21.0962 -gcp 468.409 459.69 80.9786 21.0989 -gcp 260.294 354.861 79.0595 21.9993 -gcp 260.073 250.914 79.0595 22.9011 -gcp 156.181 459.47 78.0973 21.0989 -gcp 52.0126 459.58 77.1365 21.0975 -gcp 260.128 563.858 79.0595 20.1971 -gcp 260.128 667.751 79.0595 19.294 ",
    'Gopalpur': "-gcp 302.424 469.45 84.8819 19.2743 -gcp 301.961 359.202 84.8785 20.165 -gcp 301.961 248.145 84.8785 21.0599 -gcp 413.019 469.334 85.8155 19.2745 -gcp 522.688 468.409 86.7524 19.2745 -gcp 302.424 580.392 84.8785 18.3841 -gcp 301.961 690.987 84.8785 17.4914 -gcp 191.829 468.872 83.9415 19.2745 -gcp 80.3086 469.797 82.9978 19.2768 ",
    'Agartala': "-gcp 260.294 459.47 91.2496 23.8875 -gcp 364.241 459.763 92.1946 23.8869 -gcp 468.483 459.176 93.1377 23.8889 -gcp 350.44 407.643 92.0518 24.3535 -gcp 312.855 369.103 91.6925 24.6577 -gcp 364.388 279.25 92.1355 25.4306 -gcp 260 355.228 91.2465 24.7621 -gcp 155.832 279.25 90.3546 25.4355 -gcp 208.026 368.002 90.8035 24.6587 -gcp 155.832 459.176 90.2965 23.8879 -gcp 52.1044 459.396 89.3435 23.8879 -gcp 207.586 549.029 90.8025 23.117 -gcp 156.272 639.433 90.3585 22.3462 -gcp 260.33 563.564 91.2495 23.0166 -gcp 364.223 639.213 92.1355 22.3531 -gcp 350.679 511.701 92.0499 23.4271 ",
    'Sohra': "-gcp 302.536 469.474 91.733 25.269 -gcp 417.471 469.16 92.6858 25.2694 -gcp 532.405 469.16 93.6348 25.2675 -gcp 579.196 469.16 94.1113 25.2675 -gcp 501.316 354.539 93.3474 26.2067 -gcp 417.157 270.065 92.6188 26.8151 -gcp 301.908 354.539 91.7308 26.1397 -gcp 302.222 239.29 91.7308 27.014 -gcp 202.361 412.007 90.9196 25.7341 -gcp 103.127 354.225 90.1005 26.2086 -gcp 186.973 469.395 90.77 25.2694 -gcp 72.0383 469.631 89.8052 25.2714 -gcp 244.205 569.728 91.2878 24.4956 -gcp 102.892 583.859 90.1104 24.3342 -gcp 302.143 699.736 91.7308 23.5328 -gcp 302.379 584.566 91.7308 24.3972 -gcp 417.314 668.412 92.6208 23.7336 -gcp 501.159 584.095 93.3415 24.3401 -gcp 359.846 568.55 92.1758 24.5015 ",
    'Goa': "-gcp 260.469 459.375 73.825 15.4917 -gcp 155.721 459.375 72.8851 15.4917 -gcp 104.189 459.796 72.4246 15.4917 -gcp 52.0255 459.165 71.9579 15.4948 -gcp 364.533 459.375 74.7551 15.4933 -gcp 416.591 459.848 75.2187 15.4917 -gcp 468.413 458.981 75.6877 15.4917 -gcp 260.18 667.924 73.8209 13.6871 -gcp 260.18 615.392 73.8177 14.1413 -gcp 259.943 564.28 73.8209 14.5956 -gcp 260.18 355.101 73.8209 16.4003 -gcp 260.653 303.161 73.8209 16.8483 -gcp 260.298 251.812 73.8177 17.2995 -gcp 125.716 381.662 72.6198 16.1887 -gcp 208.24 368.884 73.3697 16.2789 -gcp 155.708 278.551 72.9123 17.0723 -gcp 207.619 549.55 73.3666 14.7045 -gcp 125.494 537.038 72.6229 14.801 -gcp 156.673 639.495 72.9123 13.9204 -gcp 312.579 550.124 74.2798 14.7037 -gcp 395.237 537.321 75.0223 14.7994 -gcp 364.989 639.033 74.7345 13.9189 -gcp 485.69 589.478 75.8188 14.3405 -gcp 312.093 369.049 74.2779 16.2851 -gcp 395.311 381.347 75.0246 16.191 -gcp 363.978 279.934 74.7306 17.0731 ",
    'Mumbai': "-gcp 929.971 1526.37 72.8765 19.1346 -gcp 1284.41 1528.34 73.8246 19.1327 -gcp 1641.56 1531.31 74.7726 19.1327 -gcp 928.736 1165.26 72.8765 20.0386 -gcp 927.255 805.141 72.8721 20.9424 -gcp 570.099 1528.34 71.924 19.1305 -gcp 214.426 1528.34 70.9737 19.1349 -gcp 928.736 1885.5 72.8743 18.2333 -gcp 927.255 2247.1 72.8743 17.3296 ",
    #'Mumbai': "-gcp 308.974 508.496 72.8754 19.135 -gcp 375.377 508.989 73.8223 19.1355 -gcp 442.396 508.989 74.7748 19.1355 -gcp 308.604 575.515 72.8721 18.2296 -gcp 308.358 641.795 72.8743 17.3258 -gcp 242.201 508.619 71.924 19.1355 -gcp 175.675 508.866 70.9715 19.1355 -gcp 308.851 442.586 72.8721 20.0415 -gcp 309.097 375.566 72.8743 20.9431 -gcp 308.111 776.45 72.8654 15.5249 -gcp 575.203 508.866 76.6715 19.1333 -gcp 308.111 241.282 72.8554 22.7451 -gcp 40.7737 508.373 69.061 19.13 ",
    'Bhuj': "-gcp 259.397 434.409 69.6427 23.2429 -gcp 259.397 332.216 69.6427 24.1421 -gcp 259.182 280.581 69.644 24.5923 -gcp 258.752 229.807 69.6453 25.0452 -gcp 361.16 434.409 70.6167 23.239 -gcp 463.354 434.409 71.5907 23.2442 -gcp 259.397 536.387 69.6401 22.3437 -gcp 259.182 638.151 69.6427 21.4432 -gcp 157.419 434.409 68.6713 23.2416 -gcp 55.0102 434.624 67.696 23.2442 ",
    'Hyderabad': "-gcp 299.447 420.685 78.4733 17.446 -gcp 419.411 420.793 79.4125 17.4458 -gcp 538.546 420.649 80.3499 17.4462 -gcp 298.978 300.649 78.4726 18.3477 -gcp 299.267 181.226 78.4733 19.25 -gcp 179.267 420.361 77.5326 17.4465 -gcp 59.988 420.288 76.5908 17.4465 -gcp 299.555 540 78.4722 16.5442 -gcp 298.978 659.712 78.4729 15.642 ",
    'Karaikal': "-gcp 260.325 459.303 79.8418 10.915 -gcp 364.333 458.924 80.7484 10.9194 -gcp 468.179 459.357 81.662 10.9124 -gcp 260.054 563.203 79.8348 10.0163 -gcp 259.621 667.049 79.8383 9.10618 -gcp 312.626 549.79 80.2916 10.1353 -gcp 364.333 639.573 80.7431 9.35296 -gcp 350.919 511.713 80.6258 10.4626 -gcp 350.054 407.001 80.6258 11.3709 -gcp 363.9 278.924 80.744 12.4876 -gcp 259.621 355.511 79.8339 11.819 -gcp 259.621 250.799 79.8374 12.7186 -gcp 156.316 279.141 78.9343 12.4806 -gcp 208.239 370.006 79.3893 11.7035 -gcp 155.667 459.249 78.9203 10.9159 -gcp 51.4964 458.924 78.0137 10.9159 -gcp 169.216 407.123 79.0463 11.3744 -gcp 170.007 512.267 79.0515 10.4626 -gcp 207.61 549.688 79.3858 10.1301 -gcp 155.951 639.864 78.9378 9.34946 ",
    'Kochi': "-gcp 225.479 867.557 75 8 -gcp 225.224 728.385 75 9 -gcp 225.326 589.417 75 10 -gcp 225.326 449.531 75 11 -gcp 225.224 310.359 75 12 -gcp 362.153 867.047 76 8 -gcp 362.153 728.64 76 9 -gcp 362.268 589.544 76 10 -gcp 362.421 449.455 76 11 -gcp 362.345 310.588 76 12 -gcp 499.452 867.506 77 8 -gcp 499.223 728.334 77 9 -gcp 499.376 589.544 77 10 -gcp 499.414 449.455 77 11 -gcp 499.414 310.512 77 12 -gcp 636.387 867.544 78 8 -gcp 636.273 728.41 78 9 -gcp 636.158 589.506 78 10 -gcp 636.387 449.569 78 11 -gcp 636.387 310.627 78 12 ",
    'Thiruvananthapuram': "-gcp 300.315 438.534 76.8665 8.53324 -gcp 300.315 545.084 76.8657 7.62791 -gcp 300.261 652.554 76.866 6.72259 -gcp 300.18 696.391 76.8642 6.36304 -gcp 193.251 438.48 75.957 8.53231 -gcp 85.997 438.398 75.0471 8.53416 -gcp 43.485 438.398 74.6843 8.53231 -gcp 300.504 331.388 76.8646 9.43579 -gcp 300.423 224.297 76.8646 10.3397 -gcp 300.22 181.339 76.866 10.7016 -gcp 407.372 438.48 77.7727 8.53185 -gcp 514.616 438.388 78.6801 8.53185 -gcp 383.505 491.28 77.5628 8.09417 -gcp 392.906 384.812 77.6443 8.98717 -gcp 407.315 253.416 77.7683 10.1 -gcp 485.455 331.453 78.4336 9.43911 -gcp 246.827 345.879 76.4143 9.31776 -gcp 193.125 253.776 75.96 10.1003 ",
    'Paradip': "-gcp 260.404 459.47 86.6532 20.2591 -gcp 260.33 615.281 86.6491 18.9299 -gcp 208.21 549.8 86.201 19.4794 -gcp 155.942 639.213 85.7512 18.7023 -gcp 364.131 640.387 87.5504 18.7058 -gcp 312.157 549.653 87.1024 19.4847 -gcp 394.082 537.908 87.8493 19.5687 -gcp 364.131 459.8 87.5868 20.2618 -gcp 468.079 459.213 88.5249 20.2618 -gcp 394.082 381.105 87.8528 20.9584 -gcp 311.864 368.772 87.0967 21.0424 -gcp 363.544 280.094 87.5482 21.816 -gcp 260.477 303.291 86.6504 21.5902 -gcp 155.942 279.507 85.7512 21.8195 -gcp 207.622 369.653 86.2028 21.0354 -gcp 125.11 381.692 85.4419 20.9566 -gcp 156.309 459.433 85.7132 20.2583 -gcp 52.5082 459.433 84.7689 20.2591 ",
    'Kolkata': "-gcp 315.969 623.777 89 21 -gcp 419.599 622.92 90 21 -gcp 108.424 623.491 87 21 -gcp 7.0783 511.011 86 22 -gcp 109.281 512.439 87 22 -gcp 212.91 512.439 88 22 -gcp 418.171 512.439 90 22 -gcp 8.50571 400.245 86 23 -gcp 110.708 401.672 87 23 -gcp 213.196 401.387 88 23 -gcp 315.113 401.672 89 23 -gcp 417.029 400.53 90 23 -gcp 415.602 289.478 90 24 -gcp 314.827 290.334 89 24 -gcp 212.91 289.763 88 24 -gcp 112.135 290.62 87 24 ",
    'Mohanbari': "-gcp 358.222 401.55 95.9684 27.9746 -gcp 363.654 470.408 96.0137 27.3741 -gcp 279.38 509.315 95.1906 27.0374 -gcp 412.178 271.909 96.5158 29.0797 -gcp 303.825 277.635 95.4466 29.0325 -gcp 150.546 319.038 93.9197 28.6722 -gcp 27.6591 397.164 92.7353 27.9899 -gcp 276.407 618.989 95.1778 26.0864 -gcp 449.837 489.494 96.8643 27.1998 -gcp 17.4184 505.681 92.6438 27.0463 -gcp 130.726 632.202 93.772 25.9683 ",
    'Patna': "-gcp 160.457 233.736 84.1033 27.5204 -gcp 314.323 310.082 85.612 26.8763 -gcp 464.078 344.144 87.0507 26.5858 -gcp 211.55 441.044 84.6179 25.7422 -gcp 136.966 350.604 83.9003 26.5158 -gcp 404.176 597.847 86.4486 24.3867 -gcp 328.418 547.341 85.738 24.8207 -gcp 229.168 581.403 84.7964 24.5302 -gcp 146.069 569.364 83.9878 24.6439 -gcp 135.791 491.843 83.8881 25.2723 ",
    'Lucknow': "-gcp 196.354 372.582 80.8833 26.7738 -gcp 275.994 277.075 81.9068 27.8731 -gcp 168.919 215.059 80.5172 28.5661 -gcp 132.581 283.373 80.0551 27.7645 -gcp 108.356 369.615 79.7436 26.7844 -gcp 114.17 517.873 79.8311 25.1007 -gcp 250.315 509.636 81.5568 25.1882 -gcp 337.525 424.364 82.6699 26.1614 -gcp 261.943 354.595 81.7213 26.97 -gcp 340.433 306.145 82.7469 27.523 -gcp 37.6185 367.677 78.8405 26.8194 -gcp 43.917 477.174 78.9245 25.5558 ",
    #'Delhi': "-gcp 284.445 435.392 77.2219 28.5897 -gcp 396.982 435.465 78.2427 28.591 -gcp 510.4 435.024 79.2646 28.5871 -gcp 284.335 548.883 77.2218 27.6882 -gcp 284.335 663.071 77.2198 26.7864 -gcp 170.367 435.135 76.1979 28.59 -gcp 57.3348 435.465 75.1751 28.591 -gcp 284.39 322.708 77.2208 29.4918 -gcp 284.17 209.29 77.2213 30.3916 ",
    #'Chennai': "-gcp 929.96 1528.66 80.2112 12.9452 -gcp 929.586 1316.45 80.2115 13.125 -gcp 718.314 1528.57 80.0277 12.9451 -gcp 929.165 1105.32 80.2117 13.3051 -gcp 929.586 892.891 80.2103 13.4864 -gcp 928.639 629.012 80.2113 13.7105 -gcp 929.586 1740.24 80.2108 12.7651 -gcp 929.113 1951.57 80.2108 12.5849 -gcp 929.744 2162.42 80.2113 12.4052 -gcp 929.113 2429.45 80.2103 12.1812 -gcp 506.779 1527.97 79.8436 12.9458 -gcp 295.928 1527.97 79.658 12.9453 -gcp 32.0477 1529.86 79.4297 12.9453 -gcp 1139.33 1527.97 80.3945 12.9453 -gcp 1351.45 1528.6 80.5774 12.9453 -gcp 1562.93 1528.6 80.762 12.9464 -gcp 1827.44 1528.6 80.9899 12.9464 ",
    'Delhi': "-gcp 399.54 619.069 77.2211 28.5902 -gcp 399.918 418.478 77.2167 29.0384 -gcp 400.928 219.022 77.2209 29.4936 -gcp 599.121 620.205 77.732 28.5893 -gcp 800.092 620.205 78.244 28.5915 -gcp 400.423 818.903 77.2217 28.1386 -gcp 399.918 1020.38 77.2184 27.6867 -gcp 198.948 620.458 76.7078 28.5904 -gcp -0.25495 619.7 76.1978 28.596",
    'Chennai': "-gcp 3.70072 234.736 78 15 -gcp 110.762 235.577 79 15 -gcp 2.85938 345.373 78 14 -gcp 110.552 346.424 79 14 -gcp 218.034 346.845 80 14 -gcp 325.726 235.367 81 15 -gcp 325.726 346.424 81 14 -gcp 432.577 234.525 82 15 -gcp 433.629 345.793 82 14 -gcp 1.38702 455.799 78 13 -gcp 109.5 457.482 79 13 -gcp 218.244 457.482 80 13 -gcp 326.462 457.482 81 13 -gcp 434.47 456.641 82 13 -gcp 109.027 567.909 79 12 -gcp 218.349 568.382 80 12 -gcp 326.567 568.382 81 12 -gcp 434.785 568.54 82 12 -gcp 108.948 678.966 79 11 -gcp 217.56 680.071 80 11 -gcp 326.41 679.755 81 11 -gcp 217.697 234.951 80 15 ",
    'Vishakapatnam':"-gcp 3.4894 330.832 81 19 -gcp 129.427 332.137 82 19 -gcp 255.692 332.463 83 19 -gcp 380.977 332.79 84 19 -gcp 507.241 331.158 85 19 -gcp 1.85808 464.274 81 18 -gcp 128.449 465.742 82 18 -gcp 255.692 465.416 83 18 -gcp 381.956 465.742 84 18 -gcp 508.546 465.09 85 18 -gcp 0.226754 597.227 81 17 -gcp 126.98 598.858 82 17 -gcp 254.876 598.858 83 17 -gcp 382.119 598.532 84 17 -gcp 510.015 598.532 85 17 -gcp 510.83 731.485 85 16 -gcp 382.935 731.811 84 16 -gcp 255.039 732.79 83 16 -gcp 126.736 731.077 82 16 ",
    'Machilipatnam': "-gcp 24.0433 743.69 79 15 -gcp 163.046 744.772 80 15 -gcp 302.859 889.453 81 14 -gcp 443.214 889.724 82 14 -gcp 302.859 745.042 81 15 -gcp 442.403 744.501 82 15 -gcp 581.675 743.96 83 15 -gcp 25.0574 598.941 79 16 -gcp 164.398 600.563 80 16 -gcp 303.13 600.361 81 16 -gcp 442.065 600.158 82 16 -gcp 580.391 599.549 83 16 -gcp 26.4772 454.53 79 17 -gcp 164.905 456.051 80 17 -gcp 303.181 456.659 81 17 -gcp 441.456 456.659 82 17 -gcp 578.667 455.747 83 17 -gcp 166.426 311.234 80 18 -gcp 303.333 311.843 81 18 -gcp 440.543 312.451 82 18 ",
    # 'Sriharikota': "-gcp 886.688 968.189 80.086 15.2177 -gcp 957.285 1527.14 80.2274 13.6644 -gcp 246.108 1393.99 78.4429 14.1394 -gcp 489.676 951.084 79.0881 15.0518 -gcp 139.186 1935.27 78.197 12.6988 -gcp 865.568 2114.16 79.999 12.2603 -gcp 786.391 2236.31 79.8163 11.9809 -gcp 640.871 2280.08 79.4549 11.8723 -gcp 314.837 2109.09 78.6416 12.2825 -gcp 647.829 1864.06 79.4754 12.8733 -gcp 1178.02 643.358 80.7804 15.7972 -gcp 432.59 1795.15 78.935 13.0347 -gcp 635.973 1437.18 79.4426 13.8976 ",
    'Jaipur': "-gcp 260.404 459.592 75.8176 26.8208 -gcp 364.682 459.702 76.8216 26.8199 -gcp 468.63 458.821 77.8245 26.8251 -gcp 259.853 354.433 75.8135 27.723 -gcp 259.853 251.366 75.8135 28.6287 -gcp 156.126 459.042 74.808 26.8199 -gcp 51.9576 459.042 73.8025 26.8251 -gcp 260.514 563.21 75.8135 25.922 -gcp 260.294 666.937 75.8161 25.0163 ",
    'Srinagar': "-gcp 220.306 305.865 74.8013 34.0508 -gcp 220.732 189.406 74.8026 34.5929 -gcp 220.306 112.62 74.8013 34.9552 -gcp 337.192 305.865 75.4523 34.0508 -gcp 414.405 307.145 75.8855 34.0521 -gcp 220.306 422.751 74.8026 33.51 -gcp 220.732 500.391 74.8013 33.1503 -gcp 103.847 305.865 74.1528 34.0508 -gcp 26.6338 306.292 73.7183 34.0495 ",
    'Jot': "-gcp 399.327 619.717 76.0589 32.4868 -gcp 399.015 819.286 76.0571 32.0356 -gcp 400.263 1021.04 76.0589 31.5858 -gcp 599.521 619.197 76.59 32.4872 -gcp 799.195 619.613 77.1212 32.4872 -gcp 399.223 418.691 76.0571 32.9379 -gcp 399.847 220.681 76.058 33.3894 -gcp 199.133 619.821 75.5268 32.4872 -gcp -0.540375 619.821 74.9939 32.4881 ",
    'Mukteshwar': "-gcp 299.674 486.539 79.6539 29.4591 -gcp 599.453 486.699 80.6802 29.4576 -gcp 299.273 786.88 79.6539 28.5583 -gcp -0.18541 485.977 78.6257 29.458 -gcp 300.476 186.399 79.6542 30.3573 ",
    'Lansdowne': "-gcp 299.473 486.338 78.6762 29.8502 -gcp 300.035 335.887 78.6752 30.3011 -gcp 449.924 486.098 79.1897 29.8509 -gcp 148.862 486.74 78.158 29.8509 -gcp 73.7563 486.098 77.9007 29.8509 -gcp 299.714 636.308 78.6752 29.3993 ",
    'Surkandaji': "-gcp 199.133 619.925 77.7686 30.4068 -gcp -0.956362 619.093 77.2475 30.4081 -gcp 600.145 620.341 78.8056 30.4068 -gcp 799.819 619.509 79.3267 30.4081 -gcp 400.471 419.003 78.2884 30.8584 -gcp 400.055 219.745 78.2871 31.3086 -gcp 400.471 819.39 78.2884 29.9553 -gcp 400.055 919.851 78.2884 29.7308 ",
    'Banihal': "-gcp 399.691 619.717 75.1974 33.4364 -gcp 399.587 419.315 75.1962 33.888 -gcp 400.211 219.641 75.1962 34.3383 -gcp 598.637 619.613 75.7344 33.4365 -gcp 799.559 618.989 76.2719 33.4365 -gcp 398.339 819.286 75.1969 32.9863 -gcp 398.963 1020.21 75.1975 32.5367 -gcp 198.665 620.237 74.6593 33.4371 -gcp -2.25632 619.613 74.1185 33.4371 ",
    'Jammu': "-gcp 399.535 619.613 74.8305 32.7811 -gcp 599.417 619.821 75.3621 32.7805 -gcp 799.091 619.821 75.8976 32.7831 -gcp 399.743 418.899 74.8291 33.2347 -gcp 399.743 219.433 74.8291 33.6849 -gcp 198.613 619.405 74.2949 32.7818 -gcp -1.06036 620.237 73.762 32.7818 -gcp 399.119 819.286 74.8298 32.3303 -gcp 399.119 919.955 74.8298 32.1051 ",
    'Kufri': "-gcp 399.535 619.405 77.2641 31.0966 -gcp 398.911 419.003 77.2641 31.5495 -gcp 399.535 219.953 77.2641 32.0004 -gcp 598.585 620.549 77.7889 31.0986 -gcp 798.883 621.173 78.3106 31.0976 -gcp 198.613 619.925 76.7404 31.0986 -gcp -1.68434 619.925 76.2156 31.0966 -gcp 398.911 820.222 77.2641 30.6467 -gcp 399.535 1020.52 77.2641 30.1958 ",
    'Murari': "-gcp 399.847 619.613 76.8172 31.5058 -gcp 398.911 820.326 76.8163 31.056 -gcp 399.535 1019.79 76.8163 30.6027 -gcp 599.833 619.405 77.3396 31.504 -gcp 799.923 618.989 77.8681 31.5058 -gcp 399.743 418.483 76.8163 31.9591 -gcp 399.743 218.809 76.8163 32.4106 -gcp 200.069 619.821 76.2894 31.5058 -gcp -0.852365 620.237 75.7609 31.5058 ",
}

region_codes = {
    'Bhopal': 'bhp',
    'Patiala': 'ptl',
    'Nagpur': 'ngp',
    'Gopalpur': 'gop',
    'Agartala': 'agt',
    'Sohra': 'cpj',
    'Goa': 'goa',
    'Mumbai': 'vrv',
    'Bhuj': 'bhj',
    'Hyderabad': 'hyd',
    'Karaikal': 'kkl',
    'Kochi': 'koc',
    'Thiruvananthapuram': 'tvm',
    'Paradip': 'pdp',
    'Kolkata': 'kol',
    'Mohanbari': 'mbr',
    'Patna': 'ptn',
    'Lucknow': 'lkn',
    'Delhi': 'delhi',
    'Chennai': 'cni',
    'Vishakapatnam': 'vsk',
    'Machilipatnam': 'mpt',
    # 'Sriharikota': 'shr',
    'Jaipur': 'jpr',
    'Srinagar': 'srn',
    'Jot': 'jot',
    'Mukteshwar': 'mks',
    'Lansdowne': 'ldn',
    'Surkandaji': 'sur',
    'Banihal': 'bnh',
    'Jammu': 'jmu',
    'Kufri': 'kuf',
    'Murari': 'mur',
    'Solapur': 'slp',
}


def process_gif_files2(region_name, region_row, region_codes, csv_files, shapefiles, gcp_dict, input_real_folder, parent_dir, process_folder, png_folder, extract_folder, fill_folder, georef_folder, clip_folder):
    #gif_files = [f for f in os.listdir(input_real_folder) if f.endswith('.gif')]

    # gif_files = [f for f in os.listdir(input_real_folder) if f.startswith(region_name) and f.endswith('.gif')]
    #region_code = region_name[:3].lower()


    region_code = region_codes.get(region_name)
    print("Region Code:", region_code)


    gif_files = [f for f in os.listdir(input_real_folder) if f.startswith(f"caz_{region_code}") and f.endswith('.gif')]
    
    print("Filtered GIF files:", gif_files)

    # Fetch necessary inputs based on the region name
    csv_file = csv_files.get(region_name)
    shapefile_path = shapefiles.get(region_name)
    gcp = gcp_dict.get(region_name)


    for gif_file in gif_files:
        gif_path = os.path.join(input_real_folder, gif_file)
        gif_filename = os.path.splitext(gif_file)[0]
        print("Processing:", gif_filename)  # Debug print

    # for gif_file in gif_files:
    #     gif_filename = os.path.splitext(gif_file)[0]
    #     if region_name == 'Bhopal' and 'bhp' in gif_filename:
    #         gif_path = os.path.join(input_real_folder, gif_file)
    #         print(gif_filename)
    #     if region_name == 'Nagpur' and 'ngp' in gif_filename:
    #         gif_path = os.path.join(input_real_folder, gif_file)
    #         print(gif_filename)
        
        png_path = os.path.join(png_folder, f"{gif_filename}.png")
        extract_png = os.path.join(extract_folder, f"{os.path.splitext(gif_file)[0]}_extract.png")
        output_fill_refl_png = os.path.join(fill_folder, f"{gif_filename}_extract_fill_refl.png")
        output_fill_rainrate_png = os.path.join(fill_folder, f"{gif_filename}_extract_fill_rainrate.png")
        gdal_translate_refl = os.path.join(georef_folder, f"{gif_filename}_extract_fill_gdal_refl.tif")
        gdal_translate_rainrate = os.path.join(georef_folder, f"{gif_filename}_extract_fill_gdal_rainrate.tif")
        final_georef_refl = os.path.join(georef_folder, f"{gif_filename}_extract_fill_georef_refl.tif")
        final_georef_rainrate = os.path.join(georef_folder, f"{gif_filename}_extract_fill_georef_rainrate.tif")
        clip_output_path_refl = os.path.join(clip_folder, f"{gif_filename}_extract_fill_georef_clip_refl.tif")
        clip_output_path_rainrate = os.path.join(clip_folder, f"{gif_filename}_extract_fill_georef_clip_rainrate.tif")

        convert_gif_to_png(gif_path, png_path)
        extract_image(png_path, csv_file, extract_png, region_name)
        #buffer = 2
        rgb_to_single_fill(extract_png, csv_file, output_fill_refl_png, output_fill_rainrate_png)
        georeference_pngs(output_fill_refl_png, gdal_translate_refl, final_georef_refl, gcp)
        georeference_pngs(output_fill_rainrate_png, gdal_translate_rainrate, final_georef_rainrate, gcp)
        clip_raster_to_shapefile(shapefile_path, final_georef_refl, clip_output_path_refl)
        clip_raster_to_shapefile(shapefile_path, final_georef_rainrate, clip_output_path_rainrate)



max_workers = min(8, 2)

def process_gif_files_parallel(df_regions, region_codes, csv_files, shapefiles, gcp_dict, input_real_folder, parent_dir, process_folder, png_folder, extract_folder, fill_folder, georef_folder, clip_folder):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:  
        futures = []
        for index, region_row in df_regions.iterrows():
            region_name = region_row['Region_name']
            futures.append(
                executor.submit(
                    process_gif_files2,
                    region_name,
                    region_row,
                    region_codes,
                    csv_files,
                    shapefiles,
                    gcp_dict,
                    input_real_folder,
                    parent_dir,
                    process_folder,
                    png_folder,
                    extract_folder,
                    fill_folder,
                    georef_folder,
                    clip_folder
                )
            )

       
        for future in futures:
            try:
                future.result()  
            except Exception as e:
                print(f"Error during processing: {e}")


for index, region_row in df_regions.iterrows():
    region_name = region_row['Region_name']
    gif_url = region_row['gif_url']
    bbox_crop = (region_row['x1'], region_row['y1'], region_row['x2'], region_row['y2'])
    print(region_name)

    #process_gif_files2(region_name, region_row, region_codes, csv_files, shapefiles, gcp_dict, input_real_folder, parent_dir, process_folder, png_folder, extract_folder, fill_folder, georef_folder, clip_folder)

process_gif_files_parallel(df_regions, region_codes, csv_files, shapefiles, gcp_dict, input_real_folder, parent_dir, process_folder, png_folder, extract_folder, fill_folder, georef_folder, clip_folder)

print('Processing radar gifs done')
# import sys
# sys.exit()



#To mosaic all rasters into one raster having some rasters with different bands

path = Path(clip_folder + "/")
mosaic_folder = os.path.join(output_folder, "output_mosaic_rainfallrate")
os.makedirs(mosaic_folder, exist_ok=True)
raster_files = list(path.iterdir())

raster_to_mosaic = []
raster_to_mosaic2 = []
for p in raster_files:
    raster = rasterio.open(p)
    raster_last = os.path.splitext(p)[0].split('_')[-1] 
    #print(raster_last)
    #print(raster.count)
    if raster_last == 'rainrate':
        raster_to_mosaic.append(raster)
    if raster_last == 'refl':
        raster_to_mosaic2.append(raster)


mosaic, output = merge(raster_to_mosaic, method = 'max')

mosaic2, output2 = merge(raster_to_mosaic2, method = 'max')

output_meta = raster.meta.copy()
output_meta.update(
    {"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    }
)

output_meta2 = raster.meta.copy()
output_meta2.update(
    {"driver": "GTiff",
        "height": mosaic2.shape[1],
        "width": mosaic2.shape[2],
        "transform": output2,
    }
)

# mosaic_filename = 'mosaic_rainrate_' + filename_datetime + '.tif'
# mosaic_path = os.path.join(mosaic_folder, mosaic_filename)
# with rasterio.open(mosaic_path, "w", **output_meta) as m:
#     m.write(mosaic)

mosaic_filename2 = 'mosaic_reflectivity_' + filename_datetime + '.tif'
mosaic_path2 = os.path.join(mosaic_folder, mosaic_filename2)
with rasterio.open(mosaic_path2, "w", **output_meta2) as m:
    m.write(mosaic2)

print('Mosaic done')
# import sys
# sys.exit()



#directories to be removed

directories_to_remove = [notmatched_files_folder, non_folder]
#unable to remove 'georef_folder' since files are open itseems
for directory in directories_to_remove:
    try:
        shutil.rmtree(directory)
        print(f"Removed directory: {directory}")
    except Exception as e:
        print(f"Error removing directory {directory}: {str(e)}")

print('Removed three folders')


def regrid_tiff(input_tiff_path, ref_tiff_path, output_tiff_path):
    try:
        
        with rasterio.open(input_tiff_path) as src:
            print(input_tiff_path)
            input_data = src.read(1)
            input_transform = src.transform
            input_crs = src.crs
            input_profile = src.profile

        with rasterio.open(ref_tiff_path) as ref_src:
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs

        # Determine the new dimensions and transform for the regridded raster
        width = ref_src.width
        height = ref_src.height

        # Perform the resampling using bilinear interpolation
        regridded_data = np.zeros((height, width), dtype=input_data.dtype)
        #regridded_data = np.full((height, width), np.nan, dtype=input_data.dtype)
        #regridded_data = np.full((height, width), np.nan, dtype=np.float64)

        # regridded_data = np.empty_like(ref_src.read(1), dtype = np.float32)
        # regridded_data.fill(np.nan)


        rasterio.warp.reproject(
            source=input_data,
            src_crs=input_crs,
            src_transform=input_transform,
            destination=regridded_data,
            dst_transform=ref_transform,  # Use the transform of the reference raster
            dst_crs=ref_crs,
            #resampling=Resampling.bilinear
            #resampling=Resampling.average
            resampling=Resampling.nearest
        )

        # Update the profile for the output raster using the properties of the reference raster
        output_profile = input_profile.copy()
        output_profile.update({
            'width': width,
            'height': height,
            'transform': ref_transform,
            'crs': ref_crs
        })

        nodata_value = src.nodata
        #regridded_data[regridded_data == 0] = nodata_value

        # Write the regridded raster to a new TIFF file
        with rasterio.open(output_tiff_path, 'w', **output_profile) as dst:
            dst.write(regridded_data, 1)

        print(f"Regridding completed successfully. Output saved to: {output_tiff_path}")

    except Exception as e:
        print(f"Error: {e}")

def regrid_tiffs(input_folder, ref_tiff_path, regrid_folder):
    
    input_tiff_files = glob.glob(os.path.join(input_folder, '*.tif'))

    for input_tiff_path in input_tiff_files:
       
        output_tiff_path = os.path.join(regrid_folder, os.path.splitext(os.path.basename(input_tiff_path))[0] + "_regrid.tif")

        regrid_tiff(input_tiff_path, ref_tiff_path, output_tiff_path)


if __name__ == "__main__":
    ref_tiff_path = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/Realtime_radar_02Apr2024_1622_dummy_reference/output_mosaic_rainfallrate/mosaic_reflectivity_nodata_02Apr2024_1622.tif")

    input_folder = mosaic_folder

    regrid_folder = mosaic_folder

    regrid_tiffs(input_folder, ref_tiff_path, regrid_folder)

end2 = time.time()
print('Time taken:', ((end2-start2)/60), 'minutes')
# import sys
# sys.exit()
