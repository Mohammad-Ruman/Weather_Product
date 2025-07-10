import os
import shutil
from datetime import datetime, timedelta, time
import time
import sys

# import time
import pandas as pd
#from datetime import datetime, timedelta, time
# from pysteps import nowcasts
import xarray as xr
import numpy as np
# import time
import matplotlib.pyplot as plt
import rioxarray
# from rasterio.enums import Resampling
# from cdo import *
import cartopy.crs as ccrs
import glob
from multiprocessing import Pool


import rasterio

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

# import pysteps
# from pysteps.datasets import load_dataset
# from pysteps import motion

# Step - 4 - use env met_work3

#To convert .nc to .png (Observed and forecasted .nc files)

def heatmap2(input_nc_variable, col, brk,sufix,img_extent):
    # input_nc_variable=ds.tp
    t=0  
    dpi=700 
    proj = ccrs.Mercator.GOOGLE
    # os.mkdir(folder_name)
    # img_extent = (min(input_nc_variable.lon.values), max(input_nc_variable.lon.values),
    #               min(input_nc_variable.lat.values), max(input_nc_variable.lat.values))    
    print("BBOX:",img_extent)
    for time_ in input_nc_variable.time.values:
        fig = plt.figure(frameon=False,dpi=dpi)
        ax = plt.axes(projection=proj)
        ax.set_extent(img_extent,ccrs.PlateCarree())
        ax.outline_patch.set_visible(False)
        ax.background_patch.set_visible(False)
        ax.contourf(input_nc_variable.longitude.values, input_nc_variable.latitude.values, input_nc_variable[t,:,:].values, levels=brk,colors=col,alpha=1,corner_mask=True,extend="neither",transform=ccrs.PlateCarree())
        time_ = pd.to_datetime(str(time_)).strftime('%Y%m%d%H%M_IST')
        fig.savefig(sufix+time_+'temp.png',bbox_inches='tight',transparent=True,pad_inches = 0.0)
        plt.close()
        convert="convert "+sufix+time_+'temp.png '+ sufix+time_+'.png'
        os.system(convert)
        os.remove(sufix+time_+'temp.png')
        t=t+1

# def process_data(input_nc_variable, col, brk, sufix, img_extent, units, t, dpi, proj): # previous
#     data = input_nc_variable[t, :, :].values
#     time_ = input_nc_variable.time.values[t]
#     time_str = pd.to_datetime(str(time_)).strftime('%Y%m%d%H%M_IST')

#     fig = plt.figure(frameon=False, dpi=dpi)
#     ax = plt.axes(projection=proj)
#     ax.set_extent(img_extent, ccrs.PlateCarree())
#     # ax.outline_patch.set_visible(False)
#     # ax.background_patch.set_visible(False)
#     contour = ax.contourf(input_nc_variable.longitude.values, input_nc_variable.latitude.values, data, levels=brk, colors=col, alpha=1, corner_mask=True, extend="neither", transform=ccrs.PlateCarree())
#     fig.savefig(sufix + time_str + 'temp.png', bbox_inches='tight', transparent=True, pad_inches=0.0)
#     plt.close()
    
#     convert = "convert " + sufix + time_str + 'temp.png ' + sufix + time_str + '.png'
#     os.system(convert)
#     os.remove(sufix + time_str + 'temp.png')
#     return time_str

def process_data(input_nc_variable, col, brk, sufix, img_extent, units, t, dpi, proj, radar_gap_mask_path, new_col, new_brk):
    data = input_nc_variable[t, :, :].values
    time_ = input_nc_variable.time.values[t]
    time_str = pd.to_datetime(str(time_)).strftime('%Y%m%d%H%M_IST')

    lon = input_nc_variable.longitude.values
    lat = input_nc_variable.latitude.values

    # ---------- Original Plot ----------
    fig = plt.figure(frameon=False, dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_extent(img_extent, ccrs.PlateCarree())
    ax.set_frame_on(False)
    # ax.outline_patch.set_visible(False)
    ax.contourf(lon, lat, data, levels=brk, colors=col, alpha=1, corner_mask=True, extend="neither", transform=ccrs.PlateCarree())
    
    out_path = sufix + time_str + '.png'
    fig.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    print(f"Saved: {out_path}")

    # ---------- Masked Plot with Radar Mask ----------
    if radar_gap_mask_path:
        with rasterio.open(radar_gap_mask_path) as src:
            radar_mask = src.read(1)

        # Resize if shapes don't match
        if radar_mask.shape != data.shape:
            pad_rows = data.shape[0] - radar_mask.shape[0]
            pad_cols = data.shape[1] - radar_mask.shape[1]
            radar_mask = np.pad(radar_mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

        masked_data = data.copy()

        # Save original NaN locations
        original_nan_mask = np.isnan(masked_data)

        # Apply masking logic: where data is ~0 AND radar_mask == 1, set to -1
        # Define a small threshold for "zero" values (adjust as needed based on your data)
        zero_threshold = 0.04  # Based on your min value of ~0.0365
        
        # Condition: data is close to zero AND radar mask indicates gap
        mask_condition = (masked_data < zero_threshold) & (radar_mask == 1) & (~original_nan_mask)
        masked_data[mask_condition] = -1

        # For background areas (where data was originally NaN), set to 0 instead of keeping as NaN
        # This will make the background white/transparent instead of grey
        background_condition = original_nan_mask & (radar_mask == 0)
        masked_data[background_condition] = 0
        
        # Keep radar gap areas as NaN if they were originally NaN (this maintains transparency)
        radar_gap_nan_condition = original_nan_mask & (radar_mask == 1)
        masked_data[radar_gap_nan_condition] = np.nan

        # Use the new color scheme and breaks passed as parameters
        # Create masked array for plotting
        masked_data_plot = np.ma.masked_invalid(masked_data)

        fig2 = plt.figure(frameon=False, dpi=dpi, facecolor='white')
        ax2 = plt.axes(projection=proj, facecolor='white')
        ax2.set_extent(img_extent, ccrs.PlateCarree())
        ax2.set_frame_on(False)
        # ax2.outline_patch.set_visible(False)


        
        cs = ax2.contourf(
            lon, lat, masked_data_plot,
            levels=new_brk,
            colors=new_col,
            alpha=1,
            corner_mask=True,
            extend="max",
            transform=ccrs.PlateCarree()
        )

        aware_folder = "aware"
        os.makedirs(aware_folder, exist_ok=True)
        out_path_aware = os.path.join(aware_folder, sufix + time_str + '.png')

        fig2.savefig(out_path_aware, bbox_inches='tight', transparent=True, pad_inches=0.0)
        plt.close()
        print(f"Masked heatmap saved: {out_path_aware}")

    return time_str


# def heatmap(input_nc_variable, col, brk, sufix, img_extent, units):  # Prev
#     dpi = 700
#     proj = ccrs.Mercator.GOOGLE

#     # Create a list of arguments for each process
#     tasks = [(input_nc_variable, col, brk, sufix, img_extent, units, t, dpi, proj) for t in range(len(input_nc_variable.time.values))]

#     # Create a pool of worker processes
#     with Pool(processes=3) as pool:  # Adjust the number of processes as needed
#         results = pool.starmap(process_data, tasks)

#     print("All plots generated.")

def heatmap(input_nc_variable, col, brk, sufix, img_extent, units, radar_gap_mask_path=None, new_col=None, new_brk=None):
    dpi = 700
    proj = ccrs.Mercator.GOOGLE

    # Default new color scheme and breaks if not provided
    if new_col is None:
        new_col = [
            '#A0A0A0',   # -1: Radar gaps (existing grey)
            '#ffffff',   # 0: No Rain (white)
            '#fffebd',   # 0-2.4: Very Light Rainfall
            '#fffe02',   # 2.4-15.6: Light Rainfall
            '#02c3ff',   # 15.6-64.4: Moderate Rainfall
            '#30982f',   # 64.4-115.5: Heavy Rainfall
            '#f95c05',   # 115.5-204.4: Very Heavy Rainfall
            '#720101'    # >204.4: Extremely Heavy Rainfall
        ]
    
    if new_brk is None:
        new_brk = [-1.5, -0.5, 0.1, 2.4, 15.6, 64.4, 115.5, 204.4, 1000]

    # Prepare tasks
    tasks = [(input_nc_variable, col, brk, sufix, img_extent, units, t, dpi, proj, radar_gap_mask_path, new_col, new_brk)
             for t in range(len(input_nc_variable.time.values))]

    with Pool(processes=3) as pool:
        results = pool.starmap(process_data, tasks)

    print("All plots generated.")


def heatmap_nc(input_nc_variable, col, brk, sufix, img_extent):
    """
    Generates a heatmap from a NetCDF variable with a single timestep and saves it as a PNG file.
    
    Parameters:
    input_nc_variable: xarray DataArray
        The variable to be visualized.
    col: list
        List of colors for the heatmap.
    brk: list
        List of breakpoints for contour levels.
    sufix: str
        Prefix for the output file name.
    img_extent: tuple
        (min_lon, max_lon, min_lat, max_lat) for setting map extent.
    """
    dpi = 700  # Resolution of the image
    proj = ccrs.Mercator.GOOGLE
    
    # Extract the single time value
    time_ = input_nc_variable.time.values.item()
    time_str = pd.to_datetime(str(time_)).strftime('%Y%m%d%H%M_IST')
    
    fig = plt.figure(frameon=False, dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_extent(img_extent, ccrs.PlateCarree())
    ax.outline_patch.set_visible(False)
    ax.background_patch.set_visible(False)
    
    # Plot the data
    ax.contourf(
        input_nc_variable.longitude.values, 
        input_nc_variable.latitude.values, 
        input_nc_variable.values,  # Directly use the values since it's a single timestep
        levels=brk,
        colors=col,
        alpha=1,
        corner_mask=True,
        extend="neither",
        transform=ccrs.PlateCarree()
    )
    
    # Save the figure
    output_filename = f"{sufix}{time_str}.png"
    fig.savefig(output_filename, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    
    print(f"Saved heatmap as: {output_filename}")


# input_dir = dir3

start = time.time()

#system_time = datetime.now()
# Get the time argument from the command line
if len(sys.argv) != 2:
    print("Usage: python imd_radar_process_realtime_NOWCAST.py <system_time>")
    sys.exit(1)

system_time_str = sys.argv[1]
#print('current', current_time_str)

dir3 = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/")
radar_mask = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{system_time_str}/radar_uncovered_region_mask_{system_time_str}.tif"

try:
    system_time = datetime.strptime(system_time_str, "%d%b%Y_%H%M")
    print("Received time:", system_time)
except ValueError as e:
    print("Error parsing date:", e)
    sys.exit(1)

#system_time = datetime(2024, 5, 16, 14, 10, 00)

# buffer_duration = timedelta(minutes=13)
# correct_time = (system_time - buffer_duration)
# print(correct_time)


# Round the time to the nearest available time slot (00, 10, 20, 30, 40 or 50 minutes past the hour)
nearest_hour = (system_time.minute // 10) * 10
correct_time = system_time.replace(minute=nearest_hour, second=0, microsecond=0)
print("Adjusted datetime:", correct_time)

big_folder = correct_time.strftime("%d%b%Y_%H%M")
#202502101330
big_folder2 = correct_time.strftime("%Y%m%d%H%M")

Fore_obs_folder_name = 'Forecast_obs_pngs_' + str(big_folder)
Fore_obs_pngs_folder = os.path.join(dir3, Fore_obs_folder_name)
os.makedirs(Fore_obs_pngs_folder, exist_ok=True)

os.chdir(Fore_obs_pngs_folder)
#ONE WAY TO GET TIMESTAMPS

# system_time = datetime.now()
# buffer_duration = timedelta(minutes=4)

# timestamps = [system_time - timedelta(minutes=10*i) - buffer_duration for i in range(0, 6)]
# folder_names = [timestamp.strftime("%d%b%Y_%H%M") for timestamp in timestamps]

#ONE WAY TO GET TIMESTAMPS

# destination_folder = os.path.expanduser("/data/nowcast_imd_radar/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Reflectivity_tiffs_02Apr2024_0040/near_data/")

# folder_path = os.path.dirname(destination_folder)

# datetime_folder = folder_path.split("/")[-2].split('_')[2] + "_" + folder_path.split("/")[-2].split('_')[3]

# print(datetime_folder)

timestamp_str = str(big_folder)
timestamp_format = "%d%b%Y_%H%M"

T = datetime.strptime(timestamp_str, timestamp_format)

print(T)

timestamps_ahead = pd.date_range(T + pd.Timedelta(minutes=10), periods=6, freq='10T')

#timestamps_behind = pd.date_range(T - pd.Timedelta(minutes=10), periods=6, freq='-10T')

timestamps_behind = pd.date_range(T, periods=6, freq='-10T')

time_dummy = timestamps_behind.tolist()[::-1] + timestamps_ahead.tolist()

# time_dummy=pd.date_range(datetime.today()-timedelta(hours=0.9), datetime.today()+timedelta(hours=1), freq='10T').floor('10min')
# print(time_dummy)
# print(time_dummy[6:])
# print(time_dummy[:6])

#BBOX= (75.425 , 80.325, 8.075 , 13.875)
      
# nc_f = os.path.expanduser("/data/nowcast_imd_radar/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/ds_precip_forecast_rf.nc")
# nc_o = os.path.expanduser("/data/nowcast_imd_radar/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/ds_obs_precip_rf.nc")

parent_directory = Fore_obs_pngs_folder

# Define the name of the folder you know
folder_name_1 = "Forecast_folder_" + str(big_folder)
folder_name_2 = "Observ_folder_" + str(big_folder)

# Construct the full path to the folder
folder_path_1 = os.path.join(parent_directory, folder_name_1)
folder_path_2 = os.path.join(parent_directory, folder_name_2)

# Check if the folder exists
if os.path.exists(folder_path_1) and os.path.isdir(folder_path_1):
    files_in_folder = os.listdir(folder_path_1)
    nc_files = [file for file in files_in_folder if file.endswith('.nc') and file.startswith('ds_precip')]
    for nc_file in nc_files:
        nc_f = os.path.join(folder_path_1, nc_file)
        #print(os.path.join(folder_path_1, nc_file))
else:
    print("Folder '{}' not found.".format(folder_name_1))

# Check if the folder exists
if os.path.exists(folder_path_2) and os.path.isdir(folder_path_2):
    files_in_folder = os.listdir(folder_path_2)
    nc_files = [file for file in files_in_folder if file.endswith('.nc') and file.startswith('ds_precip')]
    for nc_file in nc_files:
        nc_o = os.path.join(folder_path_2, nc_file)
        #print(os.path.join(folder_path_1, nc_file))
else:
    print("Folder '{}' not found.".format(folder_name_2))

ds_precip_forecast_rf = xr.open_dataset(nc_f)
ds_obs_precip_rf = xr.open_dataset(nc_o)
temp2_time = (len(ds_obs_precip_rf.time))
print("No of timestamps in observed data",temp2_time)


temp1 = xr.Dataset({'rf': (['time','latitude','longitude' ],  ds_precip_forecast_rf.rf.values)},
                 coords={'longitude': (['longitude'], ds_precip_forecast_rf.longitude.values,{'units':'degrees_east'}),
                         'latitude': (['latitude'], ds_precip_forecast_rf.latitude.values,{'units':'degrees_north'}),
                         'time': time_dummy[6:]})


temp2 = xr.Dataset({'rf': (['time','latitude','longitude' ],  ds_obs_precip_rf.rf.values)},
                 coords={'longitude': (['longitude'], ds_obs_precip_rf.longitude.values,{'units':'degrees_east'}),
                         'latitude': (['latitude'], ds_obs_precip_rf.latitude.values,{'units':'degrees_north'}),
                         'time': time_dummy[:temp2_time]})

Forecast_nc_name = 'temp_forecast_' + str(big_folder) + '.nc'
Forecast_nc_stack = os.path.join(Fore_obs_pngs_folder, Forecast_nc_name)

Observ_nc_name = 'temp_observ_' + str(big_folder) + '.nc'
Observ_nc_stack = os.path.join(Fore_obs_pngs_folder, Observ_nc_name)

temp1.to_netcdf(Forecast_nc_stack)
temp2.to_netcdf(Observ_nc_stack)

#col=["#000096","#0064FF","#00B4FF","#33DB80","#9BEB4A","#FFEB00","#FFB300","#FF6400","#EB1E00","#AF0000"]
#brk=[0.1,0.5,1,2,3,5,10,15,20,25,10000]
# ------------------------------------------------------------------------------------------------------------------------------------
# col = ["#e6eeff", "#80aaff", "#3377ff", "#003cb3","#002266", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]
# brk=[0.5,2.5,5,10,20,30,40,50,75,100,1000]
# BBOX= (65.025, 100.025, 5.025, 39.975)

# units1 = "Forecasted Precipitation rate (mm/hr)"
# units2 = "Observed Precipitation rate (mm/hr)"

# heatmap(temp1.rf,col,brk,"f_",BBOX, units1)
# heatmap(temp2.rf,col,brk,"h_",BBOX, units2)
# ----------------------------------------------------------------------------------------------------------------------------------------
col = ["#e6eeff", "#80aaff", "#3377ff", "#003cb3", "#002266", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]
brk = [0.5,2.5,5,10,20,30,40,50,75,100,1000]
BBOX = (65.025, 100.025, 5.025, 39.975)
units1 = "Forecasted Precipitation rate (mm/hr)"
units2 = "Observed Precipitation rate (mm/hr)"

heatmap(temp1.rf, col, brk, "f_", BBOX, units1, radar_gap_mask_path=radar_mask)
heatmap(temp2.rf, col, brk, "h_", BBOX, units2, radar_gap_mask_path=radar_mask)





#heatmap(temp1.rf,col,brk,"f_",BBOX)

#heatmap(temp2.rf,col,brk,"h_",BBOX)


parent_directory = Fore_obs_pngs_folder
tiffs = os.listdir(parent_directory)
tiff_files = [file for file in tiffs if file.endswith('.tif')]
for tiff_file in tiff_files:
    if tiff_file.startswith('accumulated_last1hr_rf'):
        last1hr = os.path.join(parent_directory, tiff_file)
    if tiff_file.startswith('accumulated_next1hr_rf'):
        next1hr = os.path.join(parent_directory, tiff_file)

def pixel_to_coords(row, col, transform):
    """Convert pixel coordinates to geographic coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y

# def heatmap_tiff(input_tiff, coll, brk, sufix, img_extent, units, big_folder):
#     tiff_image = imageio.imread(input_tiff)
#     tiff_data = np.array(tiff_image)
#     print(tiff_data)

#     tiff_data = np.ma.masked_invalid(tiff_data)
#     tiff_data = np.ma.masked_equal(tiff_data, 255)

#     #tiff_data = np.where((tiff_data == 255) | np.isnan(tiff_data), 75, tiff_data)

#     #nodata_value = 255

#     # Extract geospatial metadata using rasterio
#     with rasterio.open(input_tiff) as src:
#         #raster_data = src.read(1)
#         # Convert 255 values to np.nan
#         # nodata_value = src.nodata
#         # raster_data = np.where(raster_data == nodata_value, np.nan, raster_data)
#         #masked_raster_data = np.ma.masked_equal(raster_data, nodata_value)
#         # Get the affine transformation matrix
#         transform = src.transform
#         # Get the coordinate reference system (CRS)
#         crs = src.crs

#     lats = []
#     lons = []

#     rows, cols = tiff_image.shape[:2]

#     #longitude values along the width (for each column at the first row)
#     for col in range(cols):
#         lon, _ = pixel_to_coords(0, col, transform)
#         lons.append(lon)

#     #latitude values along the height (for each row at the first column)
#     for row in range(rows):
#         _, lat = pixel_to_coords(row, 0, transform)
#         lats.append(lat)

#     # Ensure `coll` and `brk` are aligned
#     #assert len(coll) + 1 == len(brk), "Length of 'coll' must be one less than length of 'brk'"

#     dpi = 700
#     proj = ccrs.Mercator.GOOGLE

#     fig = plt.figure(frameon=False, dpi=dpi, facecolor='white')
#     ax = plt.axes(projection=proj, facecolor='white')
#     ax.set_extent(img_extent, ccrs.PlateCarree())
#     # ax.outline_patch.set_visible(False)
#     # ax.background_patch.set_visible(False)

#     flipped_data = np.flipud(tiff_data)

#     #img_extent = (67.4361879847097612, 97.1731435629566249, 6.3630656895180451, 34.8364858286183861)
#     print(img_extent)

#     try:

#         np_array=tiff_data.data
#         np_array[np.isnan(np_array)]=-999
        
#         contour = ax.contourf(lons, lats,np_array, levels=brk, colors=coll, alpha=1, corner_mask=True, extend="neither",
#                     transform=ccrs.PlateCarree())

#         output_filename = sufix + str(big_folder) + '_heatmap.png'

#         #fig.savefig(output_filename, bbox_inches='tight', transparent=True, pad_inches=0.0)
#         fig.savefig(output_filename, bbox_inches='tight', facecolor='white', edgecolor='white', pad_inches=0.0)
#         plt.close()
        
#         tiff_image = Image.open(output_filename)
#         tiff_image.save(output_filename, 'PNG')

#         print(f"Heatmap saved as {output_filename}")

#     except TopologicalError as e:
#         print(f"Topological error: {e}")

def heatmap_tiff(input_tiff, coll_new, coll_prev, brk_new, brk_prev, sufix, img_extent, units, big_folder, radar_gap_mask_path=None):
    # Read main TIFF with rasterio to preserve orientation
    with rasterio.open(input_tiff) as src:
        tiff_data = src.read(1)
        transform = src.transform
        crs = src.crs

    # Mask invalid values
    tiff_data = np.ma.masked_invalid(tiff_data)
    tiff_data = np.ma.masked_equal(tiff_data, 255)

    rows, cols = tiff_data.shape

    # Generate lon/lat
    lons = [pixel_to_coords(0, col, transform)[0] for col in range(cols)]
    lats = [pixel_to_coords(row, 0, transform)[1] for row in range(rows)]

    # Keep a clean copy for the "prev" (no radar mask) plot
    tiff_data2 = tiff_data.copy()

    # ---------- Save previous (no radar mask) version ----------
    # output_dir_prev = big_folder + "_prev"
    # os.makedirs(output_dir_prev, exist_ok=True)
    try:
        np_array_prev = np.array(tiff_data2, dtype=np.float32)
        np_array_prev[np.isnan(np_array_prev)] = -999

        dpi = 700
        proj = ccrs.Mercator.GOOGLE
        fig = plt.figure(frameon=False, dpi=dpi, facecolor='white')
        ax = plt.axes(projection=proj, facecolor='white')
        ax.set_extent(img_extent, ccrs.PlateCarree())

        ax.contourf(
            lons, lats, np_array_prev,
            levels=brk_prev, colors=coll_prev,
            alpha=1, corner_mask=True, extend="neither",
            transform=ccrs.PlateCarree()
        )

        output_filename_prev = os.path.join( sufix + big_folder + '_heatmap.png')
        fig.savefig(output_filename_prev, bbox_inches='tight', facecolor='white', edgecolor='white', pad_inches=0.0)
        plt.close()

        img = Image.open(output_filename_prev)
        img.save(output_filename_prev, 'PNG')
        print(f"Previous (no radar mask) heatmap saved as {output_filename_prev}")

    except TopologicalError as e:
        print(f"Topological error in prev version: {e}")

    # ---------- Apply radar mask and save new version ----------
    if radar_gap_mask_path:
        with rasterio.open(radar_gap_mask_path) as radar_src:
            radar_mask = radar_src.read(1)
            if radar_mask.shape != tiff_data.shape:
                print(f"Padding radar_mask from {radar_mask.shape} to {tiff_data.shape}")
                pad_rows = tiff_data.shape[0] - radar_mask.shape[0]
                pad_cols = tiff_data.shape[1] - radar_mask.shape[1]
                radar_mask = np.pad(radar_mask, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

            # Apply radar mask only where data < 0.1
            tiff_data = tiff_data.filled(0)
            mask_condition = (radar_mask == 1) & (tiff_data < 0.1)
            tiff_data[mask_condition] = -1

            # Add gray to colormap and breakpoints
            coll_new = ['#A0A0A0'] + coll_new
            brk_new = [-1.5] + brk_new

    # Save updated version with radar gap mask
    output_dir_new = "aware"
    os.makedirs(output_dir_new, exist_ok=True)

    try:
        np_array = np.array(tiff_data, dtype=np.float32)
        np_array[np.isnan(np_array)] = -999

        dpi = 700
        proj = ccrs.Mercator.GOOGLE
        fig = plt.figure(frameon=False, dpi=dpi, facecolor='white')
        ax = plt.axes(projection=proj, facecolor='white')
        ax.set_extent(img_extent, ccrs.PlateCarree())

        ax.contourf(
            lons, lats, np_array,
            levels=brk_new, colors=coll_new,
            alpha=1, corner_mask=True, extend="neither",
            transform=ccrs.PlateCarree()
        )

        output_filename_new = os.path.join(output_dir_new, sufix + big_folder + '_heatmap.png')
        fig.savefig(output_filename_new, bbox_inches='tight', facecolor='white', edgecolor='white', pad_inches=0.0)
        plt.close()

        img = Image.open(output_filename_new)
        img.save(output_filename_new, 'PNG')
        print(f"New (radar masked) heatmap saved as {output_filename_new}")

    except TopologicalError as e:
        print(f"Topological error in new version: {e}")

#coll = ["#e6eeff", "#80aaff", "#3377ff", "#003cb3","#002266", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]   

coll_prev = ["#80aaff", "#003cb3", "#ff80b3", "#ff0066", "#cc0000", "#660000", "#ffcccc"]   # Previous
coll_new = [   
            '#ffffff',   # 0: No Rain (white)
            '#fffebd',   # 0-2.4: Very Light Rainfall
            '#fffe02',   # 2.4-15.6: Light Rainfall
            '#02c3ff',   # 15.6-64.4: Moderate Rainfall
            '#30982f',   # 64.4-115.5: Heavy Rainfall
            '#f95c05',   # 115.5-204.4: Very Heavy Rainfall
            '#720101'    # >204.4: Extremely Heavy Rainfall
        ]    # New
#brk=[0.1,0.5,1,2,3,5,10,15,20,25,1000] 
#brk = [0.5,2.5,5,10,20,30,40,50,75,100, 1000]
brkk_new = [-0.5, 0.1, 2.4, 15.6, 64.4, 115.5, 204.4, 1000]
brkk_prev = [0, 1, 4, 16, 30, 50, 100, 1000]
#heatmaps - dashboard
units = 'Accumulated rainfall (mm) last1hr'
BBOX = (65.025, 100.025, 5.025, 39.975)
#AccumLast1hr_202502101330_heatmap
# heatmap_tiff(last1hr, coll, brkk, 'AccumLast1hr_', BBOX, units, big_folder2)
# heatmap_tiff(next1hr, coll, brkk, 'AccumNext1hr_', BBOX, units, big_folder2)
heatmap_tiff(
    input_tiff=last1hr,
    coll_new=coll_new,
    coll_prev=coll_prev,
    brk_new=brkk_new,
    brk_prev = brkk_prev,
    sufix='AccumLast1hr_',
    img_extent=(65.025, 100.025, 5.025, 39.975),
    units='Accumulated rainfall (mm) last1hr',
    big_folder=big_folder2,
    radar_gap_mask_path=radar_mask  # <-- New parameter
)
heatmap_tiff(
    input_tiff=last1hr,
    coll_new=coll_new,
    coll_prev=coll_prev,
    brk_new=brkk_new,
    brk_prev = brkk_prev,
    sufix='AccumNext1hr_',
    img_extent=(65.025, 100.025, 5.025, 39.975),
    units='Accumulated rainfall (mm) last1hr',
    big_folder=big_folder2,
    radar_gap_mask_path=radar_mask  # <-- New parameter
)

print('done') 
end = time.time()
print('time taken for heatmaps:', (end-start)/60, 'minutes')