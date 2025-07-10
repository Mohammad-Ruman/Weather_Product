import os
import shutil
from datetime import datetime, timedelta, time


import time as timesss
import pandas as pd
from pysteps import nowcasts
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
from rasterio.enums import Resampling
from cdo import *
import cartopy.crs as ccrs
import glob
import pysteps
from pysteps.datasets import load_dataset
from pysteps import motion
import xarray as xr
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
import sys



###Step - 0
start1 = timesss.time()

# Define the directory to search for folders
search_directory = os.path.expanduser("NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS")

# Count the number of subfolders starting with 'Realtime_radar_INDIA'
folder_count = sum([1 for dir in os.listdir(search_directory) if os.path.isdir(os.path.join(search_directory, dir)) and dir.startswith('Realtime_radar_INDIA')])

print("Number of subfolders starting with 'Realtime_radar_INDIA':", folder_count)


# Check if folder_count is less than 6
if folder_count < 6:
    print("Insufficint files to run the code.")
    exit()

# Define the current system time
#system_time = datetime.now()

#python pysteps_sixfiles_updated_NOWCAST.py "$(date +'%d%b%Y_%H%M')"

if len(sys.argv) != 2:
    print("Usage: python pysteps_sixfiles_updated_NOWCAST.py <system_time>")
    sys.exit(1)

system_time_str = sys.argv[1]
#print('current', current_time_str)

try:
    system_time = datetime.strptime(system_time_str, "%d%b%Y_%H%M")
    print("Received time:", system_time)
except ValueError as e:
    print("Error parsing date:", e)
    sys.exit(1)

# Define the buffer duration (5 minutes)
# buffer_duration = timedelta(minutes=5)
# correct_time = (system_time - buffer_duration)
# print("Corrected datetime:", correct_time)

#system_time = datetime(2024, 5, 11, 23, 33, 00)

# Round the time to the nearest available time slot (00, 10, 20, 30, 40 or 50 minutes past the hour)
nearest_hour = (system_time.minute // 10) * 10
correct_time = system_time.replace(minute=nearest_hour, second=0, microsecond=0)
print("Adjusted datetime:", correct_time)

# Generate timestamps for the past 6 intervals
timestamps = [correct_time - timedelta(minutes=10*i) for i in range(5, -1, -1)]
print("Timestamps:", timestamps)

# Convert timestamps to folder names
folder_names = [timestamp.strftime("%d%b%Y_%H%M") for timestamp in timestamps]
print("Folder names:", folder_names)

# Define the directory to search for folders
search_directory = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS")

# Function to find the matching folder for a given timestamp
def find_matching_folder(search_directory, folder_name):
    for root, dirs, files in os.walk(search_directory):
        for dir in dirs:
            if folder_name in dir:
                output_mosaic_folder = os.path.join(root, dir, "output_mosaic_rainfallrate")
                if os.path.exists(output_mosaic_folder):
                    return output_mosaic_folder
    return None

# Keep track of collected files to avoid duplicates
collected_files = set()

dir2 = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_INPUTS")

big_folder = correct_time.strftime("%d%b%Y_%H%M")
destination_folder_name = 'Reflectivity_tiffs_' + str(big_folder)
destination_folder = os.path.join(dir2, destination_folder_name)
os.makedirs(destination_folder, exist_ok=True)

# # Function to copy unique files from the matching folders
# def copy_unique_files(folder):
#     if folder:
#         files = os.listdir(folder)
#         if files:
#             for file in files:
#                 if file.startswith('mosaic_reflectivity') and file.endswith('_regrid.tif'):
#                     file_path = os.path.join(folder, file)
#                     # Check if the file has already been collected
#                     if file_path not in collected_files:
#                         # Copy the file to the destination folder
#                         shutil.copy(file_path, destination_folder)
#                         collected_files.add(file_path)
#                         return True  # Return True if a unique file was copied
#     return False  # Return False if no unique file was found

# # Initialize counter for copied files
# files_copied = 0

# # Iterate over timestamps and copy files
# for folder_name in folder_names:
#     matching_folder = find_matching_folder(search_directory, folder_name)
#     if matching_folder:
#         if copy_unique_files(matching_folder):
#             files_copied += 1
#     else:
#         print(f"No matching folder found for timestamp {folder_name}. Looking for the next available folder.")
#         # Find the next available folder
#         next_folder = folder_name
#         while True:
#             next_folder_dt = datetime.strptime(next_folder, "%d%b%Y_%H%M") - timedelta(minutes=10)
#             next_folder = next_folder_dt.strftime("%d%b%Y_%H%M")
#             print("Next folder to search:", next_folder)
#             matching_folder = find_matching_folder(search_directory, next_folder)
#             if matching_folder:
#                 if copy_unique_files(matching_folder):
#                     files_copied += 1
#                     break

#     if files_copied == 6:
#         break

# print("Files copied:", files_copied)# Define the current system time

# print('done')
# import sys 
# sys.exit()



# ##Step - 1a

# #Regrid 6 tiffs before stacking - using a reference tiff

# def regrid_tiff(input_tiff_path, ref_tiff_path, output_tiff_path):
#     try:
#         # Open the input and reference TIFF files
#         with rasterio.open(input_tiff_path) as src:
#             input_data = src.read(1)
#             input_transform = src.transform
#             input_crs = src.crs
#             input_profile = src.profile

#         with rasterio.open(ref_tiff_path) as ref_src:
#             ref_transform = ref_src.transform
#             ref_crs = ref_src.crs

#         # Determine the new dimensions and transform for the regridded raster
#         width = ref_src.width
#         height = ref_src.height

#         # Perform the resampling using bilinear interpolation
#         #regridded_data = np.zeros((height, width), dtype=input_data.dtype)
#         #regridded_data = np.full((height, width), np.nan, dtype=input_data.dtype)
#         regridded_data = np.full((height, width), np.nan, dtype=np.float64)

#         # regridded_data = np.empty_like(ref_src.read(1), dtype = np.float32)
#         # regridded_data.fill(np.nan)


#         rasterio.warp.reproject(
#             source=input_data,
#             src_crs=input_crs,
#             src_transform=input_transform,
#             destination=regridded_data,
#             dst_transform=ref_transform,  # Use the transform of the reference raster
#             dst_crs=ref_crs,
#             #resampling=Resampling.bilinear
#             #resampling=Resampling.average
#             resampling=Resampling.nearest
#         )

#         # Update the profile for the output raster using the properties of the reference raster
#         output_profile = input_profile.copy()
#         output_profile.update({
#             'width': width,
#             'height': height,
#             'transform': ref_transform,
#             'crs': ref_crs
#         })

#         nodata_value = src.nodata
#         regridded_data[regridded_data == 0] = nodata_value

#         # Write the regridded raster to a new TIFF file
#         with rasterio.open(output_tiff_path, 'w', **output_profile) as dst:
#             dst.write(regridded_data, 1)

#         print(f"Regridding completed successfully. Output saved to: {output_tiff_path}")

#     except Exception as e:
#         print(f"Error: {e}")

# # if __name__ == "__main__":
# #     input_tiff_path = 'E:/Precipitation/CTT/ctt_12times/202307061700IST_MT9V04.0.NH.2023187.1130.GD.06K_cloud_temperature_top_level.tif'
# #     ref_tiff_path = 'E:/Precipitation/IMD_RADAR_GIFS/Hyderabad/ANIMATED_GIFS/TRIAL_OPS/Radar_6thJuly2023_tiffs_3/HYDERABAD_maxZ_20230706_231_19-12_extract_fill_georef_clip_clip.tif'
# #     output_tiff_path = 'E:/Precipitation/CTT/ctt_12times/202307061700IST_MT9V04.0.NH.2023187.1130.GD.06K_cloud_temperature_top_level_regrid_trial.tif'

# #     regrid_tiff(input_tiff_path, ref_tiff_path, output_tiff_path)



# #To regrid tiffs stored in a folder

# def regrid_tiffs(input_folder, ref_tiff_path, regrid_folder):
#     # List all TIFF files in the input folder
#     input_tiff_files = glob.glob(os.path.join(input_folder, '*.tif'))

#     for input_tiff_path in input_tiff_files:
#         # Generate the output TIFF file path
#         output_tiff_path = os.path.join(regrid_folder, os.path.splitext(os.path.basename(input_tiff_path))[0] + "_regrid.tif")

#         regrid_tiff(input_tiff_path, ref_tiff_path, output_tiff_path)


# if __name__ == "__main__":
#     #ref_tiff_path = 'E:/Precipitation/IMD_RADAR_GIFS/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Realtime_radar_02Apr2024_1622_dummy_reference/output_mosaic_rainfallrate/mosaic_reflectivity_nodata_02Apr2024_1622.tif'
#     ref_tiff_path = os.path.expanduser("/data/NOWCAST_IMD_RADAR/Realtime_radar_02Apr2024_1622_dummy_reference/output_mosaic_rainfallrate/mosaic_reflectivity_nodata_02Apr2024_1622.tif")
#     # input_folder = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Reflectivity_tiffs_02Apr2024_0040/")
#     # output_folder = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Reflectivity_tiffs_02Apr2024_0040/regrid_data3/")
#     input_folder = destination_folder 

#     regrid_folder_name = 'Regrid_data_' + str(big_folder)
#     regrid_folder = os.path.join(input_folder, regrid_folder_name)
#     os.makedirs(regrid_folder, exist_ok=True)

#     regrid_tiffs(input_folder, ref_tiff_path, regrid_folder)

# # print('done')
# # import sys
# # sys.exit()




# Step - 1b

# take the folder "Reflectivity_tiffs_correct_time" and use the tiffs to create an .nc file


#def time_index_from_filenames(filenames):
#    '''Helper function to create a pandas DatetimeIndex
#        mosaic_reflectivity_nodata_01Apr2024_2350
#       Filename example: HYDERABAD_maxZ_20230725_000_23-52_extract_fill_georef_clip_rainrate.tif'''
#    return pd.to_datetime([f.split('_')[2] + '_' + f.split('_')[3] for f in filenames], format='%d%b%Y_%H%M')

def time_index_from_filenames(filenames):
    datetimes = pd.to_datetime([f.split('_')[2] + '_' + f.split('_')[3] for f in filenames], format='%d%b%Y_%H%M')
    # Create a DataFrame to sort filenames by datetime
    df = pd.DataFrame({'filename': filenames, 'datetime': datetimes})
    df = df.sort_values(by='datetime')
    # Return sorted filenames and the DatetimeIndex
    return df['datetime'].tolist(), df['filename'].tolist()

dir3 = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA")

#regrid_folder = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Reflectivity_tiffs_02Apr2024_0040/near_data/")

destination_folder = "NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_INPUTS/Reflectivity_tiffs_01Jul2025_2110"
#os.chdir(regrid_folder)
os.chdir(destination_folder)
filenames = glob.glob('*.tif')
time, sorted_filenames = time_index_from_filenames(filenames)
print(time)
chunks = {'x': 3389, 'y': 3245, 'band': 1}

# Load the first dataset to get the geospatial information
first_ds = xr.open_dataset(sorted_filenames[0], chunks=chunks)
print(first_ds)

# Create a DataArray with the time coordinate
time_da = xr.DataArray(time, dims='time', coords={'time': time})

# Concatenate the datasets along the time dimension
da = xr.concat([xr.open_dataset(f, chunks=chunks) for f in sorted_filenames], dim=time_da)

# Assign the coordinate values to the time dimension
da['time'] = time_da

#output_nc_stack = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/Processed_files2/trial_stack_refl.nc")


Fore_obs_folder_name = 'Forecast_obs_pngs_' + str(big_folder)
Fore_obs_pngs_folder = os.path.join(dir3, Fore_obs_folder_name)
os.makedirs(Fore_obs_pngs_folder, exist_ok=True)

#refl_nc_name = 'Reflectivity_stack_trial2.nc'
refl_nc_name = 'Reflectivity_stack_' + str(big_folder) + '.nc'
output_nc_stack = os.path.join(Fore_obs_pngs_folder, refl_nc_name)

da.to_netcdf(output_nc_stack)
# print('done')
# import sys
# sys.exit()





# Step - 2

# To predict next 6 timestamps rainfall rate (.nc file output forecast) using observed .nc file (previous 6 times)

def dbz_to_rf(dBZ):
    rf=((10**(dBZ/10))/200)**(5/8) #https://en.wikipedia.org/wiki/DBZ_(meteorology)
    return rf

dir3 = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/")

# input_dir = dir3
# os.chdir(input_dir)

#output_nc_stack = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/Reflectivity_stack_trial2.nc")
ds = xr.open_dataset(output_nc_stack)

#print(ds)
precipitation = ds['band_data'].values
#precipitation=ds.band_data.values
#print(precipitation)

#ONE WAY TO GET TIMESTAMPS

##regrid_folder = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Reflectivity_tiffs_02Apr2024_0040/near_data/")

# folder_path = os.path.dirname(regrid_folder)

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

print(time_dummy)
print(time_dummy[6:])
print(time_dummy[:6])

#time_dummy=pd.date_range(datetime.today()-timedelta(hours=0.9), datetime.today()+timedelta(hours=1), freq='10T').floor('10min')

extrapolate = nowcasts.get_method("extrapolation")

train_precip = precipitation[0:6]

train_precip = np.transpose(train_precip, (0, 2, 3, 1))

train_precip = np.squeeze(train_precip, axis=-1)


#train_precip = np.transpose(train_precip)

oflow_method = motion.get_method("LK")
motion_field = oflow_method(train_precip)

#y - lat, x- lon

extrapolate = nowcasts.get_method("extrapolation")
n_leadtimes = 6
precip_forecast = extrapolate(train_precip[-1], motion_field, n_leadtimes)
precip_forecast_rf=dbz_to_rf(precip_forecast)  ##*2 for test case remove
train_precip_rf=dbz_to_rf(train_precip)  ##*2 for test case remove


ds_precip_forecast_rf = xr.Dataset({'rf': (['time','latitude','longitude' ],  precip_forecast_rf)},
                 coords={'longitude': (['longitude'], ds.x.values,{'units':'degrees_east'}),
                         'latitude': (['latitude'], ds.y.values,{'units':'degrees_north'}),
                         'time': time_dummy[6:]})

ds_obs_precip_rf = xr.Dataset({'rf': (['time','latitude','longitude' ],  train_precip_rf)},
                 coords={'longitude': (['longitude'], ds.x.values,{'units':'degrees_east'}),
                         'latitude': (['latitude'], ds.y.values,{'units':'degrees_north'}),
                         'time': time_dummy[:6]})


# Fore_obs_folder_name = 'Forecast_obs_pngs_' + str(big_folder)
# Fore_obs_pngs_folder = os.path.join(dir3, Fore_obs_folder_name)
# os.makedirs(Fore_obs_pngs_folder, exist_ok=True)

Forecast_folder_name = 'Forecast_folder_' + str(big_folder)
Forecast_folder = os.path.join(Fore_obs_pngs_folder, Forecast_folder_name)
os.makedirs(Forecast_folder, exist_ok=True)

Observ_folder_name = 'Observ_folder_' + str(big_folder)
Observ_folder = os.path.join(Fore_obs_pngs_folder, Observ_folder_name)
os.makedirs(Observ_folder, exist_ok=True)

Forecast_nc_name = 'ds_precip_forecast_rf_' + str(big_folder) + '.nc'
Forecast_nc_stack = os.path.join(Forecast_folder, Forecast_nc_name)

Observ_nc_name = 'ds_precip_observ_rf_' + str(big_folder) + '.nc'
Observ_nc_stack = os.path.join(Observ_folder, Observ_nc_name)

ds_precip_forecast_rf.to_netcdf(Forecast_nc_stack)

ds_obs_precip_rf.to_netcdf(Observ_nc_stack)

# print('done')
# import sys
# sys.exit()


###Step 2b - to get the accumulated tiff files from - observed and forecast (last 1 hr and next 1 hr)

def accumulate_rainfall(nc_file, output_tiff):
    ds = xr.open_dataset(nc_file)
    
    rainfall_var = [var for var in ds.data_vars if 'rf' in var.lower()][0]
    
    accumulated_rainfall = ds[rainfall_var].isel(time=slice(0, 6)).sum(dim='time')

    accumulated_rainfall = accumulated_rainfall * (10/60)
    
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    transform = from_origin(lon.min(), lat.max(), abs(lon[1] - lon[0]), abs(lat[1] - lat[0]))
    
    accumulated_rainfall_array = accumulated_rainfall.values
    
    accumulated_rainfall_array = np.nan_to_num(accumulated_rainfall_array, nan=0.0)
    
    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=accumulated_rainfall_array.shape[0],
        width=accumulated_rainfall_array.shape[1],
        count=1,
        dtype=accumulated_rainfall_array.dtype,
        crs='EPSG:4326',  
        transform=transform,
    ) as dst:
        dst.write(accumulated_rainfall_array, 1)
    
    print(f"Accumulated rainfall saved to {output_tiff}")

accum_tif_observ = os.path.join(Fore_obs_pngs_folder, "accumulated_last1hr_rf" + str(big_folder) + ".tif")
accumulate_rainfall(Observ_nc_stack, accum_tif_observ)

accum_tif_fore = os.path.join(Fore_obs_pngs_folder, "accumulated_next1hr_rf" + str(big_folder) + ".tif")
accumulate_rainfall(Forecast_nc_stack, accum_tif_fore)




# Step - 3

#To convert both observed and forecasted .nc files to IWM 0.05 degree grid

# cdo_regrd ='''gridtype  = lonlat
# gridsize  = 46460
# xname     = lon
# xlongname = longitude
# xunits    = degrees_east
# yname     = lat
# ylongname = latitude
# yunits    = degrees_north
# xsize     = 230
# ysize     = 202
# xfirst    = 73.325  
# xinc      = 0.05
# yfirst    = 12.625
# yinc      = 0.05
# '''

dir3 = os.path.expanduser("/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/")

# input_dir = dir3
# os.chdir(input_dir)

cdo_regrd ='''gridtype  = lonlat
gridsize  = 490700
xname     = lon
xlongname = longitude
xunits    = degrees_east
yname     = lat
ylongname = latitude
yunits    = degrees_north
xsize     = 701
ysize     = 700
xfirst    = 65.025   
xinc      = 0.05
yfirst    = 5.025
yinc      = 0.05
'''

#For WHOLE INDIA LEVEL GRID
# gridtype  = lonlat
# gridsize  = 490700
# xname     = lon
# xlongname = longitude
# xunits    = degrees_east
# yname     = lat
# ylongname = latitude
# yunits    = degrees_north
# xsize     = 701
# ysize     = 700
# xfirst    = 65.025 
# xinc      = 0.05
# yfirst    = 5.025
# yinc      = 0.05

file2write=open("cdo_regrd.txt",'w')
file2write.write(cdo_regrd)
file2write.close()

# nc_f = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/ds_precip_forecast_rf.nc")
# nc_o = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/ds_precip_obs_rf.nc")

nc_f = Forecast_nc_stack
nc_o = Observ_nc_stack

# f_iwm = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/forecast_rf_iwm.nc")
# o_iwm = os.path.expanduser("~/gifs/Hyderabad/Test_realtime_gifs/RADAR_DATA_FOLDERS/Reflectivity_Pysteps_Inputs/Pysteps_Outputs/obs_rf_iwm.nc")


Forecast_nc_name_iwm = 'forecast_rf_iwm_' + str(big_folder) + '.nc'
Forecast_nc_stack_iwm = os.path.join(Forecast_folder, Forecast_nc_name_iwm)

Observ_nc_name_iwm = 'observ_rf_iwm' + str(big_folder) + '.nc'
Observ_nc_stack_iwm = os.path.join(Observ_folder, Observ_nc_name_iwm)

f_iwm = Forecast_nc_stack_iwm
o_iwm = Observ_nc_stack_iwm

# os.system(f'cdo -f nc remapdis,cdo_regrd.txt {nc_f} {f_iwm}')
# os.system(f'cdo -f nc remapdis,cdo_regrd.txt {nc_o} {o_iwm}')
#os.system('rm temp1.nc temp2.nc cdo_regrd.txt')

os.system(f'cdo -f nc remapbil,cdo_regrd.txt {nc_f} {f_iwm}')
os.system(f'cdo -f nc remapbil,cdo_regrd.txt {nc_o} {o_iwm}')

print('done')

end1 = timesss.time()

print('time taken for pysteps:', ((end1 - start1)/60), 'minutes')
import sys
sys.exit()