import os
import glob
import re
from datetime import datetime, timedelta
import rasterio
import numpy as np
from rasterio.warp import Resampling
import time
import sys

#python accum_IMD_RADAR.py "$(date +'%d%b%Y_%H%M')"

###Accumulated maps - 1hr, 2hr, 3hr, 6hr, 12hr, 24hr

#Get the time argument from the command line
if len(sys.argv) != 2:
    print("Usage: python accum_IMD_RADAR.py <system_time>")
    sys.exit(1)

system_time_str = sys.argv[1]

try:
    system_time = datetime.strptime(system_time_str, "%d%b%Y_%H%M")
    print("Received time:", system_time)
except ValueError as e:
    print("Error parsing date:", e)
    sys.exit(1)

#system_time = datetime(2024, 12, 22, 8, 25, 00)

start = time.time()

def create_output_folder(base_path, run_time):
    folder_name = run_time.strftime("%d%b%Y_%H%M")
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def filter_files_by_time(files, time_start, time_end):
    filtered_files = [
        file for file in files
        if time_start <= extract_time_from_filename(file) <= time_end
    ]
    for file in filtered_files:
        file_time = extract_time_from_filename(file)
        assert time_start <= file_time <= time_end, f"File {file} is outside the interval!"
    return sorted(filtered_files, key=extract_time_from_filename)


def extract_time_from_filename(filename):
    base = os.path.basename(filename)
    datetime_str = base.split('_')[2] + "_" + base.split('_')[3][:4]  # Extracts '12Nov2024_0940'
    return datetime.strptime(datetime_str, "%d%b%Y_%H%M")


def calculate_accumulated_rainfall(tiff_files, output_path):
    if len(tiff_files) < 2:
        print(f"Not enough files for accumulation. Skipping: {tiff_files}")
        return

    accumulated_data = None
    timestamps = [extract_time_from_filename(f) for f in tiff_files]

    for i in range(len(tiff_files) - 1):
        with rasterio.open(tiff_files[i]) as src1, rasterio.open(tiff_files[i + 1]) as src2:
            data1 = src1.read(1, resampling=Resampling.bilinear)
            data2 = src2.read(1, resampling=Resampling.bilinear)

            data1_mmhr = convert_to_mm_per_hour(data1)
            data2_mmhr = convert_to_mm_per_hour(data2)

            time_diff = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0
            contribution = ((data1_mmhr + data2_mmhr) / 2) * (time_diff / 60)

            if accumulated_data is None:
                accumulated_data = np.zeros_like(contribution)
            accumulated_data += contribution

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=accumulated_data.shape[0],
        width=accumulated_data.shape[1],
        count=1,
        dtype=accumulated_data.dtype,
        crs=src1.crs,
        transform=src1.transform,
    ) as dst:
        dst.write(accumulated_data, 1)

    print(f"Accumulated rainfall TIFF saved to {output_path}")


def convert_to_mm_per_hour(tiff_array):
    return ((10 ** (tiff_array / 10)) / 200) ** (5 / 8)

def validate_time_range(files, interval_start, interval_end):
    timestamps = [extract_time_from_filename(file) for file in files]
    
    timestamps = sorted(timestamps)
    
    tolerance_minutes = 10  
    if abs((timestamps[0] - interval_start).total_seconds()) > tolerance_minutes * 60:
        print(f"First file missing or outside tolerance: Expected ~{interval_start}, Found {timestamps[0]}")
        return False
    if abs((timestamps[-1] - interval_end).total_seconds()) > tolerance_minutes * 60:
        print(f"Last file missing or outside tolerance: Expected ~{interval_end}, Found {timestamps[-1]}")
        return False
    
    return True


def generate_accumulated_files(tiff_files, output_folder, current_time, intervals):
    for interval in intervals:
        print(f"Processing interval: {interval}HR")
        
        interval_start = current_time - timedelta(hours=interval)
        interval_end = current_time
        
        print(f"Interval {interval}HR: Start = {interval_start}, End = {interval_end}")

        interval_files = filter_files_by_time(tiff_files, interval_start, interval_end)
        
        if not interval_files:
            print(f"No files found for interval {interval}HR")
            continue
        
        if not validate_time_range(interval_files, interval_start, interval_end):
            print(f"Critical files missing for interval {interval}HR. Skipping.")
            continue
        
        print(f"Files used for interval {interval}HR: {interval_files}")
        
        output_filename = f"acc_rf_{interval}HR_{current_time.strftime('%H%M_%d%b%Y')}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        calculate_accumulated_rainfall(interval_files, output_path)

current_time = system_time - timedelta(minutes=5)
minute = (current_time.minute // 5) * 5
current_time = current_time.replace(minute=minute, second=0, microsecond=0)

#current_time = system_time.replace(minute=25, second=0, microsecond=0)

base_path = os.path.expanduser("/data/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/")
output_base_folder = os.path.expanduser("/data/NOWCAST_IMD_RADAR/accum_radar_data")
os.makedirs(output_base_folder, exist_ok=True)

time_start = current_time - timedelta(hours=24)  #start time for filtering files
tiff_files = []
pattern = r'mosaic_reflectivity_\d{2}[A-Za-z]{3}\d{4}_\d{4}\_regrid.tif$'

for folder in glob.glob(os.path.join(base_path, "Realtime_radar_*")):
    for file in glob.glob(os.path.join(folder, "output_mosaic_rainfallrate", "*.tif")):
        if re.match(pattern, os.path.basename(file)):
            file_time = extract_time_from_filename(file)
            if time_start <= file_time <= current_time:
                tiff_files.append(file)

tiff_files = sorted(tiff_files, key=extract_time_from_filename)

intervals = {
    "08:20": [1, 2, 3, 6, 12, 24],
    "09:20": [1],
    "10:20": [1, 2],
    "11:20": [1, 3],
    "12:20": [1, 2],
    "13:20": [1],
    "14:20": [1, 2, 3, 6],
    "15:20": [1],
    "16:20": [1, 2],
    "17:20": [1, 3],
    "18:20": [1, 2],
    "19:20": [1],
    "20:20": [1, 2, 3, 6, 12],
    "21:20": [1],
    "22:20": [1, 2],
    "23:20": [1, 3],
    "00:20": [1, 2],
    "01:20": [1],
    "02:20": [1, 2, 3, 6],
    "03:20": [1],
    "04:20": [1, 2],
    "05:20": [1, 3],
    "06:20": [1, 2],
    "07:20": [1]
}

current_hour_min = current_time.strftime("%H:%M")
selected_intervals = intervals.get(current_hour_min, [])

if not selected_intervals:
    print(f"No intervals defined for {current_hour_min}. Exiting.")
    exit(0)

output_folder = create_output_folder(output_base_folder, current_time)

generate_accumulated_files(tiff_files, output_folder, current_time, selected_intervals)

print('done')
end = time.time()
print('time taken for accum tiffs:', (end-start)/60, 'minutes')
