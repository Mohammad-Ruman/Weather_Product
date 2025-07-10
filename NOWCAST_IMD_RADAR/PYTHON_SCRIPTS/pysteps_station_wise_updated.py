import os
import glob
import numpy as np
import rasterio
import xarray as xr
import logging
from datetime import datetime, timedelta
from pysteps import motion, nowcasts
import sys
import time

st = time.time()

if len(sys.argv) != 2:
    print("Usage: python pysteps_station_wise.py <timestamp_str>")
    sys.exit(1)
    
timestamp_str = sys.argv[1]

# === CONFIGURATION ===
TEMPORAL_WINDOW_SIZE = 8  # Number of timesteps for interpolation window 
RADAR_ROOT = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS"
FORECAST_OUT = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/forecast_output_{timestamp_str}"
OBSERVED_DATA_DIR = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/observed_data_{timestamp_str}"
PYSTEPS_TEMP_DIR = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/temp_pysteps_{timestamp_str}"

# === LOGGING CONFIGURATION ===
def setup_logging():
    """Setup logging configuration"""
    log_dir = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/LOGS"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"pysteps_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Configure logging - file only, no console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='w')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    return logger

# Initialize logger
logger = setup_logging()

logger.info(f"Configuration loaded:")
logger.info(f"TEMPORAL_WINDOW_SIZE: {TEMPORAL_WINDOW_SIZE}")
logger.info(f"RADAR_ROOT: {RADAR_ROOT}")
logger.info(f"FORECAST_OUT: {FORECAST_OUT}")
logger.info(f"NC_TEMP_DIR: {PYSTEPS_TEMP_DIR}")

os.makedirs(FORECAST_OUT, exist_ok=True)
os.makedirs(PYSTEPS_TEMP_DIR, exist_ok=True)
logger.info("Output directories created/verified")

# === UTILITY FUNCTIONS ===
def dbz_to_rf(dBZ):
    """Convert dBZ to rainfall rate (mm/h)"""
    logger.debug(f"Converting dBZ to rainfall rate. Input shape: {dBZ.shape}, dtype: {dBZ.dtype}")
    
    rf = ((10**(dBZ/10))/200)**(5/8)
    
    logger.debug(f"dBZ to RF conversion complete")
    return rf


def extract_time_station(file):
    """Extract station name and timestamp from filename"""
    logger.debug(f"Extracting time and station from file: {file}")
    
    name = os.path.basename(file)
    parts = name.split("_")
    station = parts[1]  # caz_hyd
    time_str = parts[2] + parts[3]  # 25Jun2025_1112
    timestamp = datetime.strptime(time_str, "%d%b%Y%H%M")
    
    logger.debug(f"Extracted - Station: {station}, Time: {timestamp}")
    return station, timestamp

def find_file_for_time(station, time_check):
    """Find radar file for a station and timestamp folder; return first match ignoring exact time."""
    logger.debug(f"Searching for file - Station: {station}, Time: {time_check}")
    
    base_folder = os.path.join(
        RADAR_ROOT,
        f"Realtime_radar_INDIA_{time_check.strftime('%d%b%Y_%H%M')}",
        "Processing_results/clipped_tifs"
    )
    
    if not os.path.exists(base_folder):
        logger.debug(f"Base folder does not exist: {base_folder}")
        return None

    pattern = f"caz_{station}_*clip_refl.tif"
    all_files = glob.glob(os.path.join(base_folder, pattern))

    logger.debug(f"Found {len(all_files)} files matching pattern: {pattern}")
    
    if all_files:
        logger.info(f"✅ Found file for {station} at {time_check}: {os.path.basename(all_files[0])}")
        return all_files[0]
    else:
        logger.warning(f"❌ No file found for {station} at {time_check}")
        return None

def get_interpolation_window(target_idx, available_files, total_timesteps):
    """
    Determine the best interpolation window for missing data
    
    Args:
        target_idx: Index of missing timestep
        available_files: List of available file indices
        total_timesteps: Total number of timesteps
    
    Returns:
        tuple: (ref_idx1, ref_idx2, interpolation_weight, is_interpolation)
    """
    logger.debug(f"Getting interpolation window for target_idx={target_idx}, available_files={available_files}")
    
    # Sort available files
    sorted_files = sorted(available_files)
    print(f"Sorted available files: {sorted_files}")
    
    # Find the best reference frames
    ref_idx1 = None
    ref_idx2 = None
    is_interpolation = False
    
    # Check if we can interpolate (target is between two available frames)
    for i in range(len(sorted_files) - 1):
        if sorted_files[i] < target_idx < sorted_files[i + 1]:
            ref_idx1 = sorted_files[i]
            ref_idx2 = sorted_files[i + 1]
            is_interpolation = True
            break
    
    # If we can't interpolate, find the two closest frames for extrapolation
    if not is_interpolation:
        if len(sorted_files) < 2:
            logger.error(f"Insufficient files for interpolation/extrapolation at index {target_idx}")
            return None, None, None, False
        
        # For edge cases, use the last two available frames
        if target_idx >= max(sorted_files):
            # Missing at end - use last two frames for forward extrapolation
            ref_idx1 = sorted_files[-2]
            ref_idx2 = sorted_files[-1]
        else:
            # Missing at beginning - use first two frames for backward extrapolation
            ref_idx1 = sorted_files[0]
            ref_idx2 = sorted_files[1]
    
    # Calculate interpolation/extrapolation weight
    if is_interpolation:
        # For interpolation in a sequence [0,1,2,3]:
        # Missing at 1 -> between 0 and 2 -> weight = 1/2 = 0.5 
        # Missing at 2 -> between 1 and 3 -> weight = 1/2 = 0.5 
        
        
        # Calculate weight based on position in the reference span
        total_span = ref_idx2 - ref_idx1
        if total_span == 2:  # Adjacent frames with one missing in between
            if target_idx == ref_idx1 + 1:  # Missing exactly in the middle
                weight = 1.0 / 2.0  # Use 1/2 for the intermediate position
            else:
                weight = 2.0 / 3.0  # Use 2/3 for the second position
        else:
            # For larger spans, use proportional weight
            steps_from_first = target_idx - ref_idx1
            weight = steps_from_first / total_span
        
        logger.info(f"INTERPOLATION: ref1={ref_idx1}, ref2={ref_idx2}, target={target_idx}, weight={weight:.3f}")
    else:
        # For extrapolation: use standard extrapolation (typically weight=1.0 for one step ahead)
        if target_idx > max(sorted_files):
            # Extrapolating forward
            weight = 1.0
            logger.info(f"FORWARD EXTRAPOLATION: ref1={ref_idx1}, ref2={ref_idx2}, target={target_idx}, weight={weight:.3f}")
        else:
            # Extrapolating backward
            weight = -1.0
            logger.info(f"BACKWARD EXTRAPOLATION: ref1={ref_idx1}, ref2={ref_idx2}, target={target_idx}, weight={weight:.3f}")
    
    return ref_idx1, ref_idx2, weight, is_interpolation

def interpolate_missing_data(files_dict, times_dict, target_idx, target_time, station):
    """
    Interpolate missing radar data using temporal window approach
    
    Args:
        files_dict: Dictionary of {index: filepath}
        times_dict: Dictionary of {index: datetime}
        target_idx: Index of missing timestep
        target_time: Target datetime
        station: Station name
    
    Returns:
        str: Path to interpolated file
    """
    logger.info(f"Interpolating missing data for {station} at index {target_idx} ({target_time})")
    
    available_indices = list(files_dict.keys())
    ref_idx1, ref_idx2, weight, is_interpolation = get_interpolation_window(target_idx, available_indices, len(times_dict))
    
    if ref_idx1 is None:
        logger.error(f"Cannot determine interpolation window for {station} at {target_time}")
        return None
    
    try:
        # Read reference files
        ref_file1 = files_dict[ref_idx1]
        ref_file2 = files_dict[ref_idx2]
        
        logger.debug(f"Using reference files: {os.path.basename(ref_file1)}, {os.path.basename(ref_file2)}")
        
        with rasterio.open(ref_file1) as src1, rasterio.open(ref_file2) as src2:
            arr1 = src1.read(1).astype(np.float32)
            arr2 = src2.read(1).astype(np.float32)
            profile = src1.profile
        
        R1 = arr1
        R2 = arr2
        
        # Create temporal stack for motion estimation (first and last frame)
        train_precip = np.stack([R1, R2], axis=0)  # [First, Last] frame
        
        # Calculate motion field using Lucas-Kanade
        oflow_method = motion.get_method("LK")
        motion_field = oflow_method(train_precip)  # Calculate motion field from first and last frame
        
        # Perform extrapolation with adjusted motion field
        extrapolate = nowcasts.get_method("extrapolation")
        n_leadtimes = 1
        
        if is_interpolation:
            logger.info(f"Performing INTERPOLATION with weight={weight:.3f}")
            # For interpolation: adjust motion field magnitude based on relative position
            adjusted_motion = motion_field * weight
            # Use first frame as input and extrapolate with adjusted motion
            precip_forecast = extrapolate(train_precip[0], adjusted_motion, n_leadtimes)
        else:
            logger.info(f"Performing EXTRAPOLATION with weight={weight:.3f}")
            # For extrapolation: use standard motion field
            if weight > 0:
                # Forward extrapolation
                precip_forecast = extrapolate(train_precip[-1], motion_field, n_leadtimes)
            else:
                # Backward extrapolation (reverse motion)
                precip_forecast = extrapolate(train_precip[0], -motion_field, n_leadtimes)
        
        forecast_dbz = precip_forecast[0]
        
        # Save interpolated file
        folder_path = os.path.join(RADAR_ROOT, f"Realtime_radar_INDIA_{target_time.strftime('%d%b%Y_%H%M')}",
                                   "Processing_results/clipped_tifs")
        os.makedirs(folder_path, exist_ok=True)
        save_name = f"caz_{station}_{target_time.strftime('%d%b%Y_%H%M')}_extract_fill_georef_clip_refl.tif"
        full_save_path = os.path.join(folder_path, save_name)
        
        with rasterio.open(full_save_path, "w", **profile) as dst:
            dst.write(forecast_dbz.astype(np.float32), 1)
        
        # Also save in temp directory
        temp_path = os.path.join(PYSTEPS_TEMP_DIR, f"interpolated_{station}_{target_time.strftime('%d%b%Y_%H%M')}_refl.tif")
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(forecast_dbz.astype(np.float32), 1)
        
        interpolation_type = "INTERPOLATED" if is_interpolation else "EXTRAPOLATED"
        logger.info(f"✅ Successfully {interpolation_type} missing data for {station} at {target_time}")
        return full_save_path
        
    except Exception as e:
        logger.error(f"❌ Error interpolating missing data for {station}: {e}", exc_info=True)
        return None

def collect_station_data(station, forecast_time, required_timesteps=8):
    """
    Collect radar data for a station, interpolating missing timesteps
    
    Args:
        station: Station name
        forecast_time: Reference forecast time
        required_timesteps: Number of timesteps needed
    
    Returns:
        tuple: (files_list, times_list, success_flag)
    """
    logger.info(f"Collecting data for station {station} with {required_timesteps} timesteps")
    
    # Initialize data structures
    files_dict = {}
    times_dict = {}
    
    # First pass: collect all available files
    for i in range(required_timesteps):
        t = forecast_time - timedelta(minutes=10 * (required_timesteps - 1 - i))
        times_dict[i] = t
        
        f = find_file_for_time(station, t)
        if f:
            files_dict[i] = f
            logger.info(f"✅ Found file for timestep {i}: {t.strftime('%H%M')}")
        else:
            logger.warning(f"❌ Missing file for timestep {i}: {t.strftime('%H%M')}")
    
    # Check if we have enough files for interpolation
    available_count = len(files_dict)
    missing_indices = [i for i in range(required_timesteps) if i not in files_dict]
    
    logger.info(f"Data collection summary: {available_count}/{required_timesteps} files available")
    logger.info(f"Missing indices: {missing_indices}")
    
    if available_count < 2:
        logger.error(f"Insufficient data for {station}: only {available_count} files available")
        return [], [], False
    
    # Second pass: interpolate missing files
    for missing_idx in missing_indices:
        target_time = times_dict[missing_idx]
        logger.info(f"Attempting to interpolate missing timestep {missing_idx}: {target_time.strftime('%H%M')}")
        
        interpolated_file = interpolate_missing_data(files_dict, times_dict, missing_idx, target_time, station)
        
        if interpolated_file:
            files_dict[missing_idx] = interpolated_file
            logger.info(f"✅ Successfully interpolated timestep {missing_idx}")
        else:
            logger.error(f"❌ Failed to interpolate timestep {missing_idx}")
    
    # Final check
    final_count = len(files_dict)
    if final_count == required_timesteps:
        # Sort by index to maintain temporal order
        files_list = [files_dict[i] for i in sorted(files_dict.keys())]
        times_list = [times_dict[i] for i in sorted(times_dict.keys())]
        
        logger.info(f"✅ Successfully collected all {required_timesteps} timesteps for {station}")
        return files_list, times_list, True
    else:
        logger.error(f"❌ Still missing data for {station}: {final_count}/{required_timesteps} files")
        return [], [], False

def stack_to_nc(files, times, output_nc):
    """Stack multiple radar files into a netCDF file"""
    logger.info(f"Stacking {len(files)} files to netCDF: {output_nc}")
    
    data = []
    for i, f in enumerate(files):
        logger.debug(f"Reading file {i+1}/{len(files)}: {os.path.basename(f)}")
        with rasterio.open(f) as src:
            arr = src.read(1)
            data.append(arr)
            profile = src.profile
            logger.debug(f"File {i+1} shape: {arr.shape}, range: [{np.min(arr):.2f}, {np.max(arr):.2f}]")

    stacked = np.stack(data, axis=0)
    logger.info(f"Stacked array shape: {stacked.shape}")
    
    ds = xr.Dataset({
        "dbz": (["time", "y", "x"], stacked)
    }, coords={
        "time": times,
        "y": np.arange(stacked.shape[1]),
        "x": np.arange(stacked.shape[2])
    })
    
    logger.debug(f"Saving netCDF file: {output_nc}")
    ds.to_netcdf(output_nc)
    print("Output netCDF path:", output_nc)
    logger.info(f"NetCDF file created successfully")
    
    return output_nc, profile

def run_station_forecast(station, files, times, outdir, station_obs_dir):
    """Run multi-timestep forecast for a station"""
    logger.info(f"Starting multi-timestep forecast for station: {station}")
    logger.info(f"Input files: {len(files)}, Output directory: {outdir}")
    
    try:
        # Create netCDF stack
        output_nc = os.path.join(PYSTEPS_TEMP_DIR, f"stack_{station}.nc")
        nc_path, profile = stack_to_nc(files, times, output_nc)
     
        # Load data
        logger.debug("Loading netCDF data")
        ds = xr.open_dataset(nc_path)
        dbz_stack = ds.dbz.values.astype(np.float32)
        logger.info(f"Loaded radar stack shape: {dbz_stack.shape}")
        
        # Save observed RF images
        logger.info(f"Saving observed data to: {station_obs_dir}")
        for i in range(len(times)):
            obs_path = os.path.join(station_obs_dir, f"observed_{station}_{times[i].strftime('%d%b%Y_%H%M')}_rf.tif")
            obs_arr = dbz_to_rf(dbz_stack[i])
            with rasterio.open(obs_path, "w", **profile) as dst:
                dst.write(obs_arr, 1)

            
        # Optical flow and forecast in dBZ
        logger.info("Calculating optical flow in dBZ space")
        V = motion.get_method("LK")(dbz_stack)
        logger.info(f"Optical flow calculated for {len(times)} timesteps")

        logger.info("Running 6-timestep extrapolation forecast in dBZ")
        extrapolate = nowcasts.get_method("extrapolation")
        forecast_dbz = extrapolate(dbz_stack[-1], V, 6)  # in dBZ
        logger.info(f"6-timestep forecast completed. Shape: {forecast_dbz.shape}")

        # Save forecast timesteps in RF
        logger.info("Saving individual forecast timesteps in RF")
        for i in range(6):
            timestamp = times[-1] + timedelta(minutes=10 * (i + 1))
            out_path = os.path.join(outdir, f"forecast_{station}_{timestamp.strftime('%d%b%Y_%H%M')}_rf.tif")
            out_rf = dbz_to_rf(forecast_dbz[i].astype(np.float32))

            logger.info(f"Saving timestep {i+1}/6 to: {os.path.basename(out_path)}")
            try:
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(out_rf, 1)
            except Exception as e:
                logger.error(f"❌ Failed to write forecast file {out_path}: {e}", exc_info=True)
            
            logger.debug(f"Timestep {i+1} saved. Range: [{np.min(out_rf):.2f}, {np.max(out_rf):.2f}] RF")

        logger.info(f"✅ All forecasts saved successfully for {station}")
        
        # Clean up temporary netCDF file
        try:
            os.remove(nc_path)
            logger.debug(f"Temporary netCDF file removed: {nc_path}")
        except:
            logger.warning(f"Could not remove temporary file: {nc_path}")
        
    except Exception as e:
        logger.error(f"❌ Error in run_station_forecast for {station}: {e}", exc_info=True)


# === MAIN LOOP ===
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("STARTING ENHANCED PYSTEPS RADAR NOWCASTING SYSTEM")
    logger.info("="*80)
    
    forecast_time = datetime.strptime(timestamp_str, "%d%b%Y_%H%M")
    stations = ['bhp', 'ptl', 'ngp', 'gop', 'agt', 'cpj', 'goa', 'vrv', 'bhj', 'hyd', 'kkl', 'koc',
 'tvm', 'pdp', 'kol', 'mbr', 'ptn', 'lkn', 'delhi', 'cni', 'vsk', 'mpt', 'shr',
 'jpr', 'srn', 'jot', 'mks', 'ldn', 'sur', 'bnh', 'jmu', 'kuf', 'mur', 'slp']
    
    logger.info(f"Forecast time: {forecast_time}")
    logger.info(f"Temporal window size: {TEMPORAL_WINDOW_SIZE} timesteps")
    logger.info(f"Processing {len(stations)} stations: {stations}")
    
    successful_stations = []
    failed_stations = []
    
    for station_idx, station in enumerate(stations):
        logger.info("="*60)
        logger.info(f"Processing station {station_idx+1}/{len(stations)}: {station}")
        logger.info("="*60)
        
        # Use enhanced data collection with interpolation
        files, times, success = collect_station_data(station, forecast_time, required_timesteps = TEMPORAL_WINDOW_SIZE)
        
        if success:
            logger.info(f"✅ Successfully collected all data for {station}. Starting forecast generation.")
            
            # Log file details
            for i, (f, t) in enumerate(zip(files, times)):
                logger.debug(f"  File {i+1}: {t.strftime('%H%M')} -> {os.path.basename(f)}")
            
            station_for_dir = os.path.join(FORECAST_OUT, station)
            os.makedirs(station_for_dir, exist_ok=True)
            logger.debug(f"Output directory created for forecast: {station_for_dir}")
            
            station_obs_dir = os.path.join(OBSERVED_DATA_DIR, station)
            os.makedirs(station_obs_dir, exist_ok=True)
            logger.debug(f"Output directory created for observed: {station_obs_dir}")
            
            try:
                run_station_forecast(station, files, times, station_for_dir, station_obs_dir)
                successful_stations.append(station)
                logger.info(f"✅ Station {station} completed successfully")
            except Exception as e:
                logger.error(f"❌ Station {station} failed during forecast generation: {e}")
                failed_stations.append((station, str(e)))
        else:
            logger.warning(f"⏭️ Skipping station {station} due to insufficient data")
            failed_stations.append((station, "Insufficient data after interpolation"))

    # Final summary
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE - FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total stations processed: {len(stations)}")
    logger.info(f"Successful: {len(successful_stations)}")
    logger.info(f"Failed: {len(failed_stations)}")
    
    if successful_stations:
        logger.info(f"✅ Successful stations: {successful_stations}")
    
    if failed_stations:
        logger.info("❌ Failed stations:")
        for station, reason in failed_stations:
            logger.info(f"  - {station}: {reason}")
    
    logger.info("="*80)
    logger.info("ENHANCED PYSTEPS RADAR NOWCASTING SYSTEM FINISHED")
    logger.info("="*80)
    
    et = time.time()
    logger.info(f"Total time taken: {(et - st) :.2f} secs")
    print(f"Total time taken: {(et - st) :.2f} secs")



