import os
import glob
import numpy as np
import rasterio
import xarray as xr
import logging
from datetime import datetime, timedelta
from pysteps import motion, nowcasts
from rasterio.transform import from_origin
import sys

if len(sys.argv) != 2:
    print("Usage: python pysteps_station_wise.py <timestamp_str>")
    sys.exit(1)
    
timestamp_str = sys.argv[1]

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

# === CONFIGURATION ===
RADAR_ROOT = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS"
FORECAST_OUT = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/forecast_output_{timestamp_str}"
OBSERVED_DATA_DIR = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/observed_data_{timestamp_str}"
PYSTEPS_TEMP_DIR = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/temp_pysteps_{timestamp_str}"

logger.info(f"Configuration loaded:")
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
    
    # Handle invalid values
    invalid_count = np.sum(~np.isfinite(dBZ))
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid dBZ values, setting to 0")
    
    # dBZ = np.where(np.isfinite(dBZ), dBZ, 0)
    rf = ((10**(dBZ/10))/200)**(5/8)
    # rf = np.where(np.isfinite(rf), rf, 0)
    
    logger.debug(f"dBZ to RF conversion complete. Output range: [{np.min(rf):.6f}, {np.max(rf):.6f}]")
    return rf

def rf_to_dbz(rf):
    """Convert rainfall rate to dBZ"""
    logger.debug(f"Converting rainfall rate to dBZ. Input shape: {rf.shape}, dtype: {rf.dtype}")
    
    # Avoid log of zero
    zero_count = np.sum(rf <= 0)
    if zero_count > 0:
        logger.debug(f"Found {zero_count} zero/negative RF values, setting to 1e-6")
    
    # rf = np.where(rf > 0, rf, 1e-6)
    dbz = 10 * np.log10(200 * (rf**(8/5)))
    # dbz = np.where(np.isfinite(dbz), dbz, 0)
    
    logger.debug(f"RF to dBZ conversion complete. Output range: [{np.min(dbz):.2f}, {np.max(dbz):.2f}]")
    return dbz

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
    
def create_one_lead_forecast(f1, f2, target_time, station):
    """Create a single timestep forecast using two previous radar images"""
    logger.info(f"Creating one-step forecast for {station} at {target_time}")
    logger.debug(f"Using files: {os.path.basename(f1)}, {os.path.basename(f2)}")
    
    try:
        # Read radar data
        logger.debug("Reading radar data from files")
        with rasterio.open(f1) as src1, rasterio.open(f2) as src2:
            arr1 = src1.read(1).astype(np.float32)
            arr2 = src2.read(1).astype(np.float32)
            profile = src1.profile
            logger.debug(f"File 1 shape: {arr1.shape}, range: [{np.min(arr1):.2f}, {np.max(arr1):.2f}]")
            logger.debug(f"File 2 shape: {arr2.shape}, range: [{np.min(arr2):.2f}, {np.max(arr2):.2f}]")

        # Convert to rainfall rate
        logger.debug("Converting dBZ to rainfall rate")
        R1 = dbz_to_rf(arr1)
        R2 = dbz_to_rf(arr2)
        
        # Stack arrays - PySTEPS expects shape (time, y, x)
        logger.debug("Stacking arrays for PySTEPS")
        R = np.stack([R1, R2], axis=0)
        logger.info(f"Stacked array shape: {R.shape}, dtype: {R.dtype}")
        
        # Ensure we have valid data
        valid_pixels = np.sum(R > 0)
        total_pixels = R.size
        logger.info(f"Valid data pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        
        if R.size == 0 or np.all(R == 0):
            logger.error(f"No valid radar data for {station} at {target_time}")
            return None
            
        # Calculate optical flow
        logger.info("Calculating optical flow using Lucas-Kanade method")
        oflow_method = motion.get_method("LK")
        V = oflow_method(R)  # Returns velocity field (vx, vy)
        logger.info(f"Optical flow calculated. Velocity field shape: vx={V[0].shape}, vy={V[1].shape}")
        logger.debug(f"Velocity range - vx: [{np.min(V[0]):.3f}, {np.max(V[0]):.3f}], vy: [{np.min(V[1]):.3f}, {np.max(V[1]):.3f}]")
        
        # Perform extrapolation nowcast for 1 timestep
        logger.info("Performing extrapolation nowcast")
        extrapolate = nowcasts.get_method("extrapolation")
        forecast = extrapolate(R[-1], V, 1)  # forecast 1 timestep ahead
        logger.info(f"Forecast completed. Output shape: {forecast.shape}")
        
        # Convert back to dBZ
        logger.debug("Converting forecast back to dBZ")
        forecast_dbz = rf_to_dbz(forecast[0])

        # Save internally in NC_TEMP_DIR
        interp_path = os.path.join(PYSTEPS_TEMP_DIR, f"one_lead_{station}_{target_time.strftime('%d%b%Y_%H%M')}.tif")
        print(f"Saving forecast to: {interp_path}")
        logger.debug(f"Saving temporary forecast to: {interp_path}")
        with rasterio.open(interp_path, "w", **profile) as dst:
            dst.write(forecast_dbz.astype(np.float32), 1)

        # # Also save into original radar folder path so it can be reused
        folder_path = os.path.join(RADAR_ROOT, f"Realtime_radar_INDIA_{target_time.strftime('%d%b%Y_%H%M')}",
                                   "Processing_results/clipped_tifs")
        os.makedirs(folder_path, exist_ok=True)
        save_name = f"caz_{station}_{target_time.strftime('%d%b%Y_%H%M')}_extract_fill_georef_clip_refl.tif"
        full_save_path = os.path.join(folder_path, save_name)
        
        logger.info(f"Saving forecast to radar folder: {full_save_path}")
        with rasterio.open(full_save_path, "w", **profile) as dst:
            dst.write(forecast_dbz.astype(np.float32), 1)

        logger.info(f"✅ One-step forecast created successfully for {station} at {target_time}")
        return full_save_path
        
    except Exception as e:
        logger.error(f"❌ Error in create_one_lead_forecast for {station}: {e}", exc_info=True)
        return None

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
        
        # save observed data to station_obs_dir
        logger.info(f"Saving observed data to: {station_obs_dir}")
        for i in range(len(times)):
            obs_path = os.path.join(station_obs_dir, f"observed_{station}_{times[i].strftime('%d%b%Y_%H%M')}.tif")
            with rasterio.open(obs_path, "w", **profile) as dst:
                dst.write(dbz_stack[i].astype(np.float32), 1)

        # Convert to rainfall rate
        logger.info("Converting full stack to rainfall rate")
        R = dbz_to_rf(dbz_stack)
        
        # Ensure we have valid data
        valid_pixels = np.sum(R > 0)
        total_pixels = R.size
        logger.info(f"Valid data in stack: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        
        if R.size == 0 or np.all(R == 0):
            logger.error(f"No valid radar data for station {station}")
            return
            
        # Calculate optical flow using all available data
        logger.info("Calculating optical flow for multi-timestep forecast")
        V = motion.get_method("LK")(R)
        logger.info(f"Optical flow calculated for {len(times)} timesteps")

        # Run extrapolation forecast
        logger.info("Running 6-timestep extrapolation forecast")
        extrapolate = nowcasts.get_method("extrapolation")
        forecast = extrapolate(R[-1], V, 6)  # 6 timesteps ahead
        logger.info(f"6-timestep forecast completed. Shape: {forecast.shape}")

        # Save forecast timesteps
        logger.info("Saving individual forecast timesteps")
        for i in range(6):
            timestamp = times[-1] + timedelta(minutes=10 * (i + 1))
            out_path = os.path.join(outdir, f"forecast_{station}_{timestamp.strftime('%d%b%Y_%H%M')}.tif")
            out_arr = rf_to_dbz(forecast[i].astype(np.float32))

            logger.debug(f"Saving timestep {i+1}/6 to: {os.path.basename(out_path)}")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_arr, 1)
            
            logger.debug(f"Timestep {i+1} saved. Range: [{np.min(out_arr):.2f}, {np.max(out_arr):.2f}] dBZ")

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
    logger.info("STARTING PYSTEPS RADAR NOWCASTING SYSTEM")
    logger.info("="*80)
    
    forecast_time = datetime.strptime("25Jun2025_1710", "%d%b%Y_%H%M")
    stations = ['bhp', 'ptl', 'ngp', 'gop', 'agt', 'cpj', 'goa', 'vrv', 'bhj', 'hyd', 'kkl', 'koc',
 'tvm', 'pdp', 'kol', 'mbr', 'ptn', 'lkn', 'delhi', 'cni', 'vsk', 'mpt', 'shr',
 'jpr', 'srn', 'jot', 'mks', 'ldn', 'sur', 'bnh', 'jmu', 'kuf', 'mur', 'slp']
    
    logger.info(f"Forecast time: {forecast_time}")
    logger.info(f"Processing {len(stations)} stations: {stations}")
    
    successful_stations = []
    failed_stations = []
    
    for station_idx, station in enumerate(stations):
        logger.info("="*60)
        logger.info(f"Processing station {station_idx+1}/{len(stations)}: {station}")
        logger.info("="*60)
        
        times = []
        files = []
        missing_files = []

        # Collect required timesteps
        for i in range(6):
            t = forecast_time - timedelta(minutes=10 * (5 - i))
            logger.debug(f"Looking for timestep {i+1}/6: {t}")
            
            f = find_file_for_time(station, t)
            if f:
                files.append(f)
                times.append(t)
                logger.info(f"✅ Found file for {t.strftime('%H%M')}: {os.path.basename(f)}")
            else:
                missing_files.append((i, t))
                logger.warning(f"❌ Missing file for {t.strftime('%H%M')}")
                
                # Try to estimate missing file using PySTEPS
                if i >= 2:  # Only try to estimate if we have at least 2 previous files
                    prev1 = find_file_for_time(station, t - timedelta(minutes=10))
                    prev2 = find_file_for_time(station, t - timedelta(minutes=20))
                    
                    if prev1 and prev2:
                        logger.info(f"Attempting to estimate missing {t.strftime('%H%M')} using PySTEPS")
                        try:
                            forecast_file = create_one_lead_forecast(prev2, prev1, t, station)
                            if forecast_file:
                                files.append(forecast_file)
                                times.append(t)
                                logger.info(f"✅ Successfully estimated file for {t.strftime('%H%M')}")
                            else:
                                logger.error(f"❌ Failed to create forecast for {station} at {t}")
                                break
                        except Exception as e:
                            logger.error(f"❌ Forecast estimation failed for {station} at {t}: {e}")
                            break
                    else:
                        logger.error(f"❌ Cannot estimate {t.strftime('%H%M')} for {station}: insufficient prior data")
                        logger.debug(f"prev1: {prev1 is not None}, prev2: {prev2 is not None}")
                        break
                else:
                    logger.error(f"❌ Cannot estimate {t.strftime('%H%M')} for {station}: need more historical data")
                    break

        # Check if we have enough data
        logger.info(f"Data collection summary for {station}:")
        logger.info(f"  - Required files: 6")
        logger.info(f"  - Found files: {len(files)}")
        logger.info(f"  - Missing files: {len(missing_files)}")
        
        if len(files) == 6:
            logger.info(f"✅ Sufficient data available for {station}. Starting forecast generation.")
            
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
            logger.warning(f"⏭️ Skipping station {station} due to insufficient data ({len(files)}/6 files)")
            failed_stations.append((station, f"Insufficient data: {len(files)}/6 files"))

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
    logger.info("PYSTEPS RADAR NOWCASTING SYSTEM FINISHED")
    logger.info("="*80)


# Available stations for reference:
# stations = ['goa', 'vrv', 'koc', 'kkl', 'cni', 'pdp', 'kol', 'gop', 'vsk', 'mpt', 'tvm', 'shr', 'hyd']  
# Full list: ['bhp', 'ptl', 'ngp', 'gop', 'agt', 'cpj', 'goa', 'vrv', 'bhj', 'hyd', 'kkl', 'koc',
#  'tvm', 'pdp', 'kol', 'mbr', 'ptn', 'lkn', 'delhi', 'cni', 'vsk', 'mpt', 'shr',
#  'jpr', 'srn', 'jot', 'mks', 'ldn', 'sur', 'bnh', 'jmu', 'kuf', 'mur', 'slp']