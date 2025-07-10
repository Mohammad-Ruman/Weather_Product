# import os
# import numpy as np
# import rasterio
# from datetime import datetime, timedelta
# from pysteps import motion, nowcasts
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()

# RADAR_ROOT = "./interpolated_outputs"
# os.makedirs(RADAR_ROOT, exist_ok=True)

# file_1130 = "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1610/Processing_results/clipped_tifs/caz_vsk_25Jun2025_1530_extract_fill_georef_clip_refl.tif"
# file_1230 = "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1710/Processing_results/clipped_tifs/caz_vsk_25Jun2025_1630_extract_fill_georef_clip_refl.tif"
# station = "vsk"

# t1 = datetime(2025, 6, 25, 15, 30)
# t2 = datetime(2025, 6, 25, 16, 30)

# def interpolate_radar_simple(arr1, motion_field, weight):
#     """
#     Interpolate using only forward extrapolation from arr1
    
#     Parameters:
#     -----------
#     arr1: 2D numpy array
#         Radar image at t1
#     arr2: 2D numpy array
#         Radar image at t2 (not used in this version)
#     motion_field: 3D numpy array
#         Motion field from pysteps
#     weight: float (0-1)
#         Interpolation weight (0=arr1, 1=arr2)
    
#     Returns:
#     --------
#     interpolated: 2D numpy array
#         Interpolated radar image (forward extrapolation only)
#     """
    
#     extrapolate = nowcasts.get_method("extrapolation")
#     n_leadtimes = 1
    
#     # Extrapolate arr1 forward using scaled motion field
#     forward_extrap = extrapolate(arr1, motion_field * weight, n_leadtimes)
#     forward_result = forward_extrap[0]
#     print("Shape of forward extrapolation result:", forward_result.shape)

    
#     return forward_result



# try:
#     # Step 1: Load input images
#     with rasterio.open(file_1130) as src1, rasterio.open(file_1230) as src2:
#         arr1 = src1.read(1).astype(np.float32)
#         arr2 = src2.read(1).astype(np.float32)
#         profile = src1.profile
        
#     logger.info(f"Loaded radar data - Shape: {arr1.shape}")
#     logger.info(f"Data range - File 1: {arr1.min():.2f} to {arr1.max():.2f}")
#     logger.info(f"Data range - File 2: {arr2.min():.2f} to {arr2.max():.2f}")
    
#     # Step 2: Preprocess data - handle invalid values
#     arr1_processed = arr1.copy()
#     arr2_processed = arr2.copy()
    
#     # Step 3: Calculate motion field using first and last frame
#     train_precip = np.stack([arr1_processed, arr2_processed], axis=0)
    
#     # Use Lucas-Kanade optical flow
#     oflow_method = motion.get_method("LK")
#     motion_field = oflow_method(train_precip)
#     print("Shape of motion field",motion_field.shape)
    
#     logger.info(f"Motion field shape: {motion_field.shape}")
#     logger.info(f"Motion field range: {motion_field.min():.3f} to {motion_field.max():.3f}")
#     logger.info(f"Motion field magnitude: {np.sqrt(motion_field[0]**2 + motion_field[1]**2).mean():.3f}")
    
#     # Step 4: Create interpolated images at intermediate times
#     interpolated_files = {}
#     validation_results = {}
    
#     for i in range(1, 6):  # 5 intermediate steps
#         target_time = t1 + timedelta(minutes=i * 10)
#         weight = i / 6  # total 6 steps between t1 and t2
        
#         logger.info(f"Interpolating {target_time.strftime('%H:%M')} with weight {weight:.3f}")
        
#         # Create interpolated image using pysteps extrapolation
#         interpolated = interpolate_radar_simple(arr1_processed, motion_field, weight)
        
#         # Validate results
#         time_str = target_time.strftime("%H%M")
#         # Save to TIFF
#         save_name = f"caz_{station}_{target_time.strftime('%d%b%Y_%H%M')}_interpolated.tif"
#         full_save_path = os.path.join(RADAR_ROOT, save_name)
        
#         # Update profile for output
#         output_profile = profile.copy()
#         output_profile.update(dtype=rasterio.float32, count=1)
        
#         with rasterio.open(full_save_path, "w", **output_profile) as dst:
#             dst.write(interpolated.astype(np.float32), 1)
        
#         interpolated_files[time_str] = full_save_path
        
#         logger.info(f"Saved: {save_name}")
#         logger.info(f"Interpolated data range: {interpolated.min():.2f} to {interpolated.max():.2f}")
#         print()  # Add spacing between time steps

#     print("📂 Interpolated files:")
#     for time_str, path in interpolated_files.items():
#         print(f"{time_str} → {path}")
        
#     # Overall quality assessment
#     print("\n📊 Overall Quality Assessment:")
#     print(f"Original time gap: {(t2 - t1).total_seconds() / 60:.0f} minutes")
#     print(f"Interpolated time step: {((t2 - t1).total_seconds() / 60) / 6:.0f} minutes")
    
#     # Check validation results
#     all_valid = all(result['range_ok'] for result in validation_results.values())
#     avg_mean_diff = np.mean([result['mean_diff'] for result in validation_results.values()])
    
#     if all_valid:
#         print("✅ All interpolated images have valid value ranges")
#     else:
#         print("⚠️  Some interpolated images have values outside expected range")
        
#     print(f"Average mean difference: {avg_mean_diff:.3f}")
    
#     if avg_mean_diff < 1.0:
#         print("✅ Good interpolation quality - means are well-preserved")
#     else:
#         print("⚠️  Large mean differences detected - check motion field quality")
    
#     print("\n🎯 Solution Benefits:")
#     print("• No duplicate pixels at different locations")
#     print("• Uses pysteps native extrapolation method")
#     print("• Proper temporal interpolation with weighted motion fields")
#     print("• Maintains physical consistency")

# except Exception as e:
#     logger.error(f"Error during processing: {str(e)}")
#     import traceback
#     traceback.print_exc()


import os
import numpy as np
import rasterio
from datetime import datetime, timedelta
from pysteps import motion, nowcasts
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# --------------------------------------------
# Configuration
# --------------------------------------------
station = "vsk"

# Actual time inputs (earlier = t0, later = t6)
t0 = datetime(2025, 6, 25, 15, 30)  # A
t6 = datetime(2025, 6, 25, 16, 30)  # B

# Interpolated timestamps between 1530 and 1630
timestamps_interp = [t0 + timedelta(minutes=10 * i) for i in range(1, 6)]  # 1540, ..., 1620

# Input file paths
files = [
    "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1610/Processing_results/clipped_tifs/caz_vsk_25Jun2025_1530_extract_fill_georef_clip_refl.tif",  # 1530
    "NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/Realtime_radar_INDIA_25Jun2025_1710/Processing_results/clipped_tifs/caz_vsk_25Jun2025_1630_extract_fill_georef_clip_refl.tif"   # 1630
]

# Output folders
ROOT_DIR = "./interpolated_mean_outputs"
A2B_DIR = os.path.join(ROOT_DIR, "a_to_b")
B2A_DIR = os.path.join(ROOT_DIR, "b_to_a")
MEAN_DIR = os.path.join(ROOT_DIR, "mean_interpolation")
os.makedirs(A2B_DIR, exist_ok=True)
os.makedirs(B2A_DIR, exist_ok=True)
os.makedirs(MEAN_DIR, exist_ok=True)

# --------------------------------------------
# Load radar TIFFs
# --------------------------------------------
stack = []
profile = None

for fpath in files:
    with rasterio.open(fpath) as src:
        arr = src.read(1).astype(np.float32)
        stack.append(arr)
        if profile is None:
            profile = src.profile

arr_1530 = stack[0]  # A
arr_1630 = stack[1]  # B

# --------------------------------------------
# Interpolation A to B (1530 ➝ 1630)
# --------------------------------------------
motion_field_ab = motion.get_method("LK")(np.stack([arr_1530, arr_1630]))
extrapolate = nowcasts.get_method("extrapolation")

a_to_b_results = []
for i, ts in enumerate(timestamps_interp, start=1):
    weight = i / 6
    interp = extrapolate(arr_1530, motion_field_ab * weight, 1)[0]
    a_to_b_results.append(interp)

    save_name = f"caz_{station}_{ts.strftime('%d%b%Y_%H%M')}_interpolated.tif"
    save_path = os.path.join(A2B_DIR, save_name)
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(interp.astype(np.float32), 1)
    logger.info(f"[A→B] Saved: {save_name}")

# --------------------------------------------
# Interpolation B to A (1630 ➝ 1530)
# --------------------------------------------
motion_field_ba = motion.get_method("LK")(np.stack([arr_1630, arr_1530]))

b_to_a_results = []
for i, ts in enumerate(timestamps_interp, start=1):
    weight = i / 6
    interp = extrapolate(arr_1630, -motion_field_ba * weight, 1)[0]
    b_to_a_results.append(interp)

    save_name = f"caz_{station}_{ts.strftime('%d%b%Y_%H%M')}_interpolated.tif"
    save_path = os.path.join(B2A_DIR, save_name)
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(interp.astype(np.float32), 1)
    logger.info(f"[B→A] Saved: {save_name}")

# --------------------------------------------
# Mean of A→B and B→A
# --------------------------------------------
for i, ts in enumerate(timestamps_interp):
    arr_mean = (a_to_b_results[i] + b_to_a_results[i]) / 2.0

    save_name = f"caz_{station}_{ts.strftime('%d%b%Y_%H%M')}_interpolated.tif"
    save_path = os.path.join(MEAN_DIR, save_name)
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(arr_mean.astype(np.float32), 1)
    logger.info(f"[MEAN] Saved: {save_name}")
    
    
# --------------------------------------------
# Forecast using full mean-interpolated stack (1530 to 1630) → next 6 steps
# --------------------------------------------

# Reconstruct full 7-step mean stack: [1530, ..., 1630]
timestamps_full = [t0 + timedelta(minutes=10 * i) for i in range(7)]  # 1530 → 1630
mean_stack = [arr_1530] + [(a_to_b_results[i] + b_to_a_results[i]) / 2.0 for i in range(5)] + [arr_1630]
mean_stack = np.stack(mean_stack, axis=0)  # Shape: (7, H, W)

# Compute motion field from the full stack (can use last 2 or full stack — using full here)
motion_field_forecast = motion.get_method("LK")(mean_stack)
extrapolate = nowcasts.get_method("extrapolation")

# Use the last frame (1630) as the base for extrapolation
forecast_steps = 6
forecast_results = extrapolate(mean_stack[-1], motion_field_forecast, forecast_steps)

# Forecast timestamps: 16:40 → 17:30
forecast_times = [t6 + timedelta(minutes=10 * i) for i in range(1, forecast_steps + 1)]

# Save forecasts
for arr, ts in zip(forecast_results, forecast_times):
    save_name = f"caz_{station}_{ts.strftime('%d%b%Y_%H%M')}_forecast.tif"
    save_path = os.path.join(MEAN_DIR, save_name)
    with rasterio.open(save_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
    logger.info(f"[FORECAST] Saved: {save_name}")
