import os
import glob
import rasterio
import numpy as np
import xarray as xr
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from datetime import datetime
from collections import defaultdict
import time
import sys
import geopandas as gpd
from rasterio.features import rasterize
from shapely.ops import unary_union
from rasterio.transform import from_bounds
from rasterio.transform import from_origin
import time

st = time.time()

if len(sys.argv) != 2:
    print("Usage: python mosaic_station.py <timestamp_str>")
    sys.exit(1)
    
timestamp_str = sys.argv[1]
# === Root output folder ===
forecast_dir_base = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/forecast_output_{timestamp_str}"
observed_dir_base = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/observed_data_{timestamp_str}"
buffer_shapefile_dir = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/SHAPE_FILES_CSVS_ALL_REGIONS"
india_shapefile = "/home/vassar/Downloads/india_boundary/india_boundary.shp"  # Provide the actual path
uncoverd_region_mask = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/radar_uncovered_region_mask_{timestamp_str}.tif"

Fore_obs_base = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA"
Fore_obs_pngs_folder = os.path.join(Fore_obs_base, f"Forecast_obs_pngs_{timestamp_str}")
os.makedirs(Fore_obs_pngs_folder, exist_ok=True)

forecast_mosaic_dir = os.path.join(Fore_obs_pngs_folder, f"forecast_mosaics_{timestamp_str}")
observed_mosaic_dir = os.path.join(Fore_obs_pngs_folder, f"observed_mosaics_{timestamp_str}")
os.makedirs(forecast_mosaic_dir, exist_ok=True)
os.makedirs(observed_mosaic_dir, exist_ok=True)

Forecast_folder = os.path.join(Fore_obs_pngs_folder, f"Forecast_folder_{timestamp_str}")
Observ_folder = os.path.join(Fore_obs_pngs_folder, f"Observ_folder_{timestamp_str}")
os.makedirs(Forecast_folder, exist_ok=True)
os.makedirs(Observ_folder, exist_ok=True)

# === Output NetCDF paths ===
forecast_nc_stack = os.path.join(Forecast_folder, f"ds_precip_forecast_rf_{timestamp_str}.nc")
observed_nc_stack = os.path.join(Observ_folder, f"ds_precip_observ_rf_{timestamp_str}.nc")

# === Reference raster for alignment ===
ref_tiff_path = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/Realtime_radar_02Apr2024_1622_dummy_reference/output_mosaic_rainfallrate/mosaic_reflectivity_nodata_02Apr2024_1622.tif"


os.makedirs(forecast_mosaic_dir, exist_ok=True)
os.makedirs(observed_mosaic_dir, exist_ok=True)


def regrid_to_reference_array(mosaic_array, src_profile, ref_path):
    with rasterio.open(ref_path) as ref:
        dst_data = np.zeros((ref.height, ref.width), dtype=np.int16)

        reproject(
            source=mosaic_array,
            destination=dst_data,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest
        )

        # Use reference raster's full profile
        profile = ref.profile.copy()
        profile.update({
            "dtype": "int16",
            "count": 1
        })

        return dst_data.astype(np.int16), profile




def collect_files_by_timestamp(base_dir, prefix):
    files_by_time = defaultdict(list)
    station_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for station in station_dirs:
        station_path = os.path.join(base_dir, station)
        tif_files = glob.glob(os.path.join(station_path, f"{prefix}_{station}_*.tif"))
        for f in tif_files:
            fname = os.path.basename(f)
            time_part = fname.replace(f"{prefix}_{station}_", "").replace(".tif", "")
            time_part = time_part.replace("_rf", "")
            try:
                ts = datetime.strptime(time_part, "%d%b%Y_%H%M")
                files_by_time[ts].append(f)
            except Exception as e:
                print(f"⚠️ Error parsing {fname}: {e}")
    return files_by_time


def mosaic_and_regrid_all(file_dict, prefix, output_dir, ref_path):
    all_regrids = []
    all_times = []

    for ts, file_list in sorted(file_dict.items()):
        if len(file_list) == 0:
            continue

        print(f"🧩 Creating mosaic for {prefix} at {ts.strftime('%H%M')} with {len(file_list)} files")
        srcs = [rasterio.open(fp) for fp in file_list]
        mosaic, transform = merge(srcs, method="max")
        for s in srcs:
            s.close()

        with rasterio.open(file_list[0]) as ref:
            profile = ref.profile
            
        profile.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "dtype": "float32",
        })

        # Regrid and get corrected profile from reference raster
        regrid_array, ref_profile = regrid_to_reference_array(mosaic[0], profile, ref_tiff_path)

        # Save with correct extent and type
        regrid_path = os.path.join(output_dir, f"regrid_{prefix}_{ts.strftime('%d%b%Y_%H%M')}.tif")
        with rasterio.open(regrid_path, "w", **ref_profile) as dst:
            dst.write(regrid_array, 1)


        # Store for stacking
        with rasterio.open(regrid_path) as src:
            arr = src.read(1).astype(np.float32)
        all_regrids.append(arr)
        all_times.append(ts)

    return all_regrids, all_times


def save_stack_to_nc(array_stack, times, output_nc_path):
    print("times: ", times)
    if len(array_stack) == 0:
        print(f"⚠️ No data to stack for {output_nc_path}")
        return

    arr_stack = np.stack(array_stack, axis=0)
    ysize, xsize = arr_stack.shape[1:]

    with rasterio.open(ref_tiff_path) as ref:
        transform = ref.transform
        lon = np.array([transform * (i + 0.5, 0.5) for i in range(xsize)])[:, 0]
        lat = np.array([transform * (0.5, j + 0.5) for j in range(ysize)])[:, 1]

    # === Assign coordinates with CF attributes
    ds = xr.Dataset({
        "rf": (["time", "latitude", "longitude"], arr_stack)
    }, coords={
        "time": times,
        "latitude": ("latitude", lat, {"units": "degrees_north"}),
        "longitude": ("longitude", lon, {"units": "degrees_east"})
    })

    ds.attrs["Conventions"] = "CF-1.6"
    ds["rf"].attrs["coordinates"] = "longitude latitude"

    ds.to_netcdf(output_nc_path)
    print(f"✅ Saved CF-compliant NetCDF stack to {output_nc_path}")



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


# === Main Logic ===
if __name__ == "__main__":
    print("🔍 Collecting observed .tif files...")
    observed_files = collect_files_by_timestamp(observed_dir_base, "observed")

    print("🔍 Collecting forecasted .tif files...")
    forecast_files = collect_files_by_timestamp(forecast_dir_base, "forecast")

    print("\n🌀 Generating observed mosaics and stacking...")
    obs_stack, obs_times = mosaic_and_regrid_all(observed_files, "observed", observed_mosaic_dir, ref_tiff_path)
    save_stack_to_nc(obs_stack, obs_times, observed_nc_stack)

    print("\n🌩️ Generating forecast mosaics and stacking...")
    forecast_stack, forecast_times = mosaic_and_regrid_all(forecast_files, "forecast", forecast_mosaic_dir, ref_tiff_path)
    save_stack_to_nc(forecast_stack, forecast_times, forecast_nc_stack)
    
    
    
    print("\n🔄 Regridding NetCDFs to IWM 0.05° grid...")

    # === IWM Grid (Whole India 0.05 degree) ===
    cdo_regrd = '''gridtype  = lonlat
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

    # Save CDO grid description
    with open("cdo_regrd.txt", 'w') as file2write:
        file2write.write(cdo_regrd)

    # === Define output paths ===


    f_iwm = os.path.join(Forecast_folder, f"forecast_rf_iwm_{timestamp_str}.nc")
    o_iwm = os.path.join(Observ_folder, f"observ_rf_iwm_{timestamp_str}.nc")

    start1 = time.time()

    # Regrid with CDO (bilinear)
    # os.system(f'cdo -f nc remapbil,cdo_regrd.txt {forecast_nc_stack} {f_iwm}')
    # os.system(f'cdo -f nc remapbil,cdo_regrd.txt {observed_nc_stack} {o_iwm}')
    os.system(f'cdo -f nc remapcon,cdo_regrd.txt {forecast_nc_stack} {f_iwm}')
    os.system(f'cdo -f nc remapcon,cdo_regrd.txt {observed_nc_stack} {o_iwm}')
    # Optional cleanup
    # os.remove("cdo_regrd.txt")

    end1 = time.time()
    print('✅ Regridding complete.')
    print('🕒 Time taken for IWM regridding:', round((end1 - start1) / 60, 2), 'minutes')
    
    
    # Genertate radar uncovered region mask
    region_codes = {
    'Bhopal': 'bhp', 'Patiala': 'ptl', 'Nagpur': 'ngp', 'Gopalpur': 'gop',
    'Agartala': 'agt', 'Sohra': 'cpj', 'Goa': 'goa', 'Mumbai': 'vrv',
    'Bhuj': 'bhj', 'Hyderabad': 'hyd', 'Karaikal': 'kkl', 'Kochi': 'koc',
    'Thiruvananthapuram': 'tvm', 'Paradip': 'pdp', 'Kolkata': 'kol',
    'Mohanbari': 'mbr', 'Patna': 'ptn', 'Lucknow': 'lkn', 'Delhi': 'delhi',
    'Chennai': 'cni', 'Vishakapatnam': 'vsk', 'Machilipatnam': 'mpt',
    'Sriharikota': 'shr', 'Jaipur': 'jpr', 'Srinagar': 'srn', 'Jot': 'jot',
    'Mukteshwar': 'mks', 'Lansdowne': 'ldn', 'Surkandaji': 'sur',
    'Banihal': 'bnh', 'Jammu': 'jmu', 'Kufri': 'kuf', 'Murari': 'mur',
    'Solapur': 'slp'
}
    # Reverse mapping: code → name
    code_to_name = {v: k for k, v in region_codes.items()}

    # === Step 1: Get active station codes
    active_station_codes = [name for name in os.listdir(forecast_dir_base)
                            if os.path.isdir(os.path.join(forecast_dir_base, name))]

    # === Step 2: Get corresponding station buffer shapefiles
    buffer_paths = []
    for code in active_station_codes:
        station_name = code_to_name.get(code)
        if station_name:
            shapefile_path = os.path.join(buffer_shapefile_dir, f"{station_name}_buffer.shp")
            if os.path.exists(shapefile_path):
                buffer_paths.append(shapefile_path)

    print(f"✅ Found {len(buffer_paths)} matching station buffer shapefiles")
    # print not found shapefiles from dictionary
    not_found_shapefiles = [name for name in region_codes.keys() if f"{name}_buffer.shp" not in os.listdir(buffer_shapefile_dir)]
    if not_found_shapefiles:
        print("⚠️ The following station buffer shapefiles were not found:")
        for name in not_found_shapefiles:
            print(f"- {name} (code: {region_codes[name]})")

    # === Step 3: Read India boundary and buffers
    india = gpd.read_file(india_shapefile).to_crs("EPSG:4326")
    buffers = [gpd.read_file(path).to_crs("EPSG:4326") for path in buffer_paths]
    buffers_union = unary_union([geom for gdf in buffers for geom in gdf.geometry])

    uncovered_geom = unary_union(india.geometry).difference(buffers_union)

    # === Step 5: Rasterize the uncovered region (value=1), everything else = 0
    # Use bounds and resolution of India



    # Accumulated

    accum_tif_observ = os.path.join(Fore_obs_pngs_folder, "accumulated_last1hr_rf" + str(timestamp_str) + ".tif")
    accumulate_rainfall(observed_nc_stack, accum_tif_observ)

    accum_tif_fore = os.path.join(Fore_obs_pngs_folder, "accumulated_next1hr_rf" + str(timestamp_str) + ".tif")
    accumulate_rainfall(forecast_nc_stack, accum_tif_fore)
    
    bounds = india.total_bounds  # (minx, miny, maxx, maxy)
    res = 0.01  # degrees (~1km); adjust as needed

    width = int((bounds[2] - bounds[0]) / res)
    height = int((bounds[3] - bounds[1]) / res)
    transform = from_bounds(*bounds, width=width, height=height)

    mask = rasterize(
        [(uncovered_geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )
    mask = mask.astype("float32")
    mask[mask == 0] = np.nan

    # === Step 6: Save the mask as GeoTIFF
    with rasterio.open(
        uncoverd_region_mask,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(mask, 1)

    print(f"🎯 Radar gap mask saved to: {uncoverd_region_mask}")
    
    et = time.time()
    print(f'🕒 Total time taken for mosaic and mask creation: {(et - st) :.2f} secs')
    

    
