import os
import rasterio
import numpy as np
import xarray as xr
from datetime import datetime
from tqdm import tqdm
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling as RioResampling

# -------------------------------
# Configuration
# -------------------------------
IWM_SAVE_DIR = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/Forecast_obs_pngs_25Jun2025_1710/ensembeld_nc_files"
INPUT_ROOT = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_25Jun2025_1710/forecast_output_25Jun2025_1710"
REF_TIFF_PATH = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/Realtime_radar_02Apr2024_1622_dummy_reference/output_mosaic_rainfallrate/mosaic_reflectivity_nodata_02Apr2024_1622.tif"
os.makedirs(IWM_SAVE_DIR, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------
def get_lat_lon_from_tiff(tif_path):
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        lon = np.array([transform * (i + 0.5, 0.5) for i in range(width)])[:, 0]
        lat = np.array([transform * (0.5, j + 0.5) for j in range(height)])[:, 1]
        return lat, lon

def save_stack_to_nc(array_stack, times, output_nc_path, lat, lon):
    print("times: ", times)
    if len(array_stack) == 0:
        print(f"⚠️ No data to stack for {output_nc_path}")
        return

    arr_stack = np.stack(array_stack, axis=0)

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

def mosaic_and_regrid_all(member_folder):
    print(f"\n🌀 Mosaicking Member: {member_folder}")
    all_times = []
    mosaicked_arrays = []
    ref_profile = None

    # Collect all station folders for this member
    station_dirs = [os.path.join(INPUT_ROOT, station, member_folder) for station in os.listdir(INPUT_ROOT)
                    if os.path.isdir(os.path.join(INPUT_ROOT, station, member_folder))]

    # Collect all timestamps available across stations
    time_to_files = {}
    for station_path in station_dirs:
        for fname in os.listdir(station_path):
            if fname.endswith(".tif"):
                timestamp = "_".join(fname.split("_")[-3:-1])
                time_to_files.setdefault(timestamp, []).append(os.path.join(station_path, fname))

    ref = rasterio.open(REF_TIFF_PATH)
    ref_profile = ref.profile
    ref_data = ref.read(1)

    for timestamp, files in sorted(time_to_files.items()):
        srcs = [rasterio.open(fp) for fp in files]
        mosaic, _ = merge(srcs, method="max")
        mosaic = mosaic[0]  # remove band dimension

        # Reproject to match ref tif
        regridded = np.zeros_like(ref_data, dtype=np.float32)
        reproject(
            source=mosaic,
            destination=regridded,
            src_transform=srcs[0].transform,
            src_crs=srcs[0].crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.bilinear
        )

        mosaicked_arrays.append(regridded)
        time_obj = datetime.strptime(timestamp, "%d%b%Y_%H%M")
        all_times.append(time_obj)

        for src in srcs:
            src.close()

    lat, lon = get_lat_lon_from_tiff(REF_TIFF_PATH)
    nc_path = os.path.join(IWM_SAVE_DIR, f"mosaic_{member_folder}.nc")
    save_stack_to_nc(mosaicked_arrays, all_times, nc_path, lat, lon)

    # === Regrid to IWM Grid ===
    iwm_nc_path = os.path.join(IWM_SAVE_DIR, f"iwm_mosaic_{member_folder}.nc")
    cdo_grid_txt = "cdo_regrd.txt"
    with open(cdo_grid_txt, "w") as f:
        f.write("""gridtype  = lonlat
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
yinc      = 0.05""")

    os.system(f"cdo -f nc remapcon,{cdo_grid_txt} {nc_path} {iwm_nc_path}")
    print(f"✅ IWM-regridded file saved to: {iwm_nc_path}")

# -------------------------------
# Run mosaicking for each member
# -------------------------------
stations = sorted(os.listdir(INPUT_ROOT))
example_station = stations[0]
members = sorted([d for d in os.listdir(os.path.join(INPUT_ROOT, example_station)) if d.startswith("member_")])

for member in tqdm(members, desc="🌍 Processing Ensemble Members"):
    mosaic_and_regrid_all(member)
