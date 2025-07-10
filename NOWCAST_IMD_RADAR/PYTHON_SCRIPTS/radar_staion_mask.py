import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union
from rasterio.transform import from_bounds
import numpy as np
# === Inputs ===
timestamp_str = "25Jun2025_1710"  # Example, replace with sys.argv[1] if needed
forecast_dir_base = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/forecast_output_{timestamp_str}"
buffer_shapefile_dir = "/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/SHAPE_FILES_CSVS_ALL_REGIONS"
india_shapefile = "/home/vassar/Downloads/india_boundary/india_boundary.shp"  # Provide the actual path
uncoverd_region_mask = f"/home/vassar/Documents/Weather_Product/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_STATIONS/psyteps_station_wise_{timestamp_str}/radar_uncovered_region_mask_{timestamp_str}.tif"

# === region_codes dictionary ===
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

# === Step 3: Read India boundary and buffers
india = gpd.read_file(india_shapefile).to_crs("EPSG:4326")
buffers = [gpd.read_file(path).to_crs("EPSG:4326") for path in buffer_paths]
buffers_union = unary_union([geom for gdf in buffers for geom in gdf.geometry])

uncovered_geom = unary_union(india.geometry).difference(buffers_union)

# === Step 5: Rasterize the uncovered region (value=1), everything else = 0
# Use bounds and resolution of India


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
