import fiona
from shapely.geometry import shape, Point
import numpy as np
import netCDF4
import os
import pandas as pd

# Step 1: Load the Vietnam boundary shapefile
shapefile_path = "D:/Disertasi/outgoing_20240529/outgoing/gadm41_VNM_shp/gadm41_VNM_0.shp"
with fiona.open(shapefile_path) as multipol:
    vietnam_boundary = next(iter(multipol))
vietnam_polygon = shape(vietnam_boundary['geometry'])

# Step 2: Create a 2-D regular grid structure
resolution = 0.1
land_sea_mask_path = "D:/Disertasi/outgoing_20240529/outgoing/gadm41_VNM_shp/IMERG_land_sea_mask.nc"
dataset = netCDF4.Dataset(land_sea_mask_path)
land_sea_mask = dataset.variables['landseamask'][:]
land_sea_mask_viet = land_sea_mask[980:1140, 1021:1101]
longitudes = dataset.variables['lon'][:]
latitudes  = dataset.variables['lat'][:]
lat_viet = latitudes[980:1140]
lon_viet = longitudes[1021:1101]
lat_min, lat_max = lat_viet[0], lat_viet[-1]
lon_min, lon_max = lon_viet[0], lon_viet[-1]
lat_viet = np.around(lat_viet, decimals=2)
lon_viet = np.around(lon_viet, decimals=2)
lon_grid, lat_grid = np.meshgrid(lon_viet, lat_viet)

# Step 3: Check if points are within the boundary
def is_within_boundary(lat, lon, polygon):
    point = Point(lon, lat)
    return point.within(polygon)

within_boundary = np.vectorize(is_within_boundary)(lat_grid, lon_grid, vietnam_polygon)
coordinates_within_boundary = np.column_stack((lat_grid[within_boundary], lon_grid[within_boundary]))
coordinates_within_boundary = np.around(coordinates_within_boundary, decimals=2)

# Save the coordinates within the boundary
output_dir = "D:/Disertasi/outgoing_20240529/outgoing/outputs"
os.makedirs(output_dir, exist_ok=True)
coordinates_within_vietnam_path = os.path.join(output_dir, "coordinates_within_vietnam.csv")
np.savetxt(coordinates_within_vietnam_path, coordinates_within_boundary, delimiter=",", header="latitude,longitude", comments='')


# Step 4: Match the grid coordinates with the land-sea mask values
# Assuming the mask and coordinates are on the same resolution and extent
masked_values = []

for lat, lon in coordinates_within_boundary:
    print(lat)
    latcx = np.where(lat_viet==lat)[0][0]
    loncx = np.where(lon_viet==lon)[0][0]
    masked_value = land_sea_mask_viet[latcx, loncx]
    masked_values.append([lat, lon, masked_value])

masked_values = np.array(masked_values)

# Save the masked values
masked_values_path = os.path.join(output_dir, "masked_values_within_vietnam.csv")
np.savetxt(masked_values_path, masked_values, delimiter=",", header="latitude,longitude,mask_value", comments='')

print("Masked values saved to:", masked_values_path)

# Step 5: Define thresholds and identify coastal locations
land_min_threshold, land_max_threshold = 40, 75
ocean_threshold = 60

def is_coastal(lat_index, lon_index, mask):
    is_coastal_location = False
    if land_min_threshold <= mask[lat_index, lon_index] <= land_max_threshold:
        neighbors = [
            (lat_index-1, lon_index), (lat_index+1, lon_index),
            (lat_index, lon_index-1), (lat_index, lon_index+1)
        ]
        for n in neighbors:
            if 0 <= n[0] < mask.shape[0] and 0 <= n[1] < mask.shape[1]:
                if mask[n[0], n[1]] >= ocean_threshold:
                    is_coastal_location = True
                    break  # Stop iteration when condition is True
    return is_coastal_location


# Identify coastal points
coastal_candidates = []
for i in range(lat_grid.shape[0]):
    for j in range(lat_grid.shape[1]):
        print("lat = ", lat_grid[i, j], "  lon = ", lon_grid[i, j], "mask = ", land_sea_mask_viet[i, j])
        if within_boundary[i, j]:
            if is_coastal(i, j, land_sea_mask_viet)==True:
                coastal_candidates.append((lat_grid[i, j], lon_grid[i, j], land_sea_mask_viet[i, j]))
                #print(f"Added coastal candidate at ({lat_grid[i, j]}, {lon_grid[i, j]}) with value {land_sea_mask_viet[i, j]}")

# Save the coastal candidates
coastal_candidates_path = os.path.join(output_dir, "coastal_candidates(40,75,60).csv")
np.savetxt(coastal_candidates_path, coastal_candidates, delimiter=",", header="latitude,longitude,value", comments='')


        
# Step 6: remove coastal points with lower values than their coastal neighbors
def remove_lower_value_coastal_points(coastal_candidates, land_sea_mask_viet, lat_viet, lon_viet):
    coastal_candidates_set = {(lat, lon): value for lat, lon, value in coastal_candidates}
    filtered_coastal_candidates = []

    for lat, lon, value in coastal_candidates:
        lat_index = np.where(lat_viet == lat)[0][0]
        lon_index = np.where(lon_viet == lon)[0][0]

        neighbors = [
            (lat_index-1, lon_index), (lat_index+1, lon_index),
            (lat_index, lon_index-1), (lat_index, lon_index+1)
        ]

        keep_point = True
        for n in neighbors:
            if 0 <= n[0] < land_sea_mask_viet.shape[0] and 0 <= n[1] < land_sea_mask_viet.shape[1]:
                neighbor_lat = lat_viet[n[0]]
                neighbor_lon = lon_viet[n[1]]
                neighbor_value = coastal_candidates_set.get((neighbor_lat, neighbor_lon), None)
                if neighbor_value is not None and neighbor_value > value:
                    keep_point = False
                    break

        if keep_point:
            filtered_coastal_candidates.append((lat, lon, value))

    return np.array(filtered_coastal_candidates)

# Apply the function to filter coastal candidates
filtered_coastal_candidates = remove_lower_value_coastal_points(coastal_candidates, land_sea_mask_viet, lat_viet, lon_viet)

# Save the filtered coastal candidates
filtered_coastal_candidates_path = os.path.join(output_dir, "filtered_coastal_candidates(40,75,60).csv")
np.savetxt(filtered_coastal_candidates_path, filtered_coastal_candidates, delimiter=",", header="latitude,longitude,value", comments='')

print("Filtered coastal candidates saved to:", filtered_coastal_candidates_path)