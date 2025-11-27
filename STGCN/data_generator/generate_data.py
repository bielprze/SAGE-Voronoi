import helper
from helper import calculate_route_distance, preprocess
import numpy as np
my_seed = 0
folder_to_load = '2024-03-01_35/'
geojson_path = 'sensors_location.geojson'
# Splitting and normalizing data
train_size, val_size = 0.5, 0.2


[clen_sensors_combined, full_data, all_keys] = helper.load_data_and_combine_with_geojson(folder_to_load, geojson_path)
clen_sensors_combined_keys = list(clen_sensors_combined.keys())
[route_distances, all_points] = calculate_route_distance(clen_sensors_combined)

route_distances = route_distances/np.max(route_distances)

np.savetxt("full_data.csv", full_data, delimiter=",")
np.savetxt("graph_data.csv", route_distances, delimiter=",")
"""
speeds_array = full_data

print(f"route_distances shape={route_distances.shape}")
print(f"speeds_array shape={speeds_array.shape}")
"""
################
train_array, val_array, test_array, scaler = preprocess(full_data, train_size, val_size)
xxx = 0