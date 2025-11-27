import json
import os
import pandas as pd
import pyproj
from pyproj import Proj
from pyproj import Transformer
import math
import numpy as np

def load_geojson_recalculate(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2100")
    points = []
    sensor_points = {}
    for feature in data.get("features", []):
        lon, lat = feature["geometry"]["coordinates"]
        sensor_id = feature["properties"].get("sensor_id")

        # projection 1: WGS84
        p1 = Proj('epsg:4326', preserve_units=False)
        # projection 2: GGRS87 / Greek Grid
        p2 = Proj('epsg:2100', preserve_units=False)

        #x1, y1 = pyproj.transform(p1, p2, lon, lat)
        x1, y1 = transformer.transform(lon, lat)
        #print(x1, y1)
        points.append((x1, y1))
        sensor_points[sensor_id] = (x1, y1)
    return sensor_points



def load_and_normalize_geojson(file_path, width=256, height=256):
    """
    Loads a GeoJSON file and normalizes the point coordinates to fit within a
    width x height bitmap.

    Args:
        file_path (str): Path to the GeoJSON file.
        width (int): The width of the output bitmap (default 256).
        height (int): The height of the output bitmap (default 256).

    Returns:
        dict: A dictionary where the key is sensor_id and the value is a tuple (x, y)
              of normalized coordinates.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    points = []
    for feature in data.get("features", []):
        coords = feature["geometry"]["coordinates"]
        points.append(coords)

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    normalized_points = {}
    for feature in data.get("features", []):
        sensor_id = feature["properties"].get("sensor_id")
        lon, lat = feature["geometry"]["coordinates"]

        if max_x - min_x != 0:
            norm_x = (lon - min_x) / (max_x - min_x) * width
        else:
            norm_x = width / 2

        if max_y - min_y != 0:
            norm_y = (lat - min_y) / (max_y - min_y) * height
        else:
            norm_y = height / 2

        normalized_points[sensor_id] = (norm_x, norm_y)

    return normalized_points


def load_sensor_data(folder_path):
    """
    Reads all CSV files in the given folder, extracts the 'Total' column as a list of integers,
    and stores it in a dictionary where the key is the sensorId from the filename.

    Parameters:
        folder_path (str): The path to the folder containing the CSV files.

    Returns:
        dict: A dictionary where keys are sensor IDs and values are lists of integers from 'Total' column.
    """
    sensor_data = {}
    sensor_data_lengths = {}
    tmp_len = 0
    for filename in os.listdir(folder_path):
        sensor_id = filename.split('.')[0]
        if sensor_id.startswith('Alle0LSA'):
            continue
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path)

            if 'Total' in df.columns:
                total_list = df['Total'].dropna().astype(int).tolist()
                sensor_data[sensor_id] = total_list
                tmp_len = len(total_list)
            else:
                print(f"Warning: 'Total' column not found in {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return sensor_data, tmp_len


###########################################################################

def clean_keys(sensors_combined, all_keys):
    clen_sensors_combined = {}
    for k in all_keys:
        if k in sensors_combined.keys():
            clen_sensors_combined[k] = sensors_combined[k]
    return clen_sensors_combined

#clean sensors_combined
def clean_keys_remove_zero(sensors_combined, all_keys, sensors_values, which_to_remove):
    clen_sensors_combined = {}
    k_id = 0
    for k in all_keys:
        if k not in which_to_remove:
            if k in sensors_combined.keys():
                val = sensors_values[k]
                all_zeros = True
                for a in range(len(val)):
                    if math.fabs(val[a]) > 0.00001:
                        all_zeros = False
                if not all_zeros:
                    clen_sensors_combined[k] = sensors_combined[k]
        else:
            xxx = 1
        k_id += 1
    return clen_sensors_combined

def load_data_and_combine_with_geojson(folder_to_load, geojson_path):
    sensors_combined = load_geojson_recalculate(geojson_path)
    sensors_values, length = load_sensor_data(folder_to_load)

    all_keys = list(sensors_values.keys())
    clen_sensors_combined = clean_keys_remove_zero(sensors_combined, all_keys, sensors_values, ['A173'])
    clen_sensors_combined_keys = list(clen_sensors_combined.keys())

    full_data = np.zeros((len(sensors_values[clen_sensors_combined_keys[0]]), len(clen_sensors_combined_keys)))
    for b in range(len(clen_sensors_combined)):
        val = sensors_values[clen_sensors_combined_keys[b]]
        for a in range(len(sensors_values[clen_sensors_combined_keys[0]])):
            if a < len(val):
                full_data[a, b] = val[a]
            else:
                full_data[a, b] = 0
    return [clen_sensors_combined, full_data, all_keys]


def calculate_route_distance(clen_sensors_combined):
    route_distances = np.zeros((len(clen_sensors_combined), len(clen_sensors_combined)))
    help_keys = list(clen_sensors_combined.keys())

    all_points = []
    for a in range(len(help_keys)):
        all_points.append(clen_sensors_combined[help_keys[a]])
        for b in range(a, len(help_keys)):
            my_key_a = help_keys[a]
            my_key_b = help_keys[b]
            route_distances[b, a] = route_distances[a, b] = math.sqrt(math.pow((clen_sensors_combined[my_key_a][0] - clen_sensors_combined[my_key_b][0]), 2) +
                                        math.pow((clen_sensors_combined[my_key_a][0] - clen_sensors_combined[my_key_b][0]), 2))
    return [route_distances, all_points]

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    """
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std
    """

    train_array = data_array[:num_train]
    val_array = data_array[num_train: (num_train + num_val)]
    test_array = data_array[(num_train + num_val):]

    return train_array, val_array, test_array
