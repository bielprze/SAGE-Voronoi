# https://github.com/mensif/DCGRU_Tensorflow2

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import pickle
import helper
from helper import *

from synth_signal.gen_signal import generate_signal
from dcgru_cell_tf2 import DCGRUCell

#signal = pickle.load(open('signal.pickle','rb'))
#G_adj_mx = pickle.load(open('G_adj_mx.pickle','rb'))
#num_nodes_old = G_adj_mx.shape[0]


train_size, val_size = 0.5, 0.2
input_sequence_length = 12
forecast_horizon = 3
# Roads Graph
sigma2 = 0.1
#epsilon = 0.5
epsilon = 0.95
id = 0
##########################################
folder_to_load = 'data_generator/2024-03-01_35/'
geojson_path = 'data_generator/sensors_location.geojson'

file_name = 'results/dcgru_results.csv'
with open(file_name, "a") as myfile:
       myfile.write("forecast_horizon, my_seed, mean_mse, sd_mse, mean_rmse, sd_rmse, mean_mae, sd_mae, mean_mre, sd_mre\n")

for id in range(10):
 for forecast_horizon in [1,2,3]:
        # Load data
        [clen_sensors_combined, full_data, all_keys] = helper.load_data_and_combine_with_geojson(folder_to_load, geojson_path)
        clen_sensors_combined_keys = list(clen_sensors_combined.keys())
        [route_distances, all_points] = calculate_route_distance(clen_sensors_combined)
        speeds_array = full_data

        print(f"route_distances shape={route_distances.shape}")
        print(f"speeds_array shape={speeds_array.shape}")

        ################
        train_array, val_array, test_array, scaler = preprocess(speeds_array, train_size, val_size)


        [X_train, y_train] = generate(train_array, input_sequence_length, forecast_horizon)
        [X_val, y_val] = generate(val_array, input_sequence_length, forecast_horizon)
        [X_test, y_test] = generate(test_array, input_sequence_length, forecast_horizon)

        X_train = np.expand_dims(np.array(X_train), -1)
        X_val = np.expand_dims(np.array(X_val), -1)
        X_test = np.expand_dims(np.array(X_test), -1)

        adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)

        num_nodes = adjacency_matrix.shape[0]
        adjacency_matrix = np.matrix(adjacency_matrix)
        # define the dcgru cell
        #dcgru_cell = DCGRUCell(units=20,adj_mx=G_adj_mx, K_diffusion=2, num_nodes=num_nodes,filter_type="random_walk")
        dcgru_cell = DCGRUCell(units=20,adj_mx=adjacency_matrix, K_diffusion=2, num_nodes=num_nodes,filter_type="random_walk")

        # wrap the dcgru cell in a keras RNN layer
        Dcgru_layer = keras.layers.RNN(dcgru_cell)

        model_dcgru = keras.models.Sequential([
          keras.Input(shape=(None, num_nodes, 1)),  # Input dimensions: sequence length (None := arbitrary length)
          Dcgru_layer,                              #                   number of nodes in the graph
          keras.layers.Dense(num_nodes * forecast_horizon)             #                   signal dimensionality (1 in the example)
          ])

        model_dcgru.summary()

        opt = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        model_dcgru.compile(loss="mse", optimizer=opt)

        # Uncomment to fit

        model_dcgru.fit(X_train, y_train, epochs=30, verbose=1)
        model_dcgru.save_weights('weights/model_dcgru_fh=' + str(forecast_horizon) + "_id=" + str(id) + '.h5')


        model_dcgru.load_weights('weights/model_dcgru_fh=' + str(forecast_horizon) + "_id=" + str(id) + '.h5')

        y_pred = model_dcgru.predict(X_test)
        shape2 = X_train.shape[2]
        start2 = 0
        stop2 = shape2

        ll_y = []
        ll_y_pred = []
        for ab in range(forecast_horizon):
            y_test_part = y_test[:,start2:stop2]
            y_pred_part = y_pred[:, start2:stop2]

            y_inv2 = scaler.inverse_transform(y_test_part)
            y_pred_inv2 = scaler.inverse_transform(y_pred_part)
            start2 += shape2
            stop2 += shape2

            ll_y.append(y_inv2)
            ll_y_pred.append(y_pred_inv2)

        y_inv = np.concatenate(ll_y, axis=1)
        y_pred_inv = np.concatenate(ll_y_pred, axis=1)

        np.save("results/y_inv", y_inv)
        np.save("results/y_pred_inv", y_pred_inv)

        evaluate_model(file_name, forecast_horizon, id, y_inv, y_pred_inv)
