import matplotlib.pyplot as plt
import numpy as np
ref = np.load("y_inv.npy")
stgcn = np.load("y_pred_inv.npy")

plt.ylabel('Traffic [number of cars per 10 minutes]')
id = 0
#plt.title("Crossing " + clen_sensors_combined_keys[which_id])
x = np.arange(0,ref.shape[0])
y = ref[:,id]
plt.plot(x, y)
x = np.arange(0,stgcn.shape[0])
y = stgcn[:,id]
plt.plot(x, y)
#plt.plot(list_datetime[from_id:to_id], y_pred_True[from_id:to_id, which_id], linewidth=ln)
#plt.plot(list_datetime[from_id:to_id], y_pred_simple[from_id:to_id, which_id], linewidth=ln)
plt.legend(["Ground truth", "SAGE", "SAGE-Voronoi","Pure LSTM"])
plt.show()