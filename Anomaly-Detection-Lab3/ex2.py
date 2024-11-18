from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import numpy as np

X_train, y_train = make_blobs(n_samples=500, centers=[[10, 0], [0, 10]])

plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()

forest_model = IForest(contamination=0.02)
dif_model = DIF(contamination=0.02)
loda_model = LODA(contamination=0.02)

forest_model.fit(X_train)
dif_model.fit(X_train)
loda_model.fit(X_train)

X_test = np.random.uniform(low=[-10, -10], high=[20, 20], size=(1000, 2))

plt.scatter(X_test[:, 0], X_test[:, 1])
plt.show()

forest_y_pred_train = forest_model.decision_function(X_train)
forest_y_pred_test= forest_model.decision_function(X_test)

dif_y_pred_train = dif_model.decision_function(X_train)
dif_y_pred_test= dif_model.decision_function(X_test)

loda_y_pred_train = loda_model.decision_function(X_train)
loda_y_pred_test= loda_model.decision_function(X_test)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

scatter = axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=forest_y_pred_train, cmap='viridis')
axes[0, 0].set_title("IForest Training Data")
fig.colorbar(scatter, ax=axes[0, 0], label='Anomaly Score')

scatter = axes[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=dif_y_pred_train, cmap='viridis')
axes[0, 1].set_title("DIF Training Data")
fig.colorbar(scatter, ax=axes[0, 1], label='Anomaly Score')

scatter = axes[0, 2].scatter(X_train[:, 0], X_train[:, 1], c=loda_y_pred_train, cmap='viridis')
axes[0, 2].set_title("LODA Training Data")
fig.colorbar(scatter, ax=axes[0, 2], label='Anomaly Score')

scatter = axes[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=forest_y_pred_test, cmap='viridis')
axes[1, 0].set_title("IForest Test Data")
fig.colorbar(scatter, ax=axes[1, 0], label='Anomaly Score')

scatter = axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=dif_y_pred_test, cmap='viridis')
axes[1, 1].set_title("DIF Test Data")
fig.colorbar(scatter, ax=axes[1, 1], label='Anomaly Score')

scatter = axes[1, 2].scatter(X_test[:, 0], X_test[:, 1], c=loda_y_pred_test, cmap='viridis')
axes[1, 2].set_title("LODA Test Data")
fig.colorbar(scatter, ax=axes[1, 2], label='Anomaly Score')

plt.show()

# 3D

X_train, y_train = make_blobs(n_samples=500, centers=[[0, 10, 0], [10, 0, 10]], n_features=3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2])
ax.set_title("3D Training Data")
plt.show()

forest_model = IForest(contamination=0.02)
dif_model = DIF(contamination=0.02)
loda_model = LODA(contamination=0.02)

forest_model.fit(X_train)
dif_model.fit(X_train)
loda_model.fit(X_train)

X_test = np.random.uniform(low=[-10, -10, -10], high=[20, 20, 20], size=(1000, 3))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2])
ax.set_title("3D Test Data")
plt.show()

forest_y_pred_train = forest_model.decision_function(X_train)
forest_y_pred_test = forest_model.decision_function(X_test)

dif_y_pred_train = dif_model.decision_function(X_train)
dif_y_pred_test = dif_model.decision_function(X_test)

loda_y_pred_train = loda_model.decision_function(X_train)
loda_y_pred_test = loda_model.decision_function(X_test)

fig = plt.figure(figsize=(18, 12))

ax1 = fig.add_subplot(231, projection='3d')
scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=forest_y_pred_train)
ax1.set_title("IForest Training Data")
fig.colorbar(scatter, ax=ax1, label='Anomaly Score')

ax2 = fig.add_subplot(232, projection='3d')
scatter = ax2.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=dif_y_pred_train)
ax2.set_title("DIF Training Data")
fig.colorbar(scatter, ax=ax2, label='Anomaly Score')

ax3 = fig.add_subplot(233, projection='3d')
scatter = ax3.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=loda_y_pred_train)
ax3.set_title("LODA Training Data")
fig.colorbar(scatter, ax=ax3, label='Anomaly Score')

ax4 = fig.add_subplot(234, projection='3d')
scatter = ax4.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=forest_y_pred_test)
ax4.set_title("IForest Test Data")
fig.colorbar(scatter, ax=ax4, label='Anomaly Score')

ax5 = fig.add_subplot(235, projection='3d')
scatter = ax5.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=dif_y_pred_test)
ax5.set_title("DIF Test Data")
fig.colorbar(scatter, ax=ax5, label='Anomaly Score')

ax6 = fig.add_subplot(236, projection='3d')
scatter = ax6.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=loda_y_pred_test)
ax6.set_title("LODA Test Data")
fig.colorbar(scatter, ax=ax6, label='Anomaly Score')

plt.suptitle("3D Anomaly Detection Results", fontsize=16)
plt.show()
