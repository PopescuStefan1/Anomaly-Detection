import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

X, y = make_blobs(n_samples=[200, 100], 
                   centers=[[-10, -10], [10, 10]], 
                   cluster_std=[2, 6])

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

knn_model = KNN(contamination=0.07)
lof_model = LOF(contamination=0.07)

knn_model.fit(X)
lof_model.fit(X)

y_train_knn = knn_model.labels_
y_train_lof = lof_model.labels_

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_train_knn)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_train_lof)

plt.show()
