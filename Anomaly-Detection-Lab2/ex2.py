from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, 
                                                          n_test=200,
                                                          n_features=2, 
                                                          n_clusters=2,
                                                          contamination=0.1)

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.title('Train Data with Contamination')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.title('Test Data with Contamination')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

n_neighbors_values = [1, 5, 10, 15]

for i, n_neighbors in enumerate(n_neighbors_values):
    knn_model = KNN(n_neighbors=n_neighbors)
    knn_model.fit(X_train)

    y_train_pred = knn_model.labels_
    y_test_pred = knn_model.predict(X_test)

    plt.subplot(4, 4, i * 4 + 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.title('Ground Truth Labels for Training Data')

    plt.subplot(4, 4, i * 4+2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
    plt.title('Predicted Labels for Training Data')

    plt.subplot(4, 4, i * 4+3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.title('Ground Truth Labels for Test Data')

    plt.subplot(4, 4, i * 4+4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred)
    plt.title('Predicted Labels for Test Data')

plt.tight_layout()
plt.show()
