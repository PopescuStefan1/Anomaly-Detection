import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

data = sio.loadmat("./Anomaly-Detection-Lab2/datasets/cardio.mat")

X = data['X']
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

min_max_scaler = MinMaxScaler()
X_train_normalized = min_max_scaler.fit_transform(X_train)
X_test_normalized = min_max_scaler.transform(X_test)

knn_train_scores = []
knn_test_scores = []
lof_train_scores = []
lof_test_scores = []

for n_neighbors in range(30, 121, 10):  
    knn = KNN(n_neighbors=n_neighbors)
    knn.fit(X_train_normalized, y_train)
    
    y_train_pred_knn = knn.predict(X_train_normalized)
    y_test_pred_knn = knn.predict(X_test_normalized)
    
    ba_train_knn = balanced_accuracy_score(y_train, y_train_pred_knn)
    ba_test_knn = balanced_accuracy_score(y_test, y_test_pred_knn)
    
    knn_train_scores.append(ba_train_knn)
    knn_test_scores.append(ba_test_knn)
    
    lof = LOF(n_neighbors=n_neighbors)
    lof.fit(X_train_normalized)
    
    y_train_pred_lof = lof.labels_  
    y_test_pred_lof = lof.predict(X_test_normalized)  
    
    ba_train_lof = balanced_accuracy_score(y_train, y_train_pred_lof)
    ba_test_lof = balanced_accuracy_score(y_test, y_test_pred_lof)
    
    lof_train_scores.append(ba_train_lof)
    lof_test_scores.append(ba_test_lof)
    
    print(f'n_neighbors: {n_neighbors}')
    print(f'KNN - Train BA: {ba_train_knn:.4f}, Test BA: {ba_test_knn:.4f}')
    print(f'LOF - Train BA: {ba_train_lof:.4f}, Test BA: {ba_test_lof:.4f}')
    print('-' * 40)

knn_train_scores_floats = [float(score) for score in knn_train_scores]
knn_test_scores_floats = [float(score) for score in knn_test_scores]
lof_train_scores_floats = [float(score) for score in lof_train_scores]
lof_test_scores_floats = [float(score) for score in lof_test_scores]

print("\nFinal Scores:")
print(f"KNN Train Scores: {knn_train_scores_floats}\n")
print(f"KNN Test Scores: {knn_test_scores_floats}\n")
print(f"LOF Train Scores: {lof_train_scores_floats}\n")
print(f"LOF Test Scores: {lof_test_scores_floats}\n")

train_scores_knn_normalized = standardizer(np.array(knn_train_scores).reshape(-1, 1))
test_scores_knn_normalized = standardizer(np.array(knn_test_scores).reshape(-1, 1))
train_scores_lof_normalized = standardizer(np.array(lof_train_scores).reshape(-1, 1))
test_scores_lof_normalized = standardizer(np.array(lof_test_scores).reshape(-1, 1))

combined_train_scores_average = average(np.vstack([train_scores_knn_normalized, train_scores_lof_normalized]).T)
combined_test_scores_average = average(np.vstack([test_scores_knn_normalized, test_scores_lof_normalized]).T)

combined_train_scores_maximization = maximization(np.vstack([train_scores_knn_normalized, train_scores_lof_normalized]).T)
combined_test_scores_maximization = maximization(np.vstack([test_scores_knn_normalized, test_scores_lof_normalized]).T)

print("Final Train Scores - Average Combination:", combined_train_scores_average)
print("Final Test Scores - Average Combination:", combined_test_scores_average)

print("Final Train Scores - Maximization Combination:", combined_train_scores_maximization)
print("Final Test Scores - Maximization Combination:", combined_test_scores_maximization)
