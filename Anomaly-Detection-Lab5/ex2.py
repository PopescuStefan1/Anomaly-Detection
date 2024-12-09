import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

dataset = scipy.io.loadmat('datasets/shuttle.mat')

X = dataset['X']
y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_test_standardized = scaler.fit_transform(X_test)

model = PCA(contamination=0.2)
model.fit(X_standardized)

individual_variances = model.explained_variance_ratio_  
cumulative_explained_variance = np.cumsum(individual_variances)

plt.figure()
plt.bar(range(1, len(individual_variances) + 1), individual_variances, color='blue')
plt.step(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, color='orange', where='mid')
plt.show()

y_train_binary = (y_train > 1).astype(int)
y_test_binary = (y_test > 1).astype(int)

print("heer")
kpca_model = KPCA(contamination=0.2)  
kpca_model.fit(X_standardized)

y_train_pred_kpca = kpca_model.labels_  
y_test_pred_kpca = kpca_model.predict(X_test_standardized)  

print(y_train_pred_kpca)
print(y_test_pred_kpca)

train_balanced_accuracy_kpca = balanced_accuracy_score(y_train_binary, y_train_pred_kpca)
test_balanced_accuracy_kpca = balanced_accuracy_score(y_test_binary, y_test_pred_kpca)

print(f"KPCA Train Balanced Accuracy: {train_balanced_accuracy_kpca}")
print(f"KPCA Test Balanced Accuracy: {test_balanced_accuracy_kpca}")
