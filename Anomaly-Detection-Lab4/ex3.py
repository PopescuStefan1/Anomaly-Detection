import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

dataset = scipy.io.loadmat('datasets/shuttle.mat')

X = dataset['X']
y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# OCSVM
ocsvm = OCSVM()

ocsvm.fit(X_train_scaled)

y_scores_ocsvm = ocsvm.decision_function(X_test_scaled)
y_pred_ocsvm = ocsvm.predict(X_test_scaled) 
y_pred_ocsvm_binary = (y_pred_ocsvm == 1).astype(int)

balanced_acc_ocsvm = balanced_accuracy_score(y_test, y_pred_ocsvm_binary)
roc_auc_ocsvm = roc_auc_score(y_test, y_scores_ocsvm)

print("OCSVM Model:")
print(f"Balanced Accuracy: {balanced_acc_ocsvm:.4f}")
print(f"ROC AUC: {roc_auc_ocsvm:.4f}")

print('-' * 30)
# DeepSVDD
deepsvdd = DeepSVDD(n_features=9)

deepsvdd.fit(X_train_scaled)

y_scores_deepsvdd = deepsvdd.decision_function(X_test_scaled)
y_pred_deepsvdd = deepsvdd.predict(X_test_scaled) 
y_pred_deepsvdd_binary = (y_pred_deepsvdd == 1).astype(int)

balanced_acc_deepsvdd = balanced_accuracy_score(y_test, y_pred_deepsvdd_binary)
roc_auc_deepsvdd = roc_auc_score(y_test, y_scores_deepsvdd)

print("DeepSVDD Model:")
print(f"Balanced Accuracy: {balanced_acc_deepsvdd:.4f}")
print(f"ROC AUC: {roc_auc_deepsvdd:.4f}")
