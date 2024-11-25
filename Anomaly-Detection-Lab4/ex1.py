from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=0.15)

# OCSVM
# ocsvm = OCSVM(kernel='linear')
ocsvm = OCSVM(kernel='rbf')

ocsvm.fit(X_train)

y_scores = ocsvm.decision_function(X_test)
y_pred = ocsvm.predict(X_test)  
y_train_pred = ocsvm.predict(X_train)

y_pred_binary = (y_pred == 1).astype(int)
y_train_pred_binary = (y_train_pred == 1).astype(int)

balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_scores)

print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='b',)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='r',)
ax1.set_title('Ground Truth (Training Data)')

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='b')
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='r')
ax2.set_title('Ground Truth (Test Data)')

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_train[y_train_pred_binary == 0, 0], X_train[y_train_pred_binary == 0, 1], X_train[y_train_pred_binary == 0, 2], c='b')
ax3.scatter(X_train[y_train_pred_binary == 1, 0], X_train[y_train_pred_binary == 1, 1], X_train[y_train_pred_binary == 1, 2], c='r')
ax3.set_title('Predicted Labels (Training Data)')

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[y_pred_binary == 0, 0], X_test[y_pred_binary == 0, 1], X_test[y_pred_binary == 0, 2], c='b')
ax4.scatter(X_test[y_pred_binary == 1, 0], X_test[y_pred_binary == 1, 1], X_test[y_pred_binary == 1, 2], c='r')
ax4.set_title('Predicted Labels (Test Data)')

plt.tight_layout()
plt.show()

# DeepSVDD
deepSVDD = DeepSVDD(n_features=3)

deepSVDD.fit(X_train)

deepSVDD_y_scores = deepSVDD.decision_function(X_test)
deepSVDD_y_pred = deepSVDD.predict(X_test)  
deepSVDD_y_train_pred = deepSVDD.predict(X_train)

deepSVDD_y_pred_binary = (deepSVDD_y_pred == 1).astype(int)
deepSVDD_y_train_pred_binary = (deepSVDD_y_train_pred == 1).astype(int)

deepSVDD_balanced_acc = balanced_accuracy_score(y_test, deepSVDD_y_pred_binary)
deepSVDD_roc_auc = roc_auc_score(y_test, deepSVDD_y_scores)

print(f"Balanced Accuracy: {deepSVDD_balanced_acc:.4f}")
print(f"ROC AUC: {deepSVDD_roc_auc:.4f}")

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='b',)
ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='r',)
ax1.set_title('Ground Truth (Training Data)')

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='b')
ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='r')
ax2.set_title('Ground Truth (Test Data)')

ax3 = fig.add_subplot(223, projection='3d')
ax3.scatter(X_train[deepSVDD_y_train_pred_binary == 0, 0], X_train[deepSVDD_y_train_pred_binary == 0, 1], X_train[deepSVDD_y_train_pred_binary == 0, 2], c='b')
ax3.scatter(X_train[deepSVDD_y_train_pred_binary == 1, 0], X_train[deepSVDD_y_train_pred_binary == 1, 1], X_train[deepSVDD_y_train_pred_binary == 1, 2], c='r')
ax3.set_title('Predicted Labels (Training Data)')

ax4 = fig.add_subplot(224, projection='3d')
ax4.scatter(X_test[deepSVDD_y_pred_binary == 0, 0], X_test[deepSVDD_y_pred_binary == 0, 1], X_test[deepSVDD_y_pred_binary == 0, 2], c='b')
ax4.scatter(X_test[deepSVDD_y_pred_binary == 1, 0], X_test[deepSVDD_y_pred_binary == 1, 1], X_test[deepSVDD_y_pred_binary == 1, 2], c='r')
ax4.set_title('Predicted Labels (Test Data)')

plt.tight_layout()
plt.show()
