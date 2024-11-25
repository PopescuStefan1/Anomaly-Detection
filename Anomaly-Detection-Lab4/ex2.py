import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

dataset = scipy.io.loadmat('datasets/cardio.mat')

X = dataset['X']
y = dataset['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# label_pyod = (-1 * label_sklearn + 1) / 2
# label_sklearn = -1 * (2 * label_pyod - 1)
y_train = -1 * (2 * y_train - 1)
y_test = -1 * (2 * y_test - 1)

# y_train = np.where(y_train == 0, 1, -1)
# y_test = np.where(y_test == 0, 1, -1)

param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],       
    'nu': [0.01, 0.05, 0.1, 0.15, 0.2]                    
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),        
    ('ocsvm', OneClassSVM())             
])

param_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  
    'ocsvm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],       
    'ocsvm__nu': [0.01, 0.05, 0.1, 0.2, 0.4]                    
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='balanced_accuracy',          
    verbose=1          
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score (Balanced Acc):", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)  

y_scores_test = best_model.decision_function(X_test)  
balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

print(f"Balanced Accuracy (Test): {balanced_acc:.4f}")