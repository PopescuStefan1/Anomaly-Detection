import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
  
# Am luat datele de pe acest site folosind biblioteca "ucimlrepo"
# https://www.archive.ics.uci.edu/dataset/148/statlog+shuttle
# Deoarece site-ul din laborator nu functiona
statlog_shuttle = fetch_ucirepo(id=148) 
  
X = statlog_shuttle.data.features 
y = statlog_shuttle.data.targets 

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

n_splits = 1
results = {
    "IForest": {"BA": [], "ROC_AUC": []},
    "DIF": {"BA": [], "ROC_AUC": []},
    "LODA": {"BA": [], "ROC_AUC": []}
}

for _ in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4)

    forest_model = IForest(contamination=0.2)
    dif_model = DIF(contamination=0.2)
    loda_model = LODA(contamination=0.2)

    forest_model.fit(X_train)
    dif_model.fit(X_train)
    loda_model.fit(X_train)

    forest_y_pred = forest_model.predict(X_test)
    dif_y_pred = dif_model.predict(X_test)
    loda_y_pred = loda_model.predict(X_test)

    forest_decision_scores = forest_model.decision_function(X_test)
    dif_decision_scores = dif_model.decision_function(X_test)
    loda_decision_scores = loda_model.decision_function(X_test)

    forest_decision_scores = forest_decision_scores.reshape(-1, 1)
    dif_decision_scores = dif_decision_scores.reshape(-1, 1)
    loda_decision_scores = loda_decision_scores.reshape(-1, 1)

    forest_ba = balanced_accuracy_score(y_test, forest_y_pred)
    dif_ba = balanced_accuracy_score(y_test, dif_y_pred)
    loda_ba = balanced_accuracy_score(y_test, loda_y_pred)

    lb = LabelBinarizer()
    y_test_one_hot = lb.fit_transform(y_test)

    forest_auc = roc_auc_score(y_test_one_hot, forest_decision_scores, multi_class='ovr')
    dif_auc = roc_auc_score(y_test_one_hot, dif_decision_scores, multi_class='ovr')
    loda_auc = roc_auc_score(y_test_one_hot, loda_decision_scores, multi_class='ovr')
    
    results["IForest"]["BA"].append(forest_ba)
    results["IForest"]["ROC_AUC"].append(forest_auc)

    results["DIF"]["BA"].append(dif_ba)
    results["DIF"]["ROC_AUC"].append(dif_auc)

    results["LODA"]["BA"].append(loda_ba)
    results["LODA"]["ROC_AUC"].append(loda_auc)

for model, metrics in results.items():
    print(f"\n{model} Results:")
    mean_ba = np.mean(metrics["BA"])
    mean_auc = np.mean(metrics["ROC_AUC"])
    std_ba = np.std(metrics["BA"])
    std_auc = np.std(metrics["ROC_AUC"])
    print(f"Mean Balanced Accuracy (BA): {mean_ba:.4f} ± {std_ba:.4f}")
    print(f"Mean ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")
