# src/train.py
# dependencies for paths, models, math, data handling, and plotting.
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
#file and folder paths
DATA_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

#creation of output folders
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

#Splitting the data into features and target label.
X = df.drop(columns=["Class"])
y = df["Class"]

#Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25, #Use 25% of the data for testing.
    random_state=42, #Keep the split reproducible.
    stratify=y, #Preserve the same class balance in train and test sets.
)

# Keep only normal transactions for training unsupervised anomaly models.
normal_train = X_train[y_train == 0]

# Evaluate a model using anomaly scores, save reports, and return summary metrics.
def evaluate_model(name, y_true, anomaly_score):
    roc = roc_auc_score(y_true, anomaly_score)
    pr_auc = average_precision_score(y_true, anomaly_score)

    # Find the threshold that gives the best F1 score.
    precision, recall, thresholds = precision_recall_curve(y_true, anomaly_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[max(best_idx - 1, 0)]
    
    # Convert anomaly scores into predicted labels using the best threshold.
    y_pred = (anomaly_score >= best_threshold).astype(int)
    # Generate classification metrics and confusion matrix.
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)
    # evaluation results.
    print(f"\n{name}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(report)
    print(cm)
    #Save the text report.
    with open(REPORT_DIR / f"{name}_report.txt", "w") as f:
        f.write(f"{name}\n")
        f.write(f"ROC-AUC: {roc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    #plotting the precision-recall curve.
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} Precision-Recall Curve")
    plt.savefig(REPORT_DIR / f"{name}_pr_curve.png", bbox_inches="tight")
    plt.close()
    #Return model results
    return {
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "best_threshold": best_threshold,
    }
# Store results from each model
results = []
# Building an Isolation Forest pipeline
iso = Pipeline([
    ("scaler", StandardScaler()),
    ("model", IsolationForest(
        n_estimators=300, # Number of trees in the forest
        contamination=0.00172, # Expected fraud/anomaly rate
        random_state=42, # Keep model training reproducible
        n_jobs=-1,  # Use all available CPU cores
    )),
])

# Train, score, evaluate, and save the Isolation Forest model
iso.fit(normal_train)
iso_scores = -iso.decision_function(X_test)
results.append(evaluate_model("isolation_forest", y_test, iso_scores))
joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib") #exporting the model to a file



# Build a One-Class SVM pipeline.
ocsvm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", OneClassSVM(
        kernel="rbf", # Use a nonlinear radial basis function kernel
        gamma="scale", # Automatically scale kernel sensitivity within dataset
        nu=0.00172, # Expected upper bound on anomaly rate
    )),
])

'''
nu=0.00172 in OneClassSVM was chosen to match the estimated fraud rate in the credit card dataset.

For the commonly used credit card fraud dataset, there are:

* 492 fraud transactions
* 284,807 total transactions

Fraud rate:

492 / 284807 ≈ 0.001727

which rounds to 0.00172 (0.172%).

In OneClassSVM, nu has a specific meaning + constraints:

- It's an upper bound on the fraction of training errors (outliers).
- It's a lower bound on the fraction of support vectors.
- It's must be between 0 and 1.
'''
# Train on a sample for speed, then score, evaluate, and save the One-Class SVM model.
sample_normal_train = normal_train.sample(n=25000, random_state=42)
ocsvm.fit(sample_normal_train)
ocsvm_scores = -ocsvm.decision_function(X_test)
results.append(evaluate_model("one_class_svm", y_test, ocsvm_scores))
joblib.dump(ocsvm, MODEL_DIR / "one_class_svm.joblib")

# Local Outlier Factor pipeline.
lof = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LocalOutlierFactor(
        n_neighbors=35, # Number of nearby points used to estimate local density
        contamination=0.00172, # Expected fraud/anomaly rate
        novelty=True, # Allow scoring new unseen data
        n_jobs=-1, # Use all available CPU cores
    )),
])

# Training, scoring, evaluating, and saving the Local Outlier Factor model
lof.fit(normal_train)
lof_scores = -lof.decision_function(X_test)
results.append(evaluate_model("local_outlier_factor", y_test, lof_scores))
joblib.dump(lof, MODEL_DIR / "local_outlier_factor.joblib")

#Supervised Random Forest baseline pipeline.
rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300, # Number of trees in the forest
        class_weight="balanced_subsample", # Rebalance classes inside each tree sample
        random_state=42, # Keep model training reproducible
        n_jobs=-1, # Use all available CPU cores
    )),
])

# Train, score, evaluate, and save the Random Forest baseline model.
rf.fit(X_train, y_train)
rf_scores = rf.predict_proba(X_test)[:, 1]
results.append(evaluate_model("random_forest_supervised_baseline", y_test, rf_scores))
joblib.dump(rf, MODEL_DIR / "random_forest_baseline.joblib")

# Save all model results to CSV file
pd.DataFrame(results).to_csv(REPORT_DIR / "model_results.csv", index=False)
print("\nSaved reports to reports/")