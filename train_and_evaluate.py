import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Set MLflow experiment name
mlflow.set_experiment("Heart_Disease_Classification")

# 1. Load Dataset
data_path = "heart.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    # Fallback dataset if heart.csv is missing in local workspace
    X_dummy, y_dummy = make_classification(
        n_samples=300, n_features=13, random_state=42
    )
    df = pd.DataFrame(X_dummy, columns=[f"feature_{i}" for i in range(13)])
    df["target"] = y_dummy

X = df.drop(columns=["target"])
y = df["target"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to numpy to avoid cross-validation feature name mismatch
X_train_np, y_train_np = np.asarray(X_train), np.asarray(y_train)
X_test_np, y_test_np = np.asarray(X_test), np.asarray(y_test)

# ==========================================
# RUN 1: Logistic Regression
# ==========================================
with mlflow.start_run(run_name="Logistic_Regression"):
    lr_params = {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]}
    grid_lr = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_params,
        cv=5,
        scoring="accuracy",
    )
    grid_lr.fit(X_train_scaled, y_train_np)
    best_lr = grid_lr.best_estimator_

    # Predict
    y_pred_lr = best_lr.predict(X_test_scaled)
    y_prob_lr = best_lr.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc_lr = accuracy_score(y_test_np, y_pred_lr)
    prec_lr = precision_score(y_test_np, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test_np, y_pred_lr, zero_division=0)
    f1_lr = f1_score(y_test_np, y_pred_lr, zero_division=0)
    auc_lr = roc_auc_score(y_test_np, y_prob_lr)

    # Log Parameters & Metrics to MLflow
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.log_metric("precision", prec_lr)
    mlflow.log_metric("recall", rec_lr)
    mlflow.log_metric("f1", f1_lr)
    mlflow.log_metric("roc_auc", auc_lr)

    # Log Model Artifact
    mlflow.sklearn.log_model(best_lr, "model")
    print(f"[✓] Logistic Regression Completed successfully!")
    print(f"    Accuracy: {acc_lr:.4f} | F1: {f1_lr:.4f} | ROC-AUC: {auc_lr:.4f}")

# ==========================================
# RUN 2: Random Forest
# ==========================================
with mlflow.start_run(run_name="Random_Forest"):
    rf_params = {"n_estimators": [50, 100], "max_depth": [3, 5, 10]}
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=5,
        scoring="accuracy",
    )
    grid_rf.fit(X_train_np, y_train_np)
    best_rf = grid_rf.best_estimator_

    # Predict
    y_pred_rf = best_rf.predict(X_test_np)
    y_prob_rf = best_rf.predict_proba(X_test_np)[:, 1]

    # Metrics
    acc_rf = accuracy_score(y_test_np, y_pred_rf)
    prec_rf = precision_score(y_test_np, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test_np, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test_np, y_pred_rf, zero_division=0)
    auc_rf = roc_auc_score(y_test_np, y_prob_rf)

    # Log Parameters & Metrics to MLflow
    mlflow.log_params(grid_rf.best_params_)
    mlflow.log_metric("accuracy", acc_rf)
    mlflow.log_metric("precision", prec_rf)
    mlflow.log_metric("recall", rec_rf)
    mlflow.log_metric("f1", f1_rf)
    mlflow.log_metric("roc_auc", auc_rf)

    # Log Model Artifact
    mlflow.sklearn.log_model(best_rf, "model")
    print(f"[✓] Random Forest Completed successfully!")
    print(f"    Accuracy: {acc_rf:.4f} | F1: {f1_rf:.4f} | ROC-AUC: {auc_rf:.4f}")

print("\n==========================================")
print(" Execution complete! Both models logged to MLflow.")
print("==========================================")