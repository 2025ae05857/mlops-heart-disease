import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# Ensure output directories exist
os.makedirs("artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Set MLflow Experiment
mlflow.set_experiment("Heart_Disease_Classification")

# 1. Load Data
csv_path = "data/heart.csv"
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv(csv_path, names=columns, na_values="?")
df = df.dropna().reset_index(drop=True)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- RUN 1: Logistic Regression ---
with mlflow.start_run(run_name="Logistic_Regression_Production"):
    lr_params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
    mlflow.log_params(lr_params)
    
    model_lr = LogisticRegression(**lr_params)
    model_lr.fit(X_train_scaled, y_train)
    
    y_pred = model_lr.predict(X_test_scaled)
    y_prob = model_lr.predict_proba(X_test_scaled)[:, 1]
    
    metrics_lr = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    mlflow.log_metrics(metrics_lr)
    
    # Save & Log Plot Artifact
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Healthy", "Disease"], yticklabels=["Healthy", "Disease"])
    ax.set_title("Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = "artifacts/lr_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    mlflow.log_artifact(cm_path)
    mlflow.sklearn.log_model(model_lr, "model")
    print(f"[MLflow] Logistic Regression Run Completed | ROC-AUC: {metrics_lr['roc_auc']:.4f}")

# --- RUN 2: Random Forest ---
with mlflow.start_run(run_name="Random_Forest_Baseline"):
    rf_params = {"n_estimators": 100, "max_depth": 5, "min_samples_split": 5, "random_state": 42}
    mlflow.log_params(rf_params)
    
    model_rf = RandomForestClassifier(**rf_params)
    model_rf.fit(X_train, y_train)
    
    y_pred_rf = model_rf.predict(X_test)
    y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
    
    metrics_rf = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1_score": f1_score(y_test, y_pred_rf),
        "roc_auc": roc_auc_score(y_test, y_prob_rf)
    }
    mlflow.log_metrics(metrics_rf)
    
    mlflow.sklearn.log_model(model_rf, "model")
    print(f"[MLflow] Random Forest Run Completed | ROC-AUC: {metrics_rf['roc_auc']:.4f}")

print("\nSUCCESS: All experiment runs logged to MLflow under 'mlruns/'.")
