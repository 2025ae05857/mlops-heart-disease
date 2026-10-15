import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def train_and_evaluate():
    mlflow.set_experiment("Heart_Disease_Classification")

    data_path = "data/heart_cleaned.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run src/preprocess.py first.")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    models = {
        "Logistic_Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }

    best_model = None
    best_auc = 0.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            if name == "Logistic_Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc:.4f}")

            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            
            # Use cloudpickle format to bypass scikit-learn 1.6+ pickle trust checks
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}", serialization_format="cloudpickle")

            if auc > best_auc:
                best_auc = auc
                best_model = model
                with open("models/best_model.pkl", "wb") as f:
                    pickle.dump(best_model, f)

    print(f"Training Complete. Best Model saved to 'models/best_model.pkl' (AUC: {best_auc:.4f})")

if __name__ == "__main__":
    train_and_evaluate()
