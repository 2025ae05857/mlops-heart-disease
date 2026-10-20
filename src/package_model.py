import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Ensure output directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# 2. Load & Prepare Data
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

# 3. Fit Preprocessing Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train Selected Production Model
model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Serialize Artifacts to models/
scaler_path = "models/scaler.pkl"
model_path = "models/best_model.pkl"

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"SUCCESS: Saved scaler to '{scaler_path}'")
print(f"SUCCESS: Saved production model to '{model_path}'")
