import os
import pickle
import numpy as np
import pandas as pd
import pytest

# 1. Test Dataset Availability & Structure
def test_dataset_integrity():
    csv_path = "data/heart.csv"
    assert os.path.exists(csv_path), "Dataset heart.csv is missing!"
    
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(csv_path, names=columns, na_values="?")
    assert not df.empty, "Dataset is empty!"
    assert len(df.columns) == 14, f"Expected 14 columns, got {len(df.columns)}"

# 2. Test Model Artifacts
def test_model_artifacts_exist():
    assert os.path.exists("models/best_model.pkl"), "Production model file best_model.pkl does not exist!"
    assert os.path.exists("models/scaler.pkl"), "Scaler artifact scaler.pkl does not exist!"

# 3. Test Preprocessing Transformer & Model Predictions
def test_model_inference():
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # Sample 14-feature patient input record
    sample_input = np.array([[63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0]])
    
    scaled_input = scaler.transform(sample_input)
    prediction = model.predict(scaled_input)
    probabilities = model.predict_proba(scaled_input)
    
    assert prediction[0] in [0, 1], f"Invalid prediction value: {prediction[0]}"
    assert probabilities.shape == (1, 2), "Prediction probability shape mismatch!"
    assert round(np.sum(probabilities), 4) == 1.0, "Probabilities do not sum to 1.0!"

print("Unit test suite created successfully in tests/test_pipeline.py")
