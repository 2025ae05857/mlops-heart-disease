import os
import pytest
import pandas as pd
import pickle
import numpy as np

def test_data_cleaning():
    output_path = "data/heart_cleaned.csv"
    assert os.path.exists(output_path), "Cleaned CSV file was not created."
    df = pd.read_csv(output_path)
    assert df.isnull().sum().sum() == 0, "Cleaned dataset contains missing values."
    assert set(df['target'].unique()).issubset({0, 1}), "Target variable contains invalid values."

def test_model_inference():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    assert os.path.exists(model_path), "Best model file missing."
    assert os.path.exists(scaler_path), "Scaler file missing."
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    dummy_input = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    pred = model.predict(dummy_input)
    assert len(pred) == 1
    assert pred[0] in [0, 1]
