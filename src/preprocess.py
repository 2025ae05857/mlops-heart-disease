import pandas as pd
import numpy as np
import os

COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

def load_and_clean_data(raw_path="data/processed.cleveland.data", output_path="data/heart_cleaned.csv"):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_path}. Run download_data.py first.")

    # Load raw Cleveland dataset
    df = pd.read_csv(raw_path, header=None, names=COLUMNS, na_values="?")
    
    # Handle missing values by replacing NaNs with column medians
    for col in ['ca', 'thal']:
        df[col] = df[col].fillna(df[col].median())

    # Convert binary target: 0 = No Disease, 1 = Heart Disease present (values 1,2,3,4)
    df['target'] = (df['target'] > 0).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to '{output_path}' (Shape: {df.shape})")
    return df

if __name__ == "__main__":
    load_and_clean_data()
