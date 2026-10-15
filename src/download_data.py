import os
import urllib.request
import zipfile

DATA_DIR = "data"
DATA_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ZIP_PATH = os.path.join(DATA_DIR, "heart_disease.zip")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading Heart Disease dataset from UCI Repository...")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    
    print("Extracting dataset files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    print(f"Dataset successfully downloaded and extracted into '{DATA_DIR}/'")

if __name__ == "__main__":
    download_dataset()
