from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
  response = client.get("/health")
  assert response.status_code == 200
  assert response.json()["status"] == "healthy"


def test_predict_valid_input():
  payload = {
      "age": 52,
      "sex": 1,
      "cp": 0,
      "trestbps": 125,
      "chol": 212,
      "fbs": 0,
      "restecg": 1,
      "thalach": 168,
      "exang": 0,
      "oldpeak": 1.0,
      "slope": 2,
      "ca": 2,
      "thal": 3,
  }
  response = client.post("/predict", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert "prediction" in data
  assert "confidence" in data


def test_predict_invalid_input():
  payload = {"age": "invalid_age"}
  response = client.post("/predict", json=payload)
  assert response.status_code == 422