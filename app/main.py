import time
import logging
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("mlops_service")

# Initialize FastAPI Application
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="MLOps Production Serving Endpoint for Heart Disease Risk Classification",
    version="1.0.0"
)

# Prometheus Metrics Definitions
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API Request Latency", ["endpoint"])

# Mount Prometheus Metrics Endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load Serialized Models and Scalers
try:
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Successfully loaded production model and preprocessor scaler.")
except Exception as e:
    model, scaler = None, None
    logger.error(f"Failed to load model artifacts: {e}")

# Request Logger Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(f"Method: {request.method} Path: {request.url.path} Status: {response.status_code} Latency: {duration:.4f}s")
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    
    return response

# Input Schema
class PatientData(BaseModel):
    age: float = Field(..., example=63.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=1.0)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=2.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=3.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=6.0)

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is active.", "status": "healthy"}

@app.get("/health")
def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model binaries not loaded properly.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler missing.")
    
    features = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]])
    
    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = float(probabilities[prediction])
    
    return {
        "prediction": prediction,
        "label": "Heart Disease Identified" if prediction == 1 else "Healthy (No Disease)",
        "confidence": round(confidence, 4),
        "probability_disease": round(float(probabilities[1]), 4),
        "probability_healthy": round(float(probabilities[0]), 4)
    }
