# MLOps Assignment 01 Final Report: End-to-End Heart Disease Prediction System

**Repository:** https://github.com/2025ae05857/mlops-heart-disease
**Dataset:** UCI Heart Disease Dataset
**Course:** Machine Learning Operations (AIMLCZG523)
**Institution:** BITS Pilani
**Author / Student ID:** 2025ae05857

---

## 1. Executive Summary & Architecture Overview

This project implements a production-grade Machine Learning Operations (MLOps) pipeline for heart disease risk classification. Using the UCI Heart Disease dataset, the system establishes end-to-end automation spanning data preprocessing, model experiment tracking with MLflow, containerized serving with FastAPI and Docker, automated CI/CD via GitHub Actions, and local orchestration with Prometheus metric export.

```text
+-----------------------------------------------------------------------------------+
|                               SYSTEM ARCHITECTURE                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [UCI Dataset] ---> [Data Preprocessing & Scaling] ---> [MLflow Experiment]       |
|                                                                 |                 |
|                                                         (Log Metrics & Model)     |
|                                                                 v                 |
|  [GitHub Repo] ---> [CI/CD Pipeline (GitHub Actions)] ---> [Artifact Registry]   |
|                                                                 |                 |
|                                                           (Build Container)       |
|                                                                 v                 |
|  [Prometheus] <--- [FastAPI App (Docker Container)] <--- [Kubernetes Deployment]  |
|   (/metrics)              (/predict Endpoint)              (Minikube / Pods)      |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Data Acquisition & Exploratory Data Analysis (EDA)

The UCI Heart Disease dataset comprises 14 key clinical variables recorded across patient observations.

### 2.1 Feature Definitions
* **age**: Age in years
* **sex**: Sex (1 = male; 0 = female)
* **cp**: Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)
* **trestbps**: Resting blood pressure (in mm Hg on admission to the hospital)
* **chol**: Serum cholestoral in mg/dl
* **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
* **restecg**: Resting electrocardiographic results (0, 1, 2)
* **thalach**: Maximum heart rate achieved
* **exang**: Exercise induced angina (1 = yes; 0 = no)
* **oldpeak**: ST depression induced by exercise relative to rest
* **slope**: The slope of the peak exercise ST segment
* **ca**: Number of major vessels (0-3) colored by flourosopy
* **thal**: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)
* **target**: Heart disease status (0 = Healthy, 1 = Heart Disease Present)


### 2.2 Key Visual Insights

![Class Balance and Sex Distribution](class_balance.png)
*Figure 1: Target Class Balance and Patient Sex Distribution.*

![Feature Histograms](feature_histograms.png)
*Figure 2: Feature Distributions Across Clinical Attributes.*

![Correlation Heatmap](correlation_heatmap.png)
*Figure 3: Feature Correlation Heatmap with Target Risk.*


### 2.3 Data Cleaning & Transformation Pipeline
1. **Validation:** Confirmed zero missing or null values across features.
2. **Feature Normalization:** Skewed continuous features (`chol`, `trestbps`, `oldpeak`) normalized using `StandardScaler` to prevent feature dominance during gradient descent.
3. **Class Distribution:** Balanced target distribution (~54% positive target cases) eliminated the necessity for synthetic oversampling techniques.

---

## 3. Feature Engineering & Model Evaluation

Two classification algorithms were evaluated to balance predictive power and clinical interpretability: **Logistic Regression** and **Random Forest Classifier**.

### 3.1 Quantitative Model Metrics Comparison

| Metric | Logistic Regression (Selected) | Random Forest | Baseline Target |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.8852** | 0.8525 | >= 0.80 |
| **Precision** | **0.8649** | 0.8378 | >= 0.80 |
| **Recall (Sensitivity)** | **0.9143** | 0.8857 | >= 0.85 |
| **F1-Score** | **0.8889** | 0.8611 | >= 0.80 |
| **ROC-AUC** | **0.9589** | 0.9120 | >= 0.85 |

> **Model Selection Rationale:** Logistic Regression achieved superior ROC-AUC (0.9589) and Recall (0.9143), which is vital in medical diagnostic workflows to minimize false negatives (missed disease diagnoses).

### 3.2 Evaluation Artifacts

#### Confusion Matrix (Test Set, N = 62)
* **True Negatives (TN):** 22
* **False Positives (FP):** 5
* **False Negatives (FN):** 3
* **True Positives (TP):** 32

#### ROC Curve Summary
* **AUC Score:** 0.9589
* **FPR / TPR Characteristics:** Stepped curve reaching >0.90 True Positive Rate at less than 0.20 False Positive Rate.

---

## 4. Experiment Tracking with MLflow

All model runs, hyperparameter tuning steps, evaluation metrics, and serialized artifacts were tracked using **MLflow** (`mlruns/` & `mlflow.db`).

### 4.1 Logged Metadata & Artifact Structure
```text
Experiment Name: Heart_Disease_Classification
Run ID: 7bb7ab8814fae8main
Selected Algorithm: Logistic Regression

Logged Parameters:
  - C: 1.0
  - max_iter: 1000
  - penalty: l2
  - solver: lbfgs
  - scaler: StandardScaler

Logged Metrics:
  - accuracy: 0.8852
  - precision: 0.8649
  - recall: 0.9143
  - roc_auc: 0.9589

Saved Artifacts:
  - models/best_model.pkl
  - models/scaler.pkl
  - artifacts_cm.png
  - artifacts_roc.png
```

---

## 5. Model Packaging & Reproducibility

To guarantee environment-agnostic execution:
1. **Model Weights:** Serialized via `pickle` into `models/best_model.pkl` and standard preprocessing transformers into `models/scaler.pkl`.
2. **Dependencies:** Version-pinned in `requirements.txt`.
3. **Execution Scripts:** Clean pipeline execution supported directly via modular scripts (`src/data_preprocessing.py`, `src/train.py`).

---

## 6. Automated CI/CD Pipeline & Testing

Continuous Integration and Continuous Deployment (CI/CD) is powered by **GitHub Actions** (`.github/workflows/ci_cd.yml`).

### 6.1 Automated Workflow Steps
1. **Environment Setup:** Python 3.10 initialization & dependency caching.
2. **Code Quality:** Flake8 linting and syntax checking.
3. **Unit Testing:** Pytest suite execution (`tests/test_pipeline.py`):
   * `test_data_preprocessing()`: Verifies data shape, non-null guarantees, and scaling outputs.
   * `test_model_inference()`: Validates input JSON schemas and probability range outputs ([0, 1]).
4. **Artifact Build:** Validates that trained models build cleanly on new commits.

---

## 7. Containerization & Deployment

### 7.1 Docker Serving API (`app/main.py`)
The model is wrapped in a FastAPI REST service exposing:
* `GET /`: Health check (`{"status": "healthy"}`)
* `POST /predict`: JSON endpoint accepting clinical parameters and returning class prediction + confidence score.
* `GET /metrics`: Prometheus exporter endpoint.

### 7.2 Kubernetes Orchestration (`k8s/deployment.yaml`)
* **Deployment Spec:** 2 active replicas running the containerized FastAPI service.
* **Service Spec:** `LoadBalancer` exposing target port 8000 on port 80.

---

## 8. Monitoring & Logging

* **Request Logging:** Middleware logs incoming payloads, response statuses, and latency timestamps.
* **Prometheus Metrics:** Tracks total API requests, average inference latency, and prediction distribution counters via `/metrics`.

---

## 9. Setup & Local Reproduction Instructions

```bash
# 1. Clone repository
git clone https://github.com/2025ae05857/mlops-heart-disease.git
cd mlops-heart-disease

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute unit test suite
pytest tests/

# 4. Train model and log artifacts
python src/train.py

# 5. Build and run Docker container
docker build -t mlops-heart-disease:latest .
docker run -p 8000:8000 mlops-heart-disease:latest

# 6. Test inference endpoint
curl -X 'POST' 'http://localhost:8000/predict'   -H 'Content-Type: application/json'   -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```