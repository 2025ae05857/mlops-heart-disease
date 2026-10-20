with open("report.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update Section 2.2 with EDA plots
eda_section = """### 2.2 Key Visual Insights

![Class Balance and Sex Distribution](class_balance.png)
*Figure 1: Target Class Balance and Patient Sex Distribution.*

![Feature Histograms](feature_histograms.png)
*Figure 2: Feature Distributions Across Clinical Attributes.*

![Correlation Heatmap](correlation_heatmap.png)
*Figure 3: Feature Correlation Heatmap with Target Risk.*
"""

# 2. Update Section 2.4 with Model Evaluation Curves
eval_section = """### 2.4 Model Performance & Evaluation

Both Logistic Regression and Random Forest models were evaluated using 5-Fold Stratified Cross-Validation to assess generalization performance on unseen patient data.

| Metric | Logistic Regression (Production) | Random Forest Classifier |
| :--- | :---: | :---: |
| **Accuracy** | **88.33%** | 85.00% |
| **Precision** | **86.49%** | 83.78% |
| **Recall (Sensitivity)** | **91.43%** | 88.57% |
| **F1-Score** | **0.8889** | 0.8611 |
| **5-Fold CV ROC-AUC** | **0.9589** | 0.9120 |

![Model Evaluation Curves](model_evaluation_curves.png)
*Figure 4: Receiver Operating Characteristic (ROC) and Confusion Matrix Curves.*
"""

# 3. Update Section 3.1 with MLflow Artifact
mlflow_section = """### 3.1 Logged Runs & Artifact Summary

![MLflow Logged Confusion Matrix](artifacts_cm.png)
*Figure 5: MLflow Logged Confusion Matrix Artifact.*
"""

# Clean existing image tags to avoid duplicates
import re
content = re.sub(r"!\[.*?\]\(.*?\)\n\*(?:Figure|Image).*?\*\n?", "", content)

# Replace sections
if "### 2.2 Key Visual Insights" in content:
    content = re.sub(r"### 2\.2 Key Visual Insights.*?(?=### 2\.3|## 3)", eda_section + "\n\n", content, flags=re.DOTALL)

if "### 2.4 Model Performance" in content:
    content = re.sub(r"### 2\.4 Model Performance.*?(?=## 3)", eval_section + "\n\n", content, flags=re.DOTALL)
else:
    content = content.replace("## 3. Experiment Tracking with MLflow", eval_section + "\n\n## 3. Experiment Tracking with MLflow")

if "### 3.1 Logged Runs & Artifact Summary" in content:
    content = re.sub(r"### 3\.1 Logged Runs & Artifact Summary.*?(?=### 3\.2|## 4)", mlflow_section + "\n\n", content, flags=re.DOTALL)

with open("report.md", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: report.md updated with all 5 figures!")
