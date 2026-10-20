with open("report.md", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Section 2.4 Model Performance & Evaluation
sec_2_4 = """### 2.4 Model Performance & Evaluation

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

# Replace Section 2.3 closing or insert 2.4 before Section 3
if "### 2.4 Model Performance" not in content:
    content = content.replace("## 3. Experiment Tracking with MLflow", sec_2_4 + "\n\n## 3. Experiment Tracking with MLflow")

# 2. Add Figure 5 to Section 3.1
sec_3_1_figure = """### 3.1 Logged Runs & Artifact Summary

![MLflow Logged Confusion Matrix](artifacts_cm.png)
*Figure 5: MLflow Logged Confusion Matrix Artifact.*
"""

if "![MLflow Logged Confusion Matrix]" not in content:
    content = content.replace("### 3.1 Logged Runs & Artifact Summary", sec_3_1_figure)

with open("report.md", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: Section 2.4 and Figure 5 in Section 3.1 added to report.md!")
