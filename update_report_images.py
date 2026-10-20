with open("report.md", "r", encoding="utf-8") as f:
    content = f.read()

# Replace or insert figure markdown references
eda_figures = """
### 2.2 Key Visual Insights

![Class Balance and Sex Distribution](class_balance.png)
*Figure 1: Target Class Balance and Patient Sex Distribution.*

![Feature Histograms](feature_histograms.png)
*Figure 2: Feature Distributions Across Clinical Attributes.*

![Correlation Heatmap](correlation_heatmap.png)
*Figure 3: Feature Correlation Heatmap with Target Risk.*
"""

eval_figures = """
### 2.4 Model Performance & Evaluation

![Model Evaluation Curves](model_evaluation_curves.png)
*Figure 4: Receiver Operating Characteristic (ROC) and Confusion Matrix Curves.*
"""

cm_figure = """
### 3.1 Logged Runs & Artifact Summary

![MLflow Confusion Matrix](artifacts_cm.png)
*Figure 5: MLflow Logged Confusion Matrix for Logistic Regression.*
"""

# Insert figures into corresponding sections if not already present
if "![Class Balance" not in content:
    content = content.replace("### 2.2 Data Cleaning & Transformation Pipeline", eda_figures + "\n\n### 2.3 Data Cleaning & Transformation Pipeline")

if "![Model Evaluation Curves]" not in content:
    content = content.replace("### 2.4 Model Performance Comparison", eval_figures + "\n\n### 2.5 Model Performance Comparison")

if "![MLflow Confusion Matrix]" not in content:
    content = content.replace("### 3.1 Logged Runs & Artifact Summary", cm_figure)

with open("report.md", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: report.md updated with embedded figures!")
