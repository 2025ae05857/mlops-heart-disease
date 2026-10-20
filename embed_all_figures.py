with open("report.md", "r", encoding="utf-8") as f:
    content = f.read()

figures_markdown = """
## Visual Artifacts & Figures

### EDA Visualizations
![Class Balance and Sex Distribution](class_balance.png)
*Figure 1: Target Class Balance and Patient Sex Distribution.*

![Feature Histograms](feature_histograms.png)
*Figure 2: Feature Distributions Across Clinical Attributes.*

![Correlation Heatmap](correlation_heatmap.png)
*Figure 3: Feature Correlation Heatmap with Target Risk.*

---

### Model Performance Visualizations
![Model Evaluation Curves](model_evaluation_curves.png)
*Figure 4: Receiver Operating Characteristic (ROC) and Confusion Matrix Curves.*

![MLflow Logged Confusion Matrix](artifacts_cm.png)
*Figure 5: MLflow Logged Confusion Matrix for Production Model.*
"""

if "## Visual Artifacts & Figures" not in content:
    content = content + "\n\n" + figures_markdown

with open("report.md", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: report.md updated with all 5 embedded figures!")
