import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Directories setup
os.makedirs("data", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
csv_path = "data/heart.csv"

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]

print("Downloading UCI Heart Disease dataset...")
urllib.request.urlretrieve(data_url, csv_path)

# 2. Data Cleaning
df = pd.read_csv(csv_path, names=columns, na_values="?")
df = df.dropna().reset_index(drop=True)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Set global seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

# --- Plot 1: Explicit Sex Distribution ---
plt.figure(figsize=(7, 4.5))
df_sex = df.copy()
df_sex["sex_label"] = df_sex["sex"].map({0: "Female", 1: "Male"})
ax = sns.countplot(data=df_sex, x="sex_label", palette=["#e74c3c", "#3498db"])
plt.title("Patient Gender Distribution (Sex)", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Patient Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("artifacts/sex_distribution.png", dpi=300)
plt.close()

# --- Plot 2: Class Balance ---
plt.figure(figsize=(7, 4.5))
df_target = df.copy()
df_target["target_label"] = df_target["target"].map({0: "Healthy (0)", 1: "Heart Disease (1)"})
ax = sns.countplot(data=df_target, x="target_label", palette=["#2ecc71", "#e74c3c"])
plt.title("Target Class Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Clinical Diagnostic Target")
plt.ylabel("Patient Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig("artifacts/class_balance.png", dpi=300)
plt.close()

# --- Plot 3: Clean Feature Histograms Grid ---
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

# Map human-readable labels for histograms
hist_data = df.copy()

for i, col in enumerate(columns):
    ax = axes[i]
    if col == "sex":
        counts = hist_data["sex"].map({0: "Female", 1: "Male"}).value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#e74c3c", "#3498db"])
        ax.set_title("sex (Gender)", fontweight="bold")
    elif col == "target":
        counts = hist_data["target"].map({0: "Healthy", 1: "Disease"}).value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#2ecc71", "#e74c3c"])
        ax.set_title("target (Status)", fontweight="bold")
    else:
        sns.histplot(hist_data[col], kde=True, ax=ax, color="#2980b9", bins=15)
        ax.set_title(col, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

# Hide remaining empty subplots
for j in range(len(columns), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Clinical Feature Distributions & Frequency Histograms", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("artifacts/feature_histograms.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Plot 4: Correlation Heatmap ---
plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/correlation_heatmap.png", dpi=300)
plt.close()

print("NEW EDA PLOTS GENERATED WITH EXPLICIT LABELS (Female / Male)!")
