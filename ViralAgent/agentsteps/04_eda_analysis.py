"""
04_eda_analysis.py
──────────────────
Performs full EDA on final_dataset.parquet:
- Histograms & distribution plots
- Correlation analysis
- Outlier detection
- Missing-value profiling
- Train/test split
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# CONFIG (edit if needed)
# ─────────────────────────────────────────────────────────────
DATA_PATH = "/content/drive/MyDrive/viral_agent/features/final_dataset.parquet"
OUTPUT_DIR = "/content/eda_outputs"
TARGET_COL = "log_views"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# ─────────────────────────────────────────────────────────────
# 1) HISTOGRAMS / DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
sns.histplot(df[TARGET_COL], bins=40, kde=True, color="seagreen")
plt.title("Distribution of log_views")
plt.savefig(f"{OUTPUT_DIR}/distribution_log_views.png", dpi=300, bbox_inches="tight")
plt.close()

# Example: duration distribution if exists
if "duration_seconds" in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df["duration_seconds"], bins=40, kde=True, color="steelblue")
    plt.title("Distribution of video duration (seconds)")
    plt.savefig(f"{OUTPUT_DIR}/distribution_duration.png", dpi=300, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────
# 2) CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────
corr = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Numerical Features)")
plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Top correlations with target
corr_target = corr[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
corr_target.to_csv(f"{OUTPUT_DIR}/top_correlations.csv")

# ─────────────────────────────────────────────────────────────
# 3) OUTLIER DETECTION (IQR METHOD)
# ─────────────────────────────────────────────────────────────
outlier_report = {}

for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_report[col] = int(outliers)

with open(f"{OUTPUT_DIR}/outlier_report.json", "w") as f:
    json.dump(outlier_report, f, indent=2)

# ─────────────────────────────────────────────────────────────
# 4) MISSING-VALUE PROFILING
# ─────────────────────────────────────────────────────────────
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing.to_csv(f"{OUTPUT_DIR}/missing_values.csv")

# Bar plot for missing values (if any)
if not missing.empty:
    plt.figure(figsize=(10,6))
    missing.plot(kind="bar")
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    plt.savefig(f"{OUTPUT_DIR}/missing_values.png", dpi=300, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────
# 5) TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL], errors="ignore")
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape :", X_test.shape, y_test.shape)

# Save split sizes for report
with open(f"{OUTPUT_DIR}/train_test_split.txt", "w") as f:
    f.write(f"Train size: {X_train.shape}\n")
    f.write(f"Test size: {X_test.shape}\n")

print(f"EDA outputs saved to: {OUTPUT_DIR}")