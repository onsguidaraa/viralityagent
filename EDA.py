import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the FINAL merged dataset (adjust path if needed based on your config)
# Usually it's in your features folder
PARQUET_PATH = '/content/drive/MyDrive/viral_agent/data/final_model_dataset.parquet' 
df = pd.read_parquet(PARQUET_PATH)

print(f"Loaded final dataset with shape: {df.shape}")

# ==========================================
# PLOT 1: Views vs Log_Views Distribution
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot raw views 
sns.histplot(df['views'], bins=50, ax=axes[0], color='blue', kde=True)
axes[0].set_title('Raw Views Distribution (Highly Skewed)')
axes[0].set_xlabel('Views')

# Plot log_views 
sns.histplot(df['log_views'], bins=50, ax=axes[1], color='green', kde=True)
axes[1].set_title('Log-Transformed Views (Normalized)')
axes[1].set_xlabel('Log(Views + 1)')

plt.tight_layout()
plt.savefig('/content/target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved 'target_distribution.png'")

# ==========================================
# PLOT 2: Correlation Matrix (Features vs Views)
# ==========================================
plt.figure(figsize=(12, 10))

# We cannot plot all 400+ columns, so we select the most important ones for the report!
cols_to_plot = [
    'views', 'log_views', 
    'duration_seconds', 
    'rms_max', 'tempo', 'silence_ratio', 
    'hook_score', 'clarity_score'
]

# Only keep columns that actually exist in the dataframe (prevents errors)
existing_cols = [col for col in cols_to_plot if col in df.columns]

corr = df[existing_cols].corr()

# Draw the heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix: AI Features vs Performance")

plt.savefig('/content/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved 'correlation_matrix.png'")