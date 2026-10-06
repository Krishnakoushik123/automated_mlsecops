import pandas as pd
from sklearn.datasets import make_classification, make_regression
import os

os.makedirs('scratch', exist_ok=True)

# Classification CSV
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
df_class = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df_class['target'] = y
df_class.to_csv('scratch/class_data.csv', index=False)

# Regression JSON
X, y = make_regression(n_samples=500, n_features=10, random_state=42)
df_reg = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df_reg['target'] = y
df_reg.to_json('scratch/reg_data.json', orient='records')

print("Created datasets.")
