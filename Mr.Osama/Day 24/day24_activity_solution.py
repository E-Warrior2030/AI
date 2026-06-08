"""
Day 24 Activity Solution: Feature Selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

path = "day24_selection.csv"
df = pd.read_csv(path)

y = df["target"]
X = df.drop(columns=["target"])

# Variance threshold
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X)
kept_cols = X.columns[selector.get_support()]
X_var_df = pd.DataFrame(X_var, columns=kept_cols)

# Correlation filter
corr = X_var_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
removed = [c for c in upper.columns if any(upper[c] > 0.9)]
X_reduced = X_var_df.drop(columns=removed)

print("Kept after variance:", list(kept_cols))
print("Removed due to correlation:", removed)
print("Final columns:", X_reduced.columns.tolist())
