import numpy as np
import pandas as pd

np.random.seed(10)
values = np.concatenate([np.random.lognormal(10, 0.5, 1000), [1e7, 2e7]])
df = pd.DataFrame({"income": values})

def iqr_bounds(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def detect_outliers_iqr(s):
    low, high = iqr_bounds(s)
    return (s < low) | (s > high)

def detect_outliers_zscore(s, thresh=3):
    z = (s - s.mean()) / s.std(ddof=0)
    return np.abs(z) > thresh

low, high = iqr_bounds(df["income"])
df["income_capped"] = df["income"].clip(lower=low, upper=high)
df["income_log1p"] = np.log1p(df["income"])

iqr_outliers = detect_outliers_iqr(df["income"])
z_outliers = detect_outliers_zscore(df["income"])

print(iqr_outliers.sum(), z_outliers.sum())
print(df.head())