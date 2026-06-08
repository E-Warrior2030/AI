"""
Day 22 Activity Solution: Interaction Features
"""

import pandas as pd

path = "day22_interactions.csv"
df = pd.read_csv(path)

df["feature1_x_feature2"] = df["feature1"] * df["feature2"]

df["feature1_plus_feature2"] = df["feature1"] + df["feature2"]


df["high_f1_and_flag"] = ((df["feature1"] > df["feature1"].median()) & (df["flag"] == 1)).astype(int)

corrs = df[["feature1", "feature2", "feature1_x_feature2", "feature1_plus_feature2", "high_f1_and_flag", "target"]].corr()["target"]
print(corrs.sort_values(ascending=False))