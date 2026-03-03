"""
Day 28 Activity Solution: Grouping & Aggregation
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "day28_groupby.csv"
df = pd.read_csv(path)

summary = df.groupby("region")["sales"].agg(["count", "mean", "median", "std"])
print(summary)

mean_sales = df.groupby("region")["sales"].mean().reset_index()
plt.figure(figsize=(6, 3))
sns.barplot(x="region", y="sales", data=mean_sales)
plt.title("Mean Sales by Region")
plt.tight_layout()
plt.show()
