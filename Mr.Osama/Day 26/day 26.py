import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "day26_distributions.csv"
df = pd.read_csv(path)

num_cols = ["income", "age", "transactions"]

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    df[col].hist(bins=30,edgecolor="black", ax=axes[0])
    axes[0].set_title(f"{col} Histogram")
    sns.kdeplot(df[col], ax=axes[1])
    axes[1].set_title(f"{col} KDE")
    plt.tight_layout()
    plt.show()
print(df[num_cols].describe())

# Log transform on income
if (df["income"] > 0).all():
    df["income_log1p"] = np.log1p(df["income"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    df["income"].hist(bins=30, ax=axes[0])
    axes[0].set_title("Income Raw")
    df["income_log1p"].hist(bins=30, ax=axes[1])
    axes[1].set_title("Income log1p")
    plt.tight_layout()
    plt.show()