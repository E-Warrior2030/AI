import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "day30_eda.csv"
df = pd.read_csv(path)

# Univariate
for col in ["age", "income"]:
    plt.figure(figsize=(5, 3))
    df[col].hist(bins=20)
    plt.title(f"{col} histogram")
    plt.tight_layout()
    plt.show()

# Boxplot by group
plt.figure(figsize=(5, 3))
sns.boxplot(x="segment", y="income", data=df)
plt.title("Income by Segment")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df[["age", "income", "spend"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("EDA Correlations")
plt.tight_layout()
plt.show()