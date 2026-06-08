import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "day27_boxplots.csv"
df = pd.read_csv(path)

plt.figure(figsize=(6, 3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))


df["score"].hist(bins=20, edgecolor="black")
plt.title("Score Histogram")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
sns.boxplot(x="group", y="score", data=df)
plt.title("Score by Group")
plt.tight_layout()
plt.show()