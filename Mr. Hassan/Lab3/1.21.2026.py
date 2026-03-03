import pandas as pd
import numpy as np

df = pd.read_csv("day06_user_data.csv")

df.replace(["N/A", "?", "", "unknown"], np.nan, inplace=True)

df["age"] = pd.to_numeric(df["age"])
df["income"] = pd.to_numeric(df["income"])

df["age"] = df["age"].fillna(df["age"].median())
df["income"] = df["income"].fillna(df["income"].mean())

print(df)
