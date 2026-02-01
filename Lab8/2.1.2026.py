import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Ali", "Sara", "Omar", "Ali", None],
    "math": [90, None, 70, 90, 80],
    "english": [85, 80, None, 85, 75],
    "gender": ["M", None, "M", "M", "F"]
})

df["math"] = df["math"].fillna(df["math"].mean())
df["english"] = df["english"].fillna(df["english"].median())
df["gender"] = df["gender"].fillna(df["gender"].mode()[0])

df = df.drop_duplicates()

print(df)
