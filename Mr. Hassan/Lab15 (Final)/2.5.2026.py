import pandas as pd
import numpy as np

df = pd.read_csv("/mnt/data/dataset.csv")

def clean_data(df):
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper)

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip().str.lower()

    return df

df_clean = clean_data(df)
df_clean.to_csv("/mnt/data/cleaned.csv", index=False)
print(df_clean.info())
print(df_clean.head())
