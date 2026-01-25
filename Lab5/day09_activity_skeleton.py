import pandas as pd

def normalize_schema(df):
    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["income"] = pd.to_numeric(out["income"].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    out["signup"] = pd.to_datetime(out["signup"], errors="coerce", infer_datetime_format=True)
    return out

normalized = normalize_schema(df)
print(normalized.dtypes)
print(normalized.isna().sum())