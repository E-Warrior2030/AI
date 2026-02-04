import pandas as pd
import numpy as np

def clean_types(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')
    return df

def clean_missing(df):
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else df[c])
    return df

def handle_outliers(df):
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[c] = df[c].clip(lo, hi)
    return df

def clean_strings_and_dates(df):
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.lower()
        parsed = pd.to_datetime(df[c], errors='coerce')
        if parsed.notna().sum() > 0:
            df[c] = parsed
    return df

def validate_cleaned(df):
    assert not df.isna().any().any()
    assert df.duplicated().sum() == 0
    return True

def clean_data(df):
    df = df.copy()
    df = clean_types(df)
    df = clean_missing(df)
    df = handle_outliers(df)
    df = clean_strings_and_dates(df)
    df = df.drop_duplicates()
    validate_cleaned(df)
    return df
