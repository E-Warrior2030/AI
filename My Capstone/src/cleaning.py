import pandas as pd
import numpy as np
import os


# ============================================
# Data Cleaning Pipeline
# ============================================


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print("Data loaded:")
    print(df.info())

    return df


def clean_data(df):
    df = df.copy()

    # Fix column names
    df.columns = df.columns.str.replace(" ", "_")

    # Fill some important columns
    if 'Lot_Frontage' in df.columns:
        df['Lot_Frontage'] = df['Lot_Frontage'].fillna(df['Lot_Frontage'].median())

    if 'Mas_Vnr_Type' in df.columns:
        df['Mas_Vnr_Type'] = df['Mas_Vnr_Type'].fillna('None')

    # Drop unnecessary column
    df = df.drop(columns=['Alley'], errors='ignore')

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix data types
    if 'MS_SubClass' in df.columns:
        df['MS_SubClass'] = df['MS_SubClass'].astype(str)

    # Handle outliers
    if 'SalePrice' in df.columns:
        upper_limit = df['SalePrice'].quantile(0.99)
        df['SalePrice'] = np.where(df['SalePrice'] > upper_limit, upper_limit, df['SalePrice'])

    # ✅ REMOVE ALL remaining missing values
    df = df.dropna()

    # ✅ CHECK: ensure no missing values remain
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values still exist!")

    # ✅ CHECK: no negative values in important columns
    important_cols = ['SalePrice', 'Lot_Frontage']
    for col in important_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                raise ValueError(f"Negative values found in {col}!")

    return df


def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Data saved to: {path}")