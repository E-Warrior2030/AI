import pandas as pd
import numpy as np


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # Fix column names
    df.columns = df.columns.str.replace(" ", "_")

    # Replace Missing Values with Median
    if 'Lot_Frontage' in df.columns:
        df['Lot_Frontage'] = df['Lot_Frontage'].fillna(df['Lot_Frontage'].median())

    if 'Mas_Vnr_Type' in df.columns:
        df['Mas_Vnr_Type'] = df['Mas_Vnr_Type'].fillna('None')

    df = df.drop(columns=['Alley'], errors='ignore')

    # Remove Duplicates
    df = df.drop_duplicates()

    # Fix Data Types
    if 'MS_SubClass' in df.columns:
        df['MS_SubClass'] = df['MS_SubClass'].astype(str)

    # Deal With Outliers
    if 'SalePrice' in df.columns:
        upper_limit = df['SalePrice'].quantile(0.99)
        df['SalePrice'] = np.where(df['SalePrice'] > upper_limit, upper_limit, df['SalePrice'])

    return df

def save_data(df, path):
    df.to_csv(path, index=False)