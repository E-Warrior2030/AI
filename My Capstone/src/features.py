import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    df = df.copy()

    # --- One-hot encoding (categorical → numeric) ---
    categorical_cols = ['Neighborhood', 'Bldg_Type']

    for col in categorical_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    # --- Ordinal encoding ---
    if 'Exter_Qual' in df.columns:
        qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        df['Exter_Qual'] = df['Exter_Qual'].map(qual_map)

    # --- Scaling ---
    scaler = StandardScaler()
    num_cols = ['Gr_Liv_Area', 'SalePrice']

    for col in num_cols:
        if col in df.columns:
            df[col + "_scaled"] = scaler.fit_transform(df[[col]])

    # --- Domain feature (ratio) ---
    if 'SalePrice' in df.columns and 'Gr_Liv_Area' in df.columns:
        df['price_per_sqft'] = df['SalePrice'] / (df['Gr_Liv_Area'] + 1)

    # --- Interaction feature ---
    if 'Overall_Qual' in df.columns and 'Gr_Liv_Area' in df.columns:
        df['quality_area'] = df['Overall_Qual'] * df['Gr_Liv_Area']

    # --- Log transform ---
    if 'SalePrice' in df.columns:
        df['SalePrice_log'] = np.log1p(df['SalePrice'])

    # --- Binning ---
    if 'Year_Built' in df.columns:
        df['house_age'] = 2025 - df['Year_Built']
        df['age_group'] = pd.cut(df['house_age'],
                                bins=[0, 10, 30, 100],
                                labels=['New', 'Mid', 'Old'])

    return df