import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df):

    # --- 1. Histograms ---
    num_cols = ['SalePrice', 'Gr_Liv_Area', 'Overall_Qual']
    for col in num_cols:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col].dropna(), kde=True)  # dropna avoids errors
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            file_path = f"plots/hist_{col}.png"
            plt.savefig(file_path)
            plt.close()
            print(f"[INFO] Saved histogram: {file_path}")

    # --- 2. Boxplot ---
    if 'SalePrice' in df.columns and 'Overall_Qual' in df.columns:
        plt.figure()
        sns.boxplot(x=df['Overall_Qual'], y=df['SalePrice'])
        plt.title("SalePrice vs Overall Quality")
        file_path = "plots/boxplot_quality_price.png"
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Saved boxplot: {file_path}")

    # --- 3. Correlation heatmap ---
    numeric_df = df.select_dtypes(include=np.number)
    if 'SalePrice' in numeric_df.columns:
        plt.figure(figsize=(10, 8))
        top_corr = numeric_df.corr()['SalePrice'].abs().sort_values(ascending=False).head(10).index
        sns.heatmap(numeric_df[top_corr].corr(), annot=True, cmap='coolwarm')
        plt.title("Top Correlated Features with SalePrice")
        file_path = "plots/heatmap.png"
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Saved heatmap: {file_path}")

    # --- 4. Scatter plot ---
    if 'Gr_Liv_Area' in df.columns and 'SalePrice' in df.columns:
        plt.figure()
        sns.scatterplot(x=df['Gr_Liv_Area'], y=df['SalePrice'])
        plt.title("Living Area vs Sale Price")
        file_path = "plots/scatter_area_price.png"
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Saved scatter plot: {file_path}")

    # --- 5. Groupby example ---
    if 'Neighborhood_NAmes' in df.columns:
        group = df.groupby('Neighborhood_NAmes')['SalePrice'].mean()
        print("Groupby example:\n", group.head())

    # --- 6. Math & Stats ---
    if 'SalePrice' in df.columns:
        prices = df['SalePrice'].values
        mean = np.mean(prices)
        std = np.std(prices)
        print("Mean:", mean)
        print("Std:", std)

        # Z-score
        z = (prices - mean) / std
        print("Z-score sample:", z[:5])

        # Cosine similarity
        numeric_cols = numeric_df.columns
        high = df.loc[df['SalePrice'].idxmax(), numeric_cols].fillna(0)  # replace NaN with 0
        low = df.loc[df['SalePrice'].idxmin(), numeric_cols].fillna(0)
        cos_sim = np.dot(high, low) / (np.linalg.norm(high) * np.linalg.norm(low))
        print("Cosine similarity (max vs min SalePrice):", cos_sim)

        # Probability
        prob = np.mean(df['SalePrice'] > mean)
        print("P(SalePrice > mean):", prob)