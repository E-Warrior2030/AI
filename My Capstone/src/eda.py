import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df):

    # --- 1. Histograms ---
    # A histogram shows the distribution of numerical data by grouping values into bins.
    # It is used to understand data spread, skewness, and frequency, and is commonly used in statistical analysis.
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
    # A boxplot visualizes data distribution using quartiles and highlights outliers.
    # It is useful for comparing distributions across categories and detecting skewness or anomalies.
    if 'SalePrice' in df.columns and 'Overall_Qual' in df.columns:
        plt.figure()
        sns.boxplot(x=df['Overall_Qual'], y=df['SalePrice'])
        plt.title("SalePrice vs Overall Quality")
        file_path = "plots/boxplot_quality_price.png"
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Saved boxplot: {file_path}")

    # --- 3. Correlation heatmap ---
    # A heatmap shows relationships between variables using color intensity to represent correlation strength.
    # It is commonly used to identify important features and detect multicollinearity in datasets.
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
    # A scatter plot shows the relationship between two numerical variables using points.
    # It is used to identify correlations, trends, and patterns between features.
    if 'Gr_Liv_Area' in df.columns and 'SalePrice' in df.columns:
        plt.figure()
        sns.scatterplot(x=df['Gr_Liv_Area'], y=df['SalePrice'])
        plt.title("Living Area vs Sale Price")
        file_path = "plots/scatter_area_price.png"
        plt.savefig(file_path)
        plt.close()
        print(f"[INFO] Saved scatter plot: {file_path}")

    # --- 5. Groupby example ---
    # Groupby aggregates data by categories to compute summary statistics like mean or sum.
    # It is commonly used for analyzing patterns across different groups or segments.
    if 'Neighborhood_NAmes' in df.columns:
        group = df.groupby('Neighborhood_NAmes')['SalePrice'].mean()
        print("Groupby example:\n", group.head())

    # --- 6. Math & Stats ---
    # Basic statistical measures help summarize the dataset and understand its distribution.
    # These techniques are widely used in data preprocessing and feature analysis.
    if 'SalePrice' in df.columns:
        prices = df['SalePrice'].values
        mean = np.mean(prices)
        std = np.std(prices)
        print("Mean:", mean)
        print("Std:", std)

        # Z-score measures how many standard deviations a value is from the mean.
        # It is used for normalization and detecting outliers in data.
        z = (prices - mean) / std
        print("Z-score sample:", z[:5])

        # Cosine similarity measures similarity between two vectors based on angle.
        # It is commonly used in machine learning to compare data points or features.
        numeric_cols = numeric_df.columns
        high = df.loc[df['SalePrice'].idxmax(), numeric_cols].fillna(0)  # replace NaN with 0
        low = df.loc[df['SalePrice'].idxmin(), numeric_cols].fillna(0)
        cos_sim = np.dot(high, low) / (np.linalg.norm(high) * np.linalg.norm(low))
        print("Cosine similarity (max vs min SalePrice):", cos_sim)

        # Probability here estimates the likelihood of a value being above the mean.
        # It is used to understand data distribution and relative frequency.
        prob = np.mean(df['SalePrice'] > mean)
        print("P(SalePrice > mean):", prob)