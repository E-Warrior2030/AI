# Capstone Project Report

## Introduction
This project analyzes the Ames Housing dataset, which contains detailed information about residential property sales. The dataset includes a wide range of features such as house size, quality, location, and other structural attributes.

The main objective of this project was to:
- Clean and preprocess the dataset to ensure data quality  
- Engineer meaningful features to improve analysis  
- Perform exploratory data analysis (EDA) to discover patterns and relationships  
- Understand the key factors that influence house prices  

This project follows a structured data pipeline approach, similar to real-world data science workflows.

---

## Data Cleaning
The dataset initially contained missing values, inconsistent formats, and potential outliers. Several steps were taken to prepare the data:

- Standardized column names by replacing spaces with underscores for consistency  
- Filled missing numerical values (such as `Lot_Frontage`) using the median to reduce the effect of outliers  
- Filled categorical missing values (such as `Mas_Vnr_Type`) with "None"  
- Removed columns with excessive missing data (e.g., `Alley`)  
- Removed duplicate rows to avoid bias in analysis  
- Converted `MS_SubClass` to a categorical format for better interpretation  
- Capped outliers in `SalePrice` using the 99th percentile to reduce extreme values  
- Removed any remaining missing values using `dropna()` to ensure a fully clean dataset  

### Data Validation (Checks)
To ensure the dataset is reliable after cleaning, validation checks were added:

- Verified that the dataset file exists before loading  
- Ensured no missing values remain after cleaning  
- Checked that important numerical columns (such as `SalePrice`) do not contain invalid negative values  

These checks improve the robustness of the pipeline and prevent errors during analysis.

---

## Feature Engineering
To improve the dataset and extract more meaningful insights, several new features were created:

- Applied one-hot encoding to categorical variables  
- Used ordinal encoding for quality-related features  
- Scaled numerical features using `StandardScaler`  
- Created `price_per_sqft` to normalize price relative to house size  
- Created `quality_area` as an interaction between quality and size  
- Applied log transformation to `SalePrice` to reduce skewness  
- Grouped houses into age categories to simplify analysis  

These transformations help make the data more suitable for analysis and future machine learning models.

---

## Exploratory Data Analysis (EDA)
An extensive EDA process was implemented using visualizations and statistical techniques:

- **Histograms** were used to analyze the distribution of key variables such as `SalePrice`, `Gr_Liv_Area`, and `Overall_Qual`  
- **Boxplots** were used to detect outliers and compare price distributions across quality levels  
- **Scatter plots** revealed relationships between living area and sale price  
- **Correlation heatmaps** were used to identify the most important features affecting house prices  
- **Groupby analysis** was used to compare average prices across different neighborhoods  

### Statistical Analysis
Additional statistical methods were applied:

- Calculated mean and standard deviation of `SalePrice`  
- Used **Z-score normalization** to detect outliers  
- Applied **cosine similarity** to compare houses with the highest and lowest prices  
- Estimated probability of a house having a price above the mean  

### Improvements Made to EDA Code
During development, several issues were identified and fixed:

- Handled missing values before plotting to avoid runtime warnings  
- Removed invalid or empty data from correlation calculations  
- Added checks to prevent division by zero in Z-score calculations  
- Ensured cosine similarity does not fail when vectors contain only zeros  
- Skipped visualizations when insufficient data is available  

These improvements made the EDA process more stable, accurate, and professional.

---

## Key Findings
From the analysis, several important insights were discovered:

1. **Overall quality** has the strongest impact on house prices  
2. **Living area (size)** shows a strong positive correlation with price  
3. **Newer houses** tend to be more expensive than older ones  
4. Some features have weak or no correlation, meaning they are less important for prediction  
5. The distribution of `SalePrice` is skewed and benefits from transformation  

---

## Future Work
There are several ways this project can be extended:

- Build a machine learning model to predict house prices  
- Experiment with advanced models such as Random Forest or Gradient Boosting  
- Perform deeper feature selection and dimensionality reduction  
- Analyze location and neighborhood effects in more detail  
- Develop an interactive dashboard for better visualization  

---

## AI Part in This Project
I used AI in a few specific ways to help me with this project. For example, it helped me understand and remember certain functions and how to clean, fix, sort, and remove data. I also used AI to guide me in writing model code, showing me the correct approach instead of just copying and pasting.

AI didn’t only help with the code—it also helped me improve the report. I wrote everything in my own words, then asked AI to refine it. This helped improve the grammar, spelling, and vocabulary while keeping the meaning and ideas original.

Additionally, AI helped identify issues in the EDA process, such as handling missing values, preventing division errors, and improving the stability of the analysis code.

---
