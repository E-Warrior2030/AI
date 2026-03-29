# Capstone Project Report

## Introduction
This project analyzes the Ames Housing dataset, which contains detailed information about residential property sales. The goal was to clean the dataset, engineer useful features, and uncover patterns affecting house prices.

## Data Cleaning
- Standardized column names by replacing spaces with underscores
- Filled missing numerical values using median
- Filled categorical missing values with "None"
- Removed columns with excessive missing data
- Removed duplicate rows
- Converted MS_SubClass to categorical
- Capped outliers in SalePrice at the 99th percentile

## Feature Engineering
- Applied one-hot encoding to categorical variables
- Used ordinal encoding for quality-related features
- Scaled numerical features using StandardScaler
- Created price_per_sqft to normalize price by size
- Created quality_area as an interaction feature
- Applied log transformation to SalePrice
- Grouped houses into age categories

## Key Findings
1. Overall quality has the strongest impact on house prices.
2. Larger houses tend to have higher prices, showing a strong positive relationship.
3. Newer houses are generally more expensive than older ones.

## Future Work
- Build a machine learning model to predict prices
- Explore additional feature interactions
- Analyze location effects more deeply
- Build a dashboard for visualization

## AI Part in This Project
I used AI in a few specific ways to help me with this project. For example, it helped me understand and remember certain functions and how to clean, fix, sort, and remove data. I also used AI to guide me in writing model code, showing me the correct approach instead of just copying and pasting. AI didn’t only help with the code—it also helped me improve the report. I wrote everything in my own words, then asked AI to fix it, which made the grammar, spelling, and vocabulary better.

