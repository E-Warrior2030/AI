from src.cleaning import load_data, clean_data, save_data
from src.features import engineer_features
from src.eda import run_eda
import os

# 1. Load data
df = load_data("data/raw/ames.csv")
print("Before cleaning:", df.shape)

# 2. Clean data
df = clean_data(df)
print("After cleaning:", df.shape)

# 3. Feature engineering
df = engineer_features(df)
print("After feature engineering:", df.shape)

# 4. Save
save_data(df, "data/cleaned/ames_features.csv")

print("Pipeline complete ✅")

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Run EDA
run_eda(df)

print("EDA complete ✅")