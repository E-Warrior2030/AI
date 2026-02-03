import pandas as pd
import re

df = pd.read_csv("data/day12_users.csv")

def standardize_city(df):
    df["city_clean"] = (
        df["city"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    mapping = {
        "nyc": "new york",
        "new york city": "new york",
        "la": "los angeles",
        "sf": "san francisco",
        "san fran": "san francisco",
    }
    df["city_clean"] = df["city_clean"].replace(mapping)
    return df

def parse_and_localize(df):
    df["timestamp_clean"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df

df = standardize_city(df)
df = parse_and_localize(df)

print(df[["city", "city_clean", "timestamp", "timestamp_clean"]])
