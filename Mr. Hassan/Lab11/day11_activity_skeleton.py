import pandas as pd
import numpy as np

df = pd.read_csv("data/day11_income.csv")

def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)

def remove_upper_tail(s, upper_q=0.99):
    hi = s.quantile(upper_q)
    return s[s <= hi]

orig_stats = df.describe()

df["income_winsorized"] = winsorize_series(df["income"])
df_removed = df.loc[remove_upper_tail(df["income"]).index]

wins_stats = df["income_winsorized"].describe()
removed_stats = df_removed["income"].describe()

print("Original:\n", orig_stats)
print("\nWinsorized:\n", wins_stats)
print("\nUpper-tail removed:\n", removed_stats)
