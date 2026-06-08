import pandas as pd

df = pd.DataFrame({
    "student": ["Ali", "Sara", "Omar", "Lina", "Huda", "Tariq"],
    "math": [90, 78, 65, 88, None, 55],
    "english": [85, 80, None, 92, 75, 60],
    "science": [88, None, 70, 90, 80, 58]
})
df[["math", "english", "science"]] = df[["math", "english", "science"]].apply(lambda col: col.fillna(col.mean()))
df["average"] = df[["math", "english", "science"]].apply(lambda row: row.mean(), axis=1)
df["passed"] = df["average"].apply(lambda x: x >= 70)
print(df)

def fill_missing(column):
    return column.fillna(column.mean())

def calculate_average(row):
    return row[["math", "english", "science"]].mean()

df[["math", "english", "science"]] = df[["math", "english", "science"]].apply(fill_missing)
df["average"] = df.apply(calculate_average, axis=1)

print(df)