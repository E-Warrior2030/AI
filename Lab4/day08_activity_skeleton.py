import pandas as pd

df = df.drop_duplicates()
df = df.drop_duplicates(subset=["user", "day", "product"])
result = df.groupby("user").agg(event_count=("product", "count"), ever_clicked=("clicked", "max")).reset_index()
print(result)