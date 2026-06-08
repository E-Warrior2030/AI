age_edges = [0, 13, 18, 65, 120]
age_labels = ["Child", "Teen", "Adult", "Senior"]
df18["age_group"] = pd.cut(df18["age"], bins=age_edges, labels=age_labels, right=False)
print(df18["age_group"].value_counts())