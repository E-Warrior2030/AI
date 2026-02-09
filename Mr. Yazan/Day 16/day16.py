from sklearn.preprocessing import LabelEncoder

le_city = LabelEncoder()
df16["city_label"] = le_city.fit_transform(df16["city"])
print("Classes:", le_city.classes_)


