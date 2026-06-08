"""
Day 23 Activity Solution: Polynomial Features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

path = "day23_poly.csv"
df = pd.read_csv(path)

X = df[["x"]].values
y = df["y"].values

for degree in [1, 2, 5]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    score = model.score(X_poly, y)
    print(f"Degree {degree} R^2:", score)
