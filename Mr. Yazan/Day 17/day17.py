from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
df17_mm = df17.copy()
df17_mm[["CRIM_mm", "RM_mm"]] = mm_scaler.fit_transform(df17[["CRIM", "RM"]])
