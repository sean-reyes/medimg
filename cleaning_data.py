import pandas as pd

df = pd.read_csv("data/raw_data/raw_data.csv")

# Drop id first, then remove duplicates
df = df.drop(columns=["id"])
df = df.drop_duplicates()

# Drop columns that are mostly empty
df = df.drop(columns=["ca", "thal"])

# Impossible zeros become blanks
df["chol"] = df["chol"].replace(0, pd.NA)
df["trestbps"] = df["trestbps"].replace(0, pd.NA)

# True/False to 1/0 (handles real booleans or text)
mapping = {True: 1, False: 0, "TRUE": 1, "FALSE": 0}
df["fbs"] = df["fbs"].map(mapping)
df["exang"] = df["exang"].map(mapping)

# Blank slope becomes its own category
df["slope"] = df["slope"].fillna("unknown")

# Target
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

print(df.shape)
print(df.isna().sum())
df.to_csv("data/processed_data/processed_data.csv", index=False)