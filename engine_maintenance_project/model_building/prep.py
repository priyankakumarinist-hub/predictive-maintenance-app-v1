import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "engine_maintenance_project/data/engine_data.csv"
TARGET = "Engine Condition"
FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]


# Load
df = pd.read_csv(RAW_PATH)
print(f"Loaded raw data: {df.shape}")

# Missing values
n_missing = int(df.isnull().sum().sum())
print(f"Missing values found: {n_missing}")
df = df.dropna()

# Duplicate rows
n_dup = int(df.duplicated().sum())
print(f"Duplicate rows found: {n_dup}")
df = df.drop_duplicates()

# Pressures, temperatures and RPM cannot be negative; drop impossible readings.
df = df[(df[FEATURES] >= 0).all(axis=1)].reset_index(drop=True)

print(f"Cleaned data: {df.shape}")

# 3. Split
X = df[FEATURES]
y = df[TARGET]

# stratify=y keeps the class ratio (~63% faulty / 37% healthy) in both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Save
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print("Train class ratio:\n", ytrain.value_counts(normalize=True).round(3))
print("Test class ratio:\n", ytest.value_counts(normalize=True).round(3))
print("Data prepared: train/test splits written.")
