import pandas as pd

# Load JetBlue dataset
input_path = "data/processed/jetblue_conversations.parquet"

df = pd.read_parquet(input_path)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst conversation:")
print("=" * 80)

print(df.iloc[0]["conversation"])

print("=" * 80)

print("\nSecond conversation:")
print("=" * 80)

print(df.iloc[1]["conversation"])

print("=" * 80)