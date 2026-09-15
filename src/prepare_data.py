import pandas as pd
import os

# -----------------------------------
# 1. Load the raw dataset
# -----------------------------------

input_path = "data/raw/customer_support_conversations.parquet"

print("Loading dataset...")

df = pd.read_parquet(input_path)

print("Full dataset shape:", df.shape)

# -----------------------------------
# 2. Check available companies
# -----------------------------------

print("\nAvailable companies:")
print(df["company"].value_counts())

# -----------------------------------
# 3. Select JetBlue
# -----------------------------------

brand = "JetBlue"

jetblue_df = df[df["company"] == brand].copy()

print("\n-----------------------------------")
print("Selected brand:", brand)
print("JetBlue conversations:", len(jetblue_df))
print("-----------------------------------")

# -----------------------------------
# 4. Display sample conversations
# -----------------------------------

print("\nSample JetBlue conversations:\n")

for i, row in jetblue_df.head(10).iterrows():
    print(f"Conversation {i}:")
    print(row["conversation"])
    print("-" * 80)

# -----------------------------------
# 5. Save JetBlue dataset
# -----------------------------------

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/jetblue_conversations.parquet"

jetblue_df.to_parquet(output_path, index=False)

print("\nJetBlue dataset saved to:")
print(output_path)