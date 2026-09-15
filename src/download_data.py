from datasets import load_dataset
import os

# Create raw data directory
os.makedirs("data/raw", exist_ok=True)

print("Loading Customer Support on Twitter dataset...")

# Load the dataset
dataset = load_dataset(
    "TNE-AI/customer-support-on-twitter-conversation",
    split="train"
)

# Convert to pandas
df = dataset.to_pandas()

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nCompany distribution:")
print(df["company"].value_counts().head(20))

# Save dataset
output_path = "data/raw/customer_support_conversations.parquet"

df.to_parquet(output_path, index=False)

print(f"\nDataset saved to: {output_path}")