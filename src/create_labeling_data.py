import pandas as pd
import re
import os

# ---------------------------------------
# Load JetBlue conversations
# ---------------------------------------

input_path = "data/processed/jetblue_conversations.parquet"

df = pd.read_parquet(input_path)

print("Total JetBlue conversations:", len(df))


# ---------------------------------------
# Extract first customer message
# ---------------------------------------

def extract_customer_message(conversation):
    """
    Extract the first Customer message from a conversation.
    """

    if not isinstance(conversation, str):
        return None

    lines = conversation.splitlines()

    for line in lines:
        line = line.strip()

        if line.lower().startswith("customer:"):
            message = line[len("customer:"):].strip()

            if message:
                return message

    return None


# ---------------------------------------
# Extract first support response
# ---------------------------------------

def extract_support_reply(conversation):
    """
    Extract the first Support response from a conversation.
    """

    if not isinstance(conversation, str):
        return None

    lines = conversation.splitlines()

    for line in lines:
        line = line.strip()

        if line.lower().startswith("support:"):
            message = line[len("support:"):].strip()

            if message:
                return message

    return None


# ---------------------------------------
# Create candidate examples
# ---------------------------------------

records = []

for _, row in df.iterrows():

    conversation = row["conversation"]

    customer_message = extract_customer_message(conversation)
    support_reply = extract_support_reply(conversation)

    # We need both sides for useful training examples
    if customer_message and support_reply:

        records.append({
            "conversation_id": row.get("conversation_id", ""),
            "customer_message": customer_message,
            "support_reply": support_reply,
            "intent": "",
            "escalation": "",
            "escalation_reason": ""
        })


# ---------------------------------------
# Convert to DataFrame
# ---------------------------------------

examples = pd.DataFrame(records)

print("Usable conversation pairs:", len(examples))


# ---------------------------------------
# Remove duplicate customer messages
# ---------------------------------------

examples = examples.drop_duplicates(
    subset=["customer_message"]
)


# ---------------------------------------
# Select 200 examples
# ---------------------------------------

examples = examples.sample(
    n=200,
    random_state=42
).reset_index(drop=True)

# Add an ID
examples.insert(
    0,
    "example_id",
    range(1, len(examples) + 1)
)


# ---------------------------------------
# Save labeling file
# ---------------------------------------

os.makedirs("data/processed", exist_ok=True)

output_path = "data/processed/jetblue_labeling.csv"

examples.to_csv(
    output_path,
    index=False
)

print("\nCreated labeling dataset:")
print(output_path)

print("\nNumber of examples:", len(examples))

print("\nColumns:")
print(examples.columns.tolist())

print("\nFirst 5 examples:")
print(
    examples[
        [
            "example_id",
            "customer_message",
            "support_reply"
        ]
    ].head()
)