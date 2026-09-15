import time
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from intent_rules import rule_based_intent
from escalation_rules import rule_based_escalation


DATA_PATH = "data/processed/jetblue_labeling.csv"

INTENT_MODEL_PATH = "models/intent_classifier.pkl"
ESCALATION_MODEL_PATH = "models/escalation_classifier.pkl"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=[
        "customer_message",
        "intent",
        "escalation"
    ]
).reset_index(drop=True)


# --------------------------------------------------
# Create the same held-out test split
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"]
)

print("\n")
print("=" * 70)
print("HIVER AI SUPPORT AGENT - HELD-OUT EVALUATION")
print("=" * 70)

print(f"\nTotal labeled examples: {len(df)}")
print(f"Training examples: {len(train_df)}")
print(f"Held-out test examples: {len(test_df)}")


# --------------------------------------------------
# Load trained models
# --------------------------------------------------

intent_model = joblib.load(
    INTENT_MODEL_PATH
)

escalation_model = joblib.load(
    ESCALATION_MODEL_PATH
)


# --------------------------------------------------
# Intent evaluation
# --------------------------------------------------

intent_predictions = []
intent_sources = []

intent_start = time.perf_counter()

for message in test_df["customer_message"].astype(str):

    rule_intent = rule_based_intent(message)

    if rule_intent is not None:

        predicted_intent = rule_intent
        source = "rule"

    else:

        predicted_intent = intent_model.predict(
            [message]
        )[0]

        source = "ml"

    intent_predictions.append(
        predicted_intent
    )

    intent_sources.append(
        source
    )


intent_time = time.perf_counter() - intent_start


# --------------------------------------------------
# Escalation evaluation
# --------------------------------------------------

escalation_predictions = []
escalation_sources = []

escalation_start = time.perf_counter()

for message, intent in zip(
    test_df["customer_message"].astype(str),
    intent_predictions
):

    rule_escalation = rule_based_escalation(
        message,
        intent
    )

    if rule_escalation is not None:

        predicted_escalation = rule_escalation
        source = "rule"

    else:

        predicted_escalation = escalation_model.predict(
            [message]
        )[0]

        source = "ml"

    escalation_predictions.append(
        predicted_escalation
    )

    escalation_sources.append(
        source
    )


escalation_time = (
    time.perf_counter() - escalation_start
)


# --------------------------------------------------
# Add predictions
# --------------------------------------------------

test_df = test_df.copy()

test_df["predicted_intent"] = (
    intent_predictions
)

test_df["intent_source"] = (
    intent_sources
)

test_df["predicted_escalation"] = (
    escalation_predictions
)

test_df["escalation_source"] = (
    escalation_sources
)


# --------------------------------------------------
# Intent metrics
# --------------------------------------------------

intent_accuracy = accuracy_score(
    test_df["intent"],
    test_df["predicted_intent"]
)

print("\n")
print("=" * 70)
print("INTENT CLASSIFICATION")
print("=" * 70)

print(
    f"\nAccuracy: {intent_accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        test_df["intent"],
        test_df["predicted_intent"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        test_df["intent"],
        test_df["predicted_intent"]
    )
)


# --------------------------------------------------
# Escalation metrics
# --------------------------------------------------

escalation_accuracy = accuracy_score(
    test_df["escalation"],
    test_df["predicted_escalation"]
)

print("\n")
print("=" * 70)
print("ESCALATION CLASSIFICATION")
print("=" * 70)

print(
    f"\nAccuracy: "
    f"{escalation_accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        test_df["escalation"],
        test_df["predicted_escalation"],
        zero_division=0
    )
)


# --------------------------------------------------
# Rule / ML usage
# --------------------------------------------------

intent_rule_count = (
    test_df["intent_source"] == "rule"
).sum()

intent_ml_count = (
    test_df["intent_source"] == "ml"
).sum()

escalation_rule_count = (
    test_df["escalation_source"] == "rule"
).sum()

escalation_ml_count = (
    test_df["escalation_source"] == "ml"
).sum()


print("\n")
print("=" * 70)
print("RULE / ML USAGE")
print("=" * 70)

print(
    f"\nIntent handled by rules: "
    f"{intent_rule_count}/{len(test_df)}"
)

print(
    f"Intent handled by ML: "
    f"{intent_ml_count}/{len(test_df)}"
)

print(
    f"Escalation handled by rules: "
    f"{escalation_rule_count}/{len(test_df)}"
)

print(
    f"Escalation handled by ML: "
    f"{escalation_ml_count}/{len(test_df)}"
)


# --------------------------------------------------
# Timing
# --------------------------------------------------

print("\n")
print("=" * 70)
print("EVALUATION TIMING")
print("=" * 70)

print(
    f"\nIntent evaluation time: "
    f"{intent_time:.4f} seconds"
)

print(
    f"Escalation evaluation time: "
    f"{escalation_time:.4f} seconds"
)

print(
    f"Total evaluation time: "
    f"{intent_time + escalation_time:.4f} seconds"
)


# --------------------------------------------------
# Save held-out results
# --------------------------------------------------

OUTPUT_PATH = (
    "data/processed/"
    "heldout_evaluation_results.csv"
)

test_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nHeld-out results saved to:")
print(OUTPUT_PATH)

print("\n")
print("=" * 70)
print("HELD-OUT EVALUATION COMPLETED")
print("=" * 70)