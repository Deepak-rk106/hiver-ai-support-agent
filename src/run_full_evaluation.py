import os
import time
import pandas as pd

from agent import support_agent


DATA_PATH = "data/processed/jetblue_labeling.csv"
OUTPUT_PATH = "data/processed/full_decision_log.csv"


# --------------------------------------------------
# Load all labeled examples
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_message"]
).reset_index(drop=True)

print("\n" + "=" * 70)
print("HIVER AI SUPPORT AGENT - FULL DECISION EVALUATION")
print("=" * 70)

print(f"\nTotal conversations: {len(df)}")


# --------------------------------------------------
# Run complete agent
# --------------------------------------------------

results = []

start_time = time.perf_counter()

for index, message in enumerate(
    df["customer_message"].astype(str),
    start=1
):

    result = support_agent(message)

    results.append({
        "example_id": index,
        "customer_message": message,
        "intent": result["intent"],
        "intent_confidence":
            result["intent_confidence"],
        "intent_source":
            result["intent_source"],
        "escalation":
            result["escalation"],
        "escalation_confidence":
            result["escalation_confidence"],
        "escalation_source":
            result["escalation_source"],
        "similarity":
            result["similarity"],
        "decision":
            result["decision"],
        "reason":
            result["reason"],
        "response":
            result["response"]
    })

    if index % 25 == 0:
        print(
            f"Processed {index}/{len(df)}"
        )


total_time = (
    time.perf_counter() - start_time
)


# --------------------------------------------------
# Create result DataFrame
# --------------------------------------------------

results_df = pd.DataFrame(results)


# --------------------------------------------------
# Decision statistics
# --------------------------------------------------

total = len(results_df)

auto_count = (
    results_df["decision"] == "auto"
).sum()

human_count = (
    results_df["decision"] == "human_review"
).sum()


auto_rate = (
    auto_count / total * 100
)

human_rate = (
    human_count / total * 100
)


# --------------------------------------------------
# Rule / ML usage
# --------------------------------------------------

intent_rule_count = (
    results_df["intent_source"] == "rule"
).sum()

intent_ml_count = (
    results_df["intent_source"] == "ml"
).sum()

escalation_rule_count = (
    results_df["escalation_source"] == "rule"
).sum()

escalation_ml_count = (
    results_df["escalation_source"] == "ml"
).sum()


# --------------------------------------------------
# Timing
# --------------------------------------------------

average_time = (
    total_time / total
)

# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DECISION SUMMARY")
print("=" * 70)

print(
    f"\nTotal conversations: {total}"
)

print(
    f"Automatic decisions: {auto_count}"
)

print(
    f"Human review decisions: {human_count}"
)

print(
    f"Auto-resolution rate: "
    f"{auto_rate:.2f}%"
)

print(
    f"Human-review rate: "
    f"{human_rate:.2f}%"
)


print("\nAVERAGE SCORES")
print("-" * 70)

print(
    f"Average intent confidence: "
    f"{results_df['intent_confidence'].mean():.4f}"
)

print(
    f"Average escalation confidence: "
    f"{results_df['escalation_confidence'].mean():.4f}"
)

print(
    f"Average retrieval similarity: "
    f"{results_df['similarity'].mean():.4f}"
)


print("\nRULE / ML USAGE")
print("-" * 70)

print(
    f"Intent handled by rules: "
    f"{intent_rule_count}"
)

print(
    f"Intent handled by ML: "
    f"{intent_ml_count}"
)

print(
    f"Escalation handled by rules: "
    f"{escalation_rule_count}"
)

print(
    f"Escalation handled by ML: "
    f"{escalation_ml_count}"
)


print("\nTIMING")
print("-" * 70)

print(
    f"Total processing time: "
    f"{total_time:.4f} seconds"
)

print(
    f"Average time per conversation: "
    f"{average_time:.4f} seconds"
)


# --------------------------------------------------
# Decision breakdown
# --------------------------------------------------

print("\nDECISION BREAKDOWN")
print("-" * 70)

print(
    results_df["decision"].value_counts()
)


# --------------------------------------------------
# Intent breakdown
# --------------------------------------------------

print("\nINTENT BREAKDOWN")
print("-" * 70)

print(
    results_df["intent"].value_counts()
)


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)

print(
    f"Full decision log saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 70)