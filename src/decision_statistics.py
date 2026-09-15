import pandas as pd


LOG_PATH = "data/processed/decision_log.csv"


# --------------------------------------------------
# Load decision log
# --------------------------------------------------

df = pd.read_csv(LOG_PATH)

total = len(df)


# --------------------------------------------------
# Decision statistics
# --------------------------------------------------

auto_count = (
    df["decision"] == "auto"
).sum()

human_count = (
    df["decision"] == "human_review"
).sum()


auto_rate = (
    auto_count / total * 100
    if total > 0
    else 0
)

human_rate = (
    human_count / total * 100
    if total > 0
    else 0
)


# --------------------------------------------------
# Confidence statistics
# --------------------------------------------------

avg_intent_confidence = (
    df["intent_confidence"].mean()
)

avg_escalation_confidence = (
    df["escalation_confidence"].mean()
)

avg_similarity = (
    df["similarity"].mean()
)


# --------------------------------------------------
# Rule / ML statistics
# --------------------------------------------------

intent_rule_count = (
    df["intent_source"] == "rule"
).sum()

intent_ml_count = (
    df["intent_source"] == "ml"
).sum()

escalation_rule_count = (
    df["escalation_source"] == "rule"
).sum()

escalation_ml_count = (
    df["escalation_source"] == "ml"
).sum()


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n")
print("=" * 70)
print("HIVER AI SUPPORT AGENT - DECISION STATISTICS")
print("=" * 70)

print("\nDECISION SUMMARY")
print("-" * 70)

print(
    f"Total conversations: {total}"
)

print(
    f"Automatic decisions: {auto_count}"
)

print(
    f"Human review decisions: {human_count}"
)

print(
    f"Auto-resolution rate: {auto_rate:.2f}%"
)

print(
    f"Human-review rate: {human_rate:.2f}%"
)


print("\nAVERAGE SCORES")
print("-" * 70)

print(
    f"Average intent confidence: "
    f"{avg_intent_confidence:.4f}"
)

print(
    f"Average escalation confidence: "
    f"{avg_escalation_confidence:.4f}"
)

print(
    f"Average retrieval similarity: "
    f"{avg_similarity:.4f}"
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


print("\nDECISION BREAKDOWN")
print("-" * 70)

print(
    df["decision"].value_counts()
)


print("\nINTENT BREAKDOWN")
print("-" * 70)

print(
    df["intent"].value_counts()
)


print("\n")
print("=" * 70)
print("STATISTICS COMPLETED")
print("=" * 70)