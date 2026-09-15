import pandas as pd


INPUT_PATH = (
    "data/processed/heldout_evaluation_results.csv"
)

OUTPUT_PATH = (
    "data/processed/failure_analysis.csv"
)


# --------------------------------------------------
# Load held-out evaluation results
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)


# --------------------------------------------------
# Identify errors
# --------------------------------------------------

df["intent_correct"] = (
    df["intent"] ==
    df["predicted_intent"]
)

df["escalation_correct"] = (
    df["escalation"] ==
    df["predicted_escalation"]
)


failures = df[
    (~df["intent_correct"]) |
    (~df["escalation_correct"])
].copy()


# --------------------------------------------------
# Add failure type
# --------------------------------------------------

def get_failure_type(row):

    if (
        not row["intent_correct"]
        and not row["escalation_correct"]
    ):
        return "intent_and_escalation_error"

    if not row["intent_correct"]:
        return "intent_error"

    if not row["escalation_correct"]:
        return "escalation_error"

    return "none"


failures["failure_type"] = (
    failures.apply(
        get_failure_type,
        axis=1
    )
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

total = len(df)
failure_count = len(failures)

intent_errors = (
    (~df["intent_correct"]).sum()
)

escalation_errors = (
    (~df["escalation_correct"]).sum()
)


print("\n")
print("=" * 70)
print("HIVER AI SUPPORT AGENT - FAILURE ANALYSIS")
print("=" * 70)

print(
    f"\nHeld-out examples: {total}"
)

print(
    f"Total cases with errors: "
    f"{failure_count}"
)

print(
    f"Intent errors: "
    f"{intent_errors}"
)

print(
    f"Escalation errors: "
    f"{escalation_errors}"
)


# --------------------------------------------------
# Show individual failures
# --------------------------------------------------

print("\n")
print("=" * 70)
print("FAILURE CASES")
print("=" * 70)


if len(failures) == 0:

    print("\nNo failures found.")

else:

    for index, row in failures.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Customer: "
            f"{row['customer_message']}"
        )

        print(
            f"Expected intent: "
            f"{row['intent']}"
        )

        print(
            f"Predicted intent: "
            f"{row['predicted_intent']}"
        )

        print(
            f"Expected escalation: "
            f"{row['escalation']}"
        )

        print(
            f"Predicted escalation: "
            f"{row['predicted_escalation']}"
        )

        print(
            f"Failure type: "
            f"{row['failure_type']}"
        )


# --------------------------------------------------
# Save failures
# --------------------------------------------------

failures.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n")
print("=" * 70)

print(
    f"Failure analysis saved to:\n"
    f"{OUTPUT_PATH}"
)

print("=" * 70)