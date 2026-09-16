import pandas as pd
import os

from agent import support_agent


INPUT_PATH = "data/processed/heldout_evaluation_results.csv"
OUTPUT_PATH = "data/processed/human_rating_template.csv"

MAX_EXAMPLES = 10


def main():

    print("=" * 70)
    print("CREATING HUMAN RATING TEMPLATE")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    # Use the exact same 10 examples that the LLM judge will evaluate
    df = df.head(MAX_EXAMPLES)

    results = []

    for index, row in df.iterrows():

        customer_message = str(
            row["customer_message"]
        )

        agent_result = support_agent(
            customer_message
        )

        results.append({
            "example_id": row["example_id"],
            "customer_message": customer_message,
            "predicted_intent": row["predicted_intent"],
            "agent_response": agent_result["response"],

            # Human evaluator fills these columns
            "human_relevance": "",
            "human_helpfulness": "",
            "human_grounding": "",
            "human_safety": "",
            "human_overall": "",
            "human_reason": ""
        })

        print(
            f"Prepared example "
            f"{index + 1}/{len(df)}"
        )

    output_df = pd.DataFrame(results)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("HUMAN RATING TEMPLATE CREATED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print(
        f"\nExamples: {len(output_df)}"
    )


if __name__ == "__main__":
    main()