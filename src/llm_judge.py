import os
import json
import time
import pandas as pd

from google import genai
from google.genai import errors

from agent import support_agent


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_PATH = "data/processed/heldout_evaluation_results.csv"
OUTPUT_PATH = "data/processed/llm_judge_results.csv"

MODEL_NAME = "gemini-3.5-flash"

# Keep this small because Gemini free tier has request limits.
MAX_EXAMPLES = 10


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set."
    )

client = genai.Client(
    api_key=api_key
)


# --------------------------------------------------
# Gemini Judge
# --------------------------------------------------

def judge_response(
    customer_message,
    agent_response,
    intent
):

    prompt = f"""
You are an expert evaluator for a customer-support AI agent.

The support brand is JetBlue.

Evaluate the AI agent's response based on the customer's
message and the predicted intent.

Customer message:
{customer_message}

Predicted intent:
{intent}

AI agent response:
{agent_response}

Score the response from 1 to 5 on each criterion.

1. Relevance
Does the response directly address the customer's request?

2. Helpfulness
Does the response provide useful assistance?

3. Grounding
Is the response consistent with the customer-support context?
Do not reward invented policies, prices, guarantees, or unsupported facts.

4. Safety
Does the response avoid risky or unsupported claims?
If the issue requires human assistance, the response should not
pretend that the problem has already been resolved.

5. Overall quality
How appropriate is the response for a real customer-support setting?

Return ONLY valid JSON:

{{
    "relevance": 1,
    "helpfulness": 1,
    "grounding": 1,
    "safety": 1,
    "overall": 1,
    "reason": "Short explanation"
}}
"""

    max_retries = 3

    for attempt in range(max_retries):

        try:

            result = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            text = result.text.strip()

            if text.startswith("```"):
                text = text.replace(
                    "```json",
                    ""
                )
                text = text.replace(
                    "```",
                    ""
                )
                text = text.strip()

            try:

                return json.loads(text)

            except json.JSONDecodeError:

                return {
                    "relevance": None,
                    "helpfulness": None,
                    "grounding": None,
                    "safety": None,
                    "overall": None,
                    "reason": (
                        "Gemini returned invalid JSON: "
                        + text
                    )
                }

        except errors.ServerError:

            if attempt < max_retries - 1:

                wait_time = 10 * (attempt + 1)

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                print(
                    "Gemini unavailable after retries."
                )

                return {
                    "relevance": None,
                    "helpfulness": None,
                    "grounding": None,
                    "safety": None,
                    "overall": None,
                    "reason": (
                        "Gemini temporarily unavailable."
                    )
                }

        except errors.ClientError as error:

            print(
                f"Gemini API error: {error}"
            )

            return {
                "relevance": None,
                "helpfulness": None,
                "grounding": None,
                "safety": None,
                "overall": None,
                "reason": str(error)
            }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("GEMINI LLM-AS-A-JUDGE")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH
    )

    # ----------------------------------------------
    # Select only a small evaluation subset
    # ----------------------------------------------

    df = df.head(
        MAX_EXAMPLES
    )

    print(
        f"\nEvaluation set: {len(df)} examples"
    )

    # ----------------------------------------------
    # Load previous results if they exist
    # ----------------------------------------------

    if os.path.exists(OUTPUT_PATH):

        existing_df = pd.read_csv(
            OUTPUT_PATH
        )

        completed_ids = set(
            existing_df["example_id"]
            .astype(str)
        )

        results = existing_df.to_dict(
            orient="records"
        )

        print(
            f"Already completed: "
            f"{len(completed_ids)} examples"
        )

    else:

        completed_ids = set()
        results = []

    # ----------------------------------------------
    # Evaluate examples
    # ----------------------------------------------

    for index, row in df.iterrows():

        example_id = str(
            row["example_id"]
        )

        if example_id in completed_ids:

            print(
                f"\nSkipping example "
                f"{example_id} - already completed."
            )

            continue

        print(
            f"\nEvaluating "
            f"{index + 1}/{len(df)}..."
        )

        customer_message = str(
            row["customer_message"]
        )

        predicted_intent = str(
            row["predicted_intent"]
        )

        # ------------------------------------------
        # Generate response using OUR agent
        # ------------------------------------------

        agent_result = support_agent(
            customer_message
        )

        agent_response = agent_result[
            "response"
        ]

        # ------------------------------------------
        # Gemini evaluates OUR response
        # ------------------------------------------

        scores = judge_response(
            customer_message,
            agent_response,
            predicted_intent
        )

        result = {

            "example_id":
                example_id,

            "customer_message":
                customer_message,

            "predicted_intent":
                predicted_intent,

            "agent_response":
                agent_response,

            "relevance":
                scores.get("relevance"),

            "helpfulness":
                scores.get("helpfulness"),

            "grounding":
                scores.get("grounding"),

            "safety":
                scores.get("safety"),

            "overall":
                scores.get("overall"),

            "judge_reason":
                scores.get("reason")
        }

        # ------------------------------------------
        # Only save genuine Gemini evaluations
        # ------------------------------------------

        if scores.get("overall") is None:

            print(
                f"Example {example_id} was not evaluated "
                f"because Gemini returned an error."
            )

            continue

        results.append(result)

        completed_ids.add(
            example_id
        )

        # ------------------------------------------
        # SAVE IMMEDIATELY
        # ------------------------------------------

        output_df = pd.DataFrame(
            results
        )

        os.makedirs(
            os.path.dirname(OUTPUT_PATH),
            exist_ok=True
        )

        output_df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print(
            f"Saved successful result for example "
            f"{example_id}"
        )
    # ----------------------------------------------
    # Final summary
    # ----------------------------------------------

    if not results:

        print(
            "\nNo evaluations completed."
        )

        return

    output_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("LLM JUDGE COMPLETED")
    print("=" * 70)

    print(
        f"\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"\nCompleted evaluations: "
        f"{len(output_df)}"
    )

    print("\nAverage scores:")

    for column in [
        "relevance",
        "helpfulness",
        "grounding",
        "safety",
        "overall"
    ]:

        average = pd.to_numeric(
            output_df[column],
            errors="coerce"
        ).mean()

        if pd.isna(average):

            print(
                f"{column.capitalize():15}: "
                f"No score"
            )

        else:

            print(
                f"{column.capitalize():15}: "
                f"{average:.2f} / 5"
            )


if __name__ == "__main__":

    main()