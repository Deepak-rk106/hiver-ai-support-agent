import joblib
import csv
import os
from datetime import datetime
from dotenv import load_dotenv
from google import genai

from intent_rules import rule_based_intent
from escalation_rules import rule_based_escalation
from retrieval import ResponseRetriever

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Check your .env file."
    )

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = "gemini-3.6-flash"

def generate_gemini_response(
    customer_message,
    intent,
    historical_customer_message,
    historical_support_reply
):
    prompt = f"""
You are an AI customer support agent.

Your job is to answer the customer's message clearly,
helpfully, and safely.

CUSTOMER MESSAGE:
{customer_message}

DETECTED INTENT:
{intent}

HISTORICAL SIMILAR CUSTOMER MESSAGE:
{historical_customer_message}

HISTORICAL BRAND SUPPORT RESPONSE:
{historical_support_reply}

Instructions:
1. Use the historical support response as guidance.
2. Do not copy it blindly.
3. Do not invent policies, prices, refunds, guarantees,
   or account information.
4. Do not mention that you are using historical data.
5. Do not expose private information.
6. Keep the response concise and professional.
7. If the historical response does not provide enough
   information, give a safe general response.
8. Respond directly to the customer.

Return only the customer-support response.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        if response.text:
            return response.text.strip()

        return (
            "Thanks for contacting support. "
            "Please provide more details so we can assist you."
        )

    except Exception as error:
        print(f"Gemini response generation failed: {error}")

        return (
            "Thanks for contacting support. "
            "We're sorry for the inconvenience. "
            "Please provide more details so we can assist you."
        )

def generate_safe_response(customer_message, intent, decision):
    if decision == "human_review":
        return (
            "I'm sorry you're experiencing this issue. "
            "I'm escalating your request to a human support agent "
            "who can investigate it and assist you further."
        )

    responses = {
        "flight_status":
            "You can check your current flight status using your "
            "booking details or the airline's flight-status service.",

        "flight_delay":
            "I'm sorry for the delay. Please check the latest flight "
            "status for the most up-to-date information.",

        "booking":
            "I can help with general booking information. Please provide "
            "the relevant booking details if further assistance is needed.",

        "cancellation":
            "I can help with general cancellation information. "
            "Please check your booking details for the applicable "
            "cancellation options.",

        "refund":
            "Refund eligibility depends on the booking and fare conditions. "
            "Please check your booking details for the applicable refund options.",

        "baggage":
            "Baggage allowances and fees depend on your flight and fare. "
            "Please check your booking details for the applicable baggage policy.",

        "travel_credit":
            "For Travel Bank or travel credit questions, please check "
            "your account balance and the applicable credit terms.",

        "loyalty_rewards":
            "You can earn and use loyalty points according to the "
            "airline's loyalty program rules.",

        "inflight_services":
            "In-flight services can vary by aircraft and route. "
            "Please check the available services for your flight.",

        "boarding_airport":
            "Please check your flight information for the recommended "
            "airport arrival and boarding time.",

        "account_app":
            "Please check the airline app or website for account-related "
            "assistance. Account-specific issues may require human support.",

        "other":
            "Thanks for contacting support. A support agent can assist "
            "with your request."
    }

    return responses.get(
        intent,
        "Thanks for contacting support. Please provide more details "
        "so we can assist you."
    )


# ----------------------------------------
# Configuration
# ----------------------------------------

INTENT_MODEL_PATH = "models/intent_classifier.pkl"
ESCALATION_MODEL_PATH = "models/escalation_classifier.pkl"
DATA_PATH = "data/processed/jetblue_labeling.csv"

INTENT_THRESHOLD = 0.40
SIMILARITY_THRESHOLD = 0.30

LOG_PATH = "data/processed/decision_log.csv"


def log_decision(result):
    file_exists = os.path.exists(LOG_PATH)

    os.makedirs(
        os.path.dirname(LOG_PATH),
        exist_ok=True
    )

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "customer_message",
                "intent",
                "intent_confidence",
                "intent_source",
                "escalation",
                "escalation_confidence",
                "escalation_source",
                "similarity",
                "decision",
                "reason",
                "response"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            result["customer_message"],
            result["intent"],
            result["intent_confidence"],
            result["intent_source"],
            result["escalation"],
            result["escalation_confidence"],
            result["escalation_source"],
            result["similarity"],
            result["decision"],
            result["reason"],
            result["response"]
        ])


# ----------------------------------------
# Load models
# ----------------------------------------

intent_model = joblib.load(INTENT_MODEL_PATH)
escalation_model = joblib.load(ESCALATION_MODEL_PATH)

retriever = ResponseRetriever(DATA_PATH)


# ----------------------------------------
# Support Agent
# ----------------------------------------

def support_agent(customer_message):

    
# ----------------------------------------
# 1.Intent detection
# ----------------------------------------

    rule_intent = rule_based_intent(
        customer_message
    )

    if rule_intent is not None:

        intent = rule_intent

        # Rule-based classification is deterministic
        intent_confidence = 1.0

        intent_source = "rule"

    else:

        intent = intent_model.predict(
            [customer_message]
        )[0]

        intent_probabilities = (
            intent_model.predict_proba(
                [customer_message]
            )[0]
        )

        intent_confidence = max(
            intent_probabilities
        )

        intent_source = "ml"


    # 2. Escalation detection

    rule_escalation = rule_based_escalation(
        customer_message,
        intent
    )

    if rule_escalation is not None:

        escalation = rule_escalation

        # Rule-based escalation is deterministic
        escalation_confidence = 1.0

        escalation_source = "rule"

    else:

        escalation = escalation_model.predict(
            [customer_message]
        )[0]

        escalation_probabilities = (
            escalation_model.predict_proba(
                [customer_message]
            )[0]
        )

        escalation_confidence = max(
            escalation_probabilities
        )

        escalation_source = "ml"


    # 3. Retrieve similar historical examples
    results = retriever.retrieve_response(
        customer_message,
        intent=intent,
        top_k=3
    )

    best_match = results[0]

    similarity = best_match["similarity"]


    # 4. Decide whether the system should answer automatically
    if intent_confidence < INTENT_THRESHOLD:
        decision = "human_review"
        reason = "Intent confidence is below the safety threshold."

    elif escalation == "human":
        decision = "human_review"
        reason = "Conversation requires human assistance."

    elif intent == "other":
        decision = "auto"
        reason = "General feedback can be handled automatically."

    elif similarity < SIMILARITY_THRESHOLD:
        decision = "human_review"
        reason = "No sufficiently similar historical response was found."

    else:
        decision = "auto"
        reason = (
            "Intent and escalation predictions are confident "
            "enough and a similar historical response exists."
        )

# 5. Generate final response

    if decision == "auto":

        safe_response = generate_gemini_response(
            customer_message=customer_message,
            intent=intent,
            historical_customer_message=
                best_match["customer_message"],
            historical_support_reply=
                best_match["support_reply"]
        )

        response_source = "gemini"

    else:

        safe_response = generate_safe_response(
            customer_message,
            intent,
            decision
        )

        response_source = "fallback"

    # 5. Return complete result
    result = {
        "customer_message": customer_message,
        "intent": intent,
        "intent_confidence": float(intent_confidence),
        "intent_source": intent_source,
        "escalation": escalation,
        "escalation_confidence": float(escalation_confidence),
        "escalation_source": escalation_source,
        "similarity": float(similarity),
        "decision": decision,
        "reason": reason,
        "response": safe_response,
        "response_source": response_source,
        "retrieved_customer_message": best_match["customer_message"],
          "retrieved_support_reply": best_match["support_reply"]
    }
    log_decision(result)

    return result


# ----------------------------------------
# Test Agent
# ----------------------------------------

if __name__ == "__main__":

    test_message = (
        "Can you check my flight status? My flight is scheduled to depart at 6 PM today."
    )

    result = support_agent(test_message)

    print("\n" + "=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 60)

    print("\nCustomer Message:")
    print(result["customer_message"])

    print("\nPredicted Intent:")
    print(result["intent"])

    print("\nResponse Source:")
    print(result["response_source"])

    print(
        f"Intent Confidence: "
        f"{result['intent_confidence']:.4f}"
    )
    print(
        f"Intent Source: "
        f"{result['intent_source']}"
    )

    print("\nEscalation:")
    print(result["escalation"])

    print(
        f"Escalation Confidence: "
        f"{result['escalation_confidence']:.4f}"
    )
    print(
        f"Escalation Source: "
        f"{result['escalation_source']}"
    )

    print("\nSimilarity:")
    print(
        f"{result['similarity']:.4f}"
    )

    print("\nAgent Decision:")
    print(result["decision"])

    print("\nDecision Reason:")
    print(result["reason"])

    print("\nSafe Agent Response:")
    print(result["response"])

    print("\nRetrieved Historical Customer:")
    print(
        result["retrieved_customer_message"]
    )

    print("\nRetrieved Historical Support Reply:")
    print(
        result["retrieved_support_reply"]
    )

    print("\n" + "=" * 60)