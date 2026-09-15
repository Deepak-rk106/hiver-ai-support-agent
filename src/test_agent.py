from agent import support_agent


test_messages = [
    "My flight is delayed by five hours",
    "Where can I check my flight status?",
    "I lost my baggage at the airport",
    "How much baggage can I take?",
    "I want to cancel my flight",
    "How can I get a refund?",
    "My Travel Bank is not working",
    "How do I earn TrueBlue points?",
    "I cannot check in using the website",
    "The TV on my seat is not working",
    "What time should I arrive at the airport?",
    "My JetBlue account has been compromised",
    "I really enjoyed my flight and the crew was excellent"
]


print("\n" + "=" * 70)
print("HIVER AI SUPPORT AGENT - MULTI TEST")
print("=" * 70)


for i, message in enumerate(test_messages, start=1):

    result = support_agent(message)

    print("\n" + "-" * 70)
    print(f"TEST {i}")
    print("-" * 70)

    print(f"Customer: {message}")
    print(f"Intent: {result['intent']}")
    print(
        f"Intent Confidence: "
        f"{result['intent_confidence']:.4f}"
    )

    print(f"Escalation: {result['escalation']}")
    print(
        f"Escalation Confidence: "
        f"{result['escalation_confidence']:.4f}"
    )

    print(
        f"Similarity: "
        f"{result['similarity']:.4f}"
    )

    print(f"Decision: {result['decision']}")

    print(f"Reason: {result['reason']}")

    print(
        f"Retrieved Reply: "
        f"{result['retrieved_support_reply']}"
    )
    print(f"Safe Agent Response: {result['response']}")


print("\n" + "=" * 70)
print("TESTING COMPLETED")
print("=" * 70) 