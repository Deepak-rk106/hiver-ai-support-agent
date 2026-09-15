from agent import support_agent


def print_result(result):

    print("\n" + "=" * 60)
    print("           HIVER AI SUPPORT AGENT")
    print("=" * 60)

    print("\nCustomer Message:")
    print("> " + result["customer_message"])

    print("\nIntent:")
    print(f"  {result['intent']}")

    print("\nIntent Confidence:")
    print(f"  {result['intent_confidence']:.4f}")

    print("\nIntent Source:")
    print(f"  {result['intent_source']}")

    print("\nEscalation:")
    print(f"  {result['escalation']}")

    print("\nEscalation Confidence:")
    print(f"  {result['escalation_confidence']:.4f}")

    print("\nEscalation Source:")
    print(f"  {result['escalation_source']}")

    print("\nRetrieval Similarity:")
    print(f"  {result['similarity']:.4f}")

    print("\nDecision:")
    print(f"  {result['decision'].upper()}")

    print("\nReason:")
    print(f"  {result['reason']}")

    print("\nAgent Response:")
    print("  " + result["response"])

    print("\n" + "=" * 60)


def main():

    print("=" * 60)
    print("           HIVER AI SUPPORT AGENT")
    print("=" * 60)

    print("\nType a customer message.")
    print("Type 'exit' to close the demo.\n")

    while True:

        customer_message = input("Customer > ").strip()

        if customer_message.lower() == "exit":
            print("\nDemo ended.")
            break

        if not customer_message:
            print("Please enter a customer message.\n")
            continue

        result = support_agent(customer_message)

        print_result(result)


if __name__ == "__main__":
    main()