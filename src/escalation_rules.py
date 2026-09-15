def rule_based_escalation(message, intent):
    text = message.lower().strip()

    # Account security
    if any(word in text for word in [
        "compromised",
        "hacked",
        "account stolen",
        "account security"
    ]):
        return "human"

    # Lost / missing baggage
    if (
        "lost" in text and ("bag" in text or "baggage" in text)
    ) or (
        "missing" in text and ("bag" in text or "baggage" in text)
    ) or (
        "bag" in text and "did not arrive" in text
    ) or (
        "baggage" in text and "did not arrive" in text
    ):
        return "human"

    # Serious travel disruption
    if any(phrase in text for phrase in [
        "missed connection",
        "miss my connection",
        "stranded",
        "deplaned"
    ]):
        return "human"

    # Severe delays
    if any(phrase in text for phrase in [
        "delayed for hours",
        "delayed by five hours",
        "delayed for several hours",
        "delayed for five hours",
        "multiple delays"
    ]):
        return "human"

    # In-flight equipment problems
    if (
        ("tv" in text or "screen" in text) and
        ("not working" in text or "isn't working" in text)
    ) or (
        ("wifi" in text or "wi-fi" in text) and
        ("not working" in text or "isn't working" in text)
    ):
        return "human"

        # Positive feedback does not require human escalation
    if (
        "enjoyed my flight" in text
        or "enjoyed the flight" in text
        or "great flight" in text
        or "excellent crew" in text
        or "great crew" in text
        or "wonderful crew" in text
    ):
        return "auto"

    return None