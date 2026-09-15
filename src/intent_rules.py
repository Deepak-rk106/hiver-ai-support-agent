import re


def rule_based_intent(message):

    text = message.lower().strip()


    # ----------------------------------------
    # Account / application
    # ----------------------------------------

    if (
        "account" in text
        and (
            "compromised" in text
            or "hacked" in text
            or "stolen" in text
            or "security" in text
        )
    ):
        return "account_app"


    if (
        ("check in" in text or "check-in" in text)
        and (
            "website" in text
            or "app" in text
            or "online" in text
        )
    ):
        return "account_app"


    # ----------------------------------------
    # Flight status
    # ----------------------------------------

    if (
        "flight status" in text
        or "status of my flight" in text
        or "where is my flight" in text
    ):
        return "flight_status"


    # ----------------------------------------
    # Cancellation
    # ----------------------------------------

    if (
        "cancel my flight" in text
        or "cancel the flight" in text
        or "cancel flight" in text
        or "want to cancel" in text
    ):
        return "cancellation"


    # ----------------------------------------
    # Refund
    # ----------------------------------------

    if (
        "refund" in text
        or "get my money back" in text
        or "money back" in text
    ):
        return "refund"


    # ----------------------------------------
    # Travel credit
    # ----------------------------------------

    if (
        "travel bank" in text
        or "travel credit" in text
        or "credit balance" in text
    ):
        return "travel_credit"


    # ----------------------------------------
    # Loyalty / rewards
    # ----------------------------------------

    if (
        "trueblue" in text
        or "trueblue points" in text
        or "loyalty points" in text
        or "mosaic" in text
    ):
        return "loyalty_rewards"


    # ----------------------------------------
    # Baggage
    # ----------------------------------------

    if (
        "lost baggage" in text
        or "lost bag" in text
        or "missing baggage" in text
        or "missing bag" in text
        or "baggage fee" in text
        or "baggage allowance" in text
        or "baggage" in text
        or "bag" in text
    ):
        return "baggage"


    # ----------------------------------------
    # Boarding / airport
    # ----------------------------------------

    if (
        "boarding" in text
        or "boarding pass" in text
        or "airport arrival" in text
        or "arrive at the airport" in text
    ):
        return "boarding_airport"

    # ----------------------------------------
    # Inflight services
    # ----------------------------------------

    if (
        "tv" in text
        or "fly-fi" in text
        or "wifi" in text
        or "inflight entertainment" in text
    ):
        return "inflight_services"


    # ----------------------------------------
    # Flight delay
    # ----------------------------------------

    if (
        "flight delayed" in text
        or "flight is delayed" in text
        or "delay" in text
        or "delayed" in text
    ):
        return "flight_delay"


    # ----------------------------------------
    # Boarding / airport
    # ----------------------------------------

    if (
        "boarding" in text
        or "boarding pass" in text
        or "airport" in text
        or "arrive at the airport" in text
        or "airport arrival" in text
    ):
        return "boarding_airport"

        # Positive feedback / general comments
    if (
        "enjoyed my flight" in text
        or "enjoyed the flight" in text
        or "great flight" in text
        or "excellent crew" in text
        or "great crew" in text
        or "wonderful crew" in text
        or "thank you" in text
        or "thanks for the great service" in text
    ):
        return "other"
    # ----------------------------------------
    # No rule matched
    # ----------------------------------------

    return None