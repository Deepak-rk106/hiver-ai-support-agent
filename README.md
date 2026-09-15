# Hiver AI Support Agent

A safety-aware AI customer support agent that classifies customer requests, determines whether they can be handled automatically or require human assistance, retrieves relevant historical support examples, and generates a controlled response.

The project uses a hybrid architecture combining rule-based logic, machine learning, TF-IDF retrieval, confidence thresholds, and human-review fallback.

---

## 1. Project Overview

The goal of this project is to build an AI support agent capable of:

- Understanding customer support messages
- Classifying the customer's primary intent
- Determining whether the request can be automated or requires human assistance
- Retrieving relevant historical support conversations
- Avoiding unsafe or unsupported responses
- Routing uncertain or high-risk conversations to human support
- Maintaining a decision log for explainability and auditing

The system was developed using JetBlue customer-support conversations from the Customer Support on Twitter dataset.

---

## 2. Key Features

### Intent Classification

The system identifies one of 12 support intents:

1. flight_delay
2. flight_status
3. booking
4. cancellation
5. refund
6. baggage
7. travel_credit
8. account_app
9. loyalty_rewards
10. inflight_services
11. boarding_airport
12. other

### Escalation Classification

Each request is classified as:

- `auto` — can potentially be handled automatically
- `human` — requires human assistance

### Hybrid Rule + ML Architecture

The system first checks deterministic rules for obvious cases.

If no rule matches, the ML classifier is used as a fallback.

This approach is especially useful for high-risk or easily identifiable situations such as:

- Compromised accounts
- Lost baggage
- Severe delays
- In-flight equipment problems
- Serious travel disruptions

### Safety Controls

The system includes:

- Intent confidence threshold
- Retrieval similarity threshold
- Human-review fallback
- Deterministic escalation rules
- Safe response generation
- Historical response isolation

Historical customer responses are used as supporting context and are not blindly copied to the customer.

---

## 3. Architecture

```text
                 Customer Message
                        |
                        v
              +--------------------+
              |  Intent Rules      |
              +---------+----------+
                        |
                 Rule matched?
                  /          \
                Yes           No
                 |             |
                 v             v
             Intent       ML Classifier
                 \             /
                  \           /
                   v         v
                Predicted Intent
                        |
                        v
              +--------------------+
              | Escalation Rules   |
              +---------+----------+
                        |
                 Rule matched?
                  /          \
                Yes           No
                 |             |
                 v             v
            Escalation     ML Classifier
                 \             /
                  \           /
                   v         v
              Auto / Human
                        |
                        v
              Historical Retrieval
                        |
                        v
           Confidence + Similarity Gate
                        |
               +--------+--------+
               |                 |
               v                 v
             AUTO          HUMAN REVIEW
               |                 |
               v                 v
       Safe Response       Human Support
               |
               v
            Customer