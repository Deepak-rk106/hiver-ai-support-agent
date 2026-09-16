# Hiver SDE Intern — AI Customer Support Agent

An AI-powered customer support agent built for the Hiver SDE Intern Take-Home Assignment.

The system is designed to process historical customer-support conversations from Twitter for a selected brand, classify incoming customer messages, retrieve similar historical conversations, decide whether a request can be handled automatically or should be escalated to a human, and generate a grounded customer-support response.

---

# 1. Problem Framing

## Objective

The goal of this project is to build a practical AI customer-support agent using real-world customer-support conversations.

The system should be able to:

1. Classify an incoming customer message into a small set of support intents.
2. Retrieve historically similar customer-support conversations.
3. Use historical support responses as guidance.
4. Decide whether the request can be safely handled automatically or requires human support.
5. Generate a concise and safe response for automatically handled requests.
6. Log the decision and supporting information for later evaluation.

The project focuses on building a system that is not only capable of generating responses, but also knows when it should **not** answer automatically.

---

# 2. What Does "Good" Mean?

A good customer-support agent should not be judged only by whether its generated response sounds natural.

For this project, a good response should satisfy several requirements:

### 2.1 Correct intent

The system should correctly identify what the customer is asking about.

Examples:

- Flight status
- Flight delay
- Booking
- Cancellation
- Refund
- Baggage
- Travel credit
- Loyalty/rewards
- In-flight services
- Boarding/airport
- Account/application issues
- Other

### 2.2 Appropriate escalation

The system should avoid automatically answering cases where:

- Intent confidence is too low.
- The escalation model identifies a need for human support.
- There is insufficient historical evidence.
- The request falls outside the safe scope of automated handling.

### 2.3 Useful historical evidence

The retrieved historical conversation should provide useful context for the response.

Historical responses are treated as guidance rather than being copied blindly.

### 2.4 Safe response generation

The generated response should:

- Address the customer's request.
- Be concise and professional.
- Avoid inventing policies.
- Avoid inventing prices or refund guarantees.
- Avoid inventing account-specific information.
- Avoid exposing private information.
- Avoid making unsupported claims.

### 2.5 Human fallback

When the system is not sufficiently confident, it should prefer human review instead of generating an unsupported answer.

---

# 3. Dataset

## Primary Dataset

The project uses the **Customer Support on Twitter** dataset.

The dataset contains real customer-support conversations between customers and brands on Twitter.

The original dataset is large and noisy, containing approximately millions of tweets across many brands.

Because the assignment explicitly allows and encourages working with a subsample, this project works with a manageable subset rather than processing the complete dataset.

## Selected Brand

The selected brand for this implementation is:

**JetBlue**

The processed project data is stored in:

```text
data/processed/jetblue_labeling.csv

---

# 4. Data Processing

The raw Twitter support data is not directly suitable for an AI support agent.

The project therefore uses a processed representation containing customer-support examples that can be used for:

Intent classification
Historical response retrieval
Evaluation
Response generation

The processing workflow is conceptually:
Raw Customer Support Dataset
            |
            v
     Brand Selection
            |
            v
      Data Cleaning
            |
            v
   Intent/Response Preparation
            |
            v
   Processed JetBlue Dataset
            |
            +-------------------+
            |                   |
            v                   v
     ML Classification      Retrieval
     

5. System Architecture

The complete support-agent pipeline is:
                    Customer Message
                           |
                           v
                +----------------------+
                | Rule-Based Intent    |
                | Detection            |
                +----------+-----------+
                           |
                    No rule match?
                           |
                           v
                +----------------------+
                | ML Intent Classifier |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Escalation Detection |
                | Rules + ML           |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Historical Response  |
                | Retrieval             |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | Safety / Decision    |
                | Layer                |
                +----------+-----------+
                           |
              +------------+------------+
              |                         |
              v                         v
          AUTO HANDLE              HUMAN REVIEW
              |                         |
              v                         v
       Gemini Response             Safe Fallback
              |                         |
              +------------+------------+
                           |
                           v
                   Final Response
                           |
                           v
                    Decision Log


6. Intent Classification

The system uses a two-stage intent classification approach.

6.1 Rule-Based Intent Detection

The first layer uses deterministic rules.

The rule-based system is useful for obvious support queries where keywords and patterns provide a strong signal.

For example:
Customer:
My flight has been delayed for several hours

Intent:
flight_delay

Intent Confidence:
1.0000

Intent Source:
rule
Rule-based classification is deterministic, so the system assigns a confidence of 1.0 when a rule matches.

7. Machine Learning Intent Classification

If the rule-based system does not identify an intent, the request is passed to the trained ML intent classifier.

The classifier predicts:

Intent
+
Prediction probabilities
+
Maximum confidence

The current model is loaded from:

models/intent_classifier.pkl

The system uses:

INTENT_THRESHOLD = 0.40

If the maximum predicted intent confidence is below this threshold, the system sends the request for human review.

Example:

Customer:
What time does my flight depart today?

Intent:
flight_status

Intent Confidence:
0.1576

Intent Source:
ml

Because:

0.1576 < 0.40

the system selects:

Decision:
human_review

This is intentional safety behavior.

8. Escalation Detection

After intent detection, the system determines whether the request can be automatically handled.

Escalation uses:

Rule-based escalation
ML escalation classifier

The ML model is stored at:

models/escalation_classifier.pkl

The escalation output is:

auto

or:

human

For example:

Escalation:
human

Escalation Source:
rule

results in:

Decision:
human_review
9. Safety Decision Layer

The final decision is not based on the LLM alone.

The system uses multiple safety checks.

Decision logic
Case 1 — Low intent confidence
Intent confidence < 0.40
        |
        v
human_review
Case 2 — Escalation model/rule requests human support
Escalation = human
        |
        v
human_review
Case 3 — Intent is "other"

The system allows general feedback to be handled automatically.

Case 4 — Weak historical similarity

The current similarity threshold is:

SIMILARITY_THRESHOLD = 0.30

If the retrieved example falls below this threshold:

Similarity < 0.30
        |
        v
human_review
Case 5 — Safe automatic handling

If the intent is sufficiently confident, escalation is auto, and historical similarity is sufficient:

Decision:
auto

10. Gemini Response Generation

Gemini is used only after the system decides that a request is safe to handle automatically.

The current model configured for response generation is:

gemini-3.6-flash

The Gemini API key is loaded from the environment:

GEMINI_API_KEY

The key is stored in .env and is intentionally not committed to Git.

The response-generation prompt provides Gemini with:

Customer message
Detected intent
Historical customer message
Historical support response

Gemini is instructed to:

Use historical responses as guidance.
Not copy them blindly.
Avoid inventing policies.
Avoid inventing prices.
Avoid inventing refunds or guarantees.
Avoid inventing account information.
Avoid exposing private information.
Keep responses concise and professional.
Respond directly to the customer.
12. Why Gemini Is Not Used for Human Review

A major design decision is that Gemini is not called when the decision layer selects:

human_review

Instead, the system produces a safe fallback response.

For example:

I'm sorry you're experiencing this issue. I'm escalating your request
to a human support agent who can investigate it and assist you further.

This creates a clear separation:

AUTO
  |
  +--> Gemini

HUMAN_REVIEW
  |
  +--> Safe fallback

This reduces the chance of generating unsupported answers for uncertain cases.

13. Response Source

The agent records the source of the final response.

Possible values are:

gemini
fallback

For an automatically handled request:

Decision:
auto

Response Source:
gemini

For a human-review request:

Decision:
human_review

Response Source:
fallback

This distinction is useful during evaluation because it makes it possible to determine whether a response came from the LLM or the fallback mechanism.

14. Example End-to-End Run

Example customer message:

Can you check my flight status? My flight is scheduled to depart at 6 PM today.

The current system produced:

Predicted Intent:
flight_status

Intent Confidence:
1.0000

Intent Source:
rule

Escalation:
auto

Escalation Confidence:
0.5611

Escalation Source:
ml

Similarity:
0.3383

Agent Decision:
auto

Response Source:
gemini

Generated response:

I would be happy to check your flight status. Could you please share
your flight number or confirmation code so I can look up the details for you?

This demonstrates the complete path:

Customer message
      |
      v
Rule-based intent
      |
      v
ML escalation
      |
      v
Historical retrieval
      |
      v
Safety decision
      |
      v
AUTO
      |
      v
Gemini
      |
      v
Final response

16. Project Structure

The repository is organized approximately as follows:

Hiver-AI-Support-Agent/
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── jetblue_labeling.csv
│       ├── human_rating_template.csv
│       ├── llm_judge_results.csv
│       └── decision_log.csv
│
├── models/
│   ├── intent_classifier.pkl
│   └── escalation_classifier.pkl
│
├── src/
│   ├── agent.py
│   ├── intent_rules.py
│   ├── escalation_rules.py
│   ├── retrieval.py
│   ├── llm_judge.py
│   └── create_human_rating_template.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

The .env file is not included in the repository.

17. How to Set Up the Project
Step 1 — Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Hiver-AI-Support-Agent

Replace <YOUR_GITHUB_REPOSITORY_URL> with the repository URL.

Step 2 — Create a virtual environment

Windows PowerShell:

python -m venv venv

Activate it:

venv\Scripts\activate

You should see:

(venv)

at the beginning of your terminal prompt.

Step 3 — Install dependencies
pip install -r requirements.txt
Step 4 — Configure Gemini API

Create a .env file in the project root:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY

Do not commit this file.

The project .gitignore contains:

.env

so the API key remains outside the repository.

18. Run the AI Support Agent

Run:

python src\agent.py

Run the AI Support Agent

Run:

python src\agent.py