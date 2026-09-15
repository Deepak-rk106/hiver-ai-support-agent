import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ----------------------------------------
# 1. Load labeled data
# ----------------------------------------

DATA_PATH = "data/processed/jetblue_labeling.csv"

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} labeled examples")


# Remove incomplete rows
df = df.dropna(
    subset=["customer_message", "intent"]
).reset_index(drop=True)


X = df["customer_message"].astype(str)
y = df["intent"].astype(str)


# ----------------------------------------
# 2. Train/Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# ----------------------------------------
# 3. Word + Character TF-IDF
# ----------------------------------------

word_tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)


char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=5000,
    sublinear_tf=True
)


features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])


# ----------------------------------------
# 4. Create classifier
# ----------------------------------------

model = Pipeline([
    ("features", features),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0
        )
    )
])


# ----------------------------------------
# 5. Train
# ----------------------------------------

print("\nTraining improved intent classifier...")

model.fit(X_train, y_train)

print("Training completed!")


# ----------------------------------------
# 6. Predictions
# ----------------------------------------

y_pred = model.predict(X_test)


# ----------------------------------------
# 7. Evaluation
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("IMPROVED INTENT CLASSIFIER RESULTS")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ----------------------------------------
# 8. Save model
# ----------------------------------------

MODEL_PATH = "models/intent_classifier.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)