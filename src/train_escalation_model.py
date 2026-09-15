import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load labeled data
DATA_PATH = "data/processed/jetblue_labeling.csv"

df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} labeled examples")


# 2. Prepare input and target
X = df["customer_message"]
y = df["escalation"]


# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# 4. Create TF-IDF + Logistic Regression pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# 5. Train the model
print("\nTraining escalation classifier...")

model.fit(X_train, y_train)

print("Training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("ESCALATION CLASSIFIER RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# 8. Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 9. Save trained model
MODEL_PATH = "models/escalation_classifier.pkl"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")