import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResponseRetriever:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=[
                "customer_message",
                "support_reply",
                "intent"
            ]
        ).reset_index(drop=True)

        print(f"Loaded {len(self.df)} response examples")

    def _keyword_score(self, query, message):

        query_words = set(query.lower().split())
        message_words = set(message.lower().split())

        if not query_words:
            return 0.0

        common_words = query_words.intersection(message_words)

        return len(common_words) / len(query_words)

    def retrieve_response(
        self,
        customer_message,
        intent=None,
        top_k=3
    ):

        # ----------------------------------------
        # 1. Filter by intent
        # ----------------------------------------

        if intent is not None:

            filtered_df = self.df[
                self.df["intent"] == intent
            ].copy()

            if len(filtered_df) == 0:
                filtered_df = self.df.copy()

        else:

            filtered_df = self.df.copy()

        # ----------------------------------------
        # 2. Prepare messages
        # ----------------------------------------

        messages = (
            filtered_df["customer_message"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------
        # 3. TF-IDF similarity
        # ----------------------------------------

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )

        message_vectors = vectorizer.fit_transform(messages)

        query_vector = vectorizer.transform(
            [customer_message]
        )

        tfidf_scores = cosine_similarity(
            query_vector,
            message_vectors
        )[0]

        # ----------------------------------------
        # 4. Keyword similarity
        # ----------------------------------------

        keyword_scores = np.array([
            self._keyword_score(
                customer_message,
                message
            )
            for message in messages
        ])

        # ----------------------------------------
        # 5. Combined score
        # ----------------------------------------

        combined_scores = (
            0.7 * tfidf_scores
            + 0.3 * keyword_scores
        )

        # ----------------------------------------
        # 6. Get best matches
        # ----------------------------------------

        top_indices = np.argsort(
            combined_scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            row = filtered_df.iloc[index]

            results.append({

                "customer_message":
                    row["customer_message"],

                "support_reply":
                    row["support_reply"],

                "intent":
                    row["intent"],

                "similarity":
                    float(combined_scores[index])

            })

        return results


if __name__ == "__main__":

    DATA_PATH = (
        "data/processed/jetblue_labeling.csv"
    )

    retriever = ResponseRetriever(
        DATA_PATH
    )

    test_cases = [

        (
            "My flight is delayed for several hours",
            "flight_delay"
        ),

        (
            "I want to cancel my flight",
            "cancellation"
        ),

        (
            "My Travel Bank is not working",
            "travel_credit"
        ),

        (
            "I lost my baggage",
            "baggage"
        ),

        (
            "How do I earn TrueBlue points?",
            "loyalty_rewards"
        )

    ]

    for message, intent in test_cases:

        print("\n" + "=" * 60)

        print(
            f"Customer: {message}"
        )

        print(
            f"Intent filter: {intent}"
        )

        results = retriever.retrieve_response(
            message,
            intent=intent,
            top_k=3
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n--- Match {i} ---"
            )

            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )

            print(
                f"Customer: "
                f"{result['customer_message']}"
            )

            print(
                f"Support: "
                f"{result['support_reply']}"
            )