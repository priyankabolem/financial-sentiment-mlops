"""Feature engineering for financial sentiment analysis."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import re
from textblob import TextBlob


class FeatureEngineer:
    """Engineer features for sentiment analysis."""

    def __init__(self):
        """Initialize feature engineer."""
        self.financial_keywords = {
            "positive": [
                "profit",
                "gain",
                "growth",
                "surge",
                "rally",
                "bullish",
                "beat",
                "outperform",
                "upgrade",
                "strong",
                "increase",
                "rise",
                "high",
                "boom",
            ],
            "negative": [
                "loss",
                "decline",
                "crash",
                "bearish",
                "miss",
                "underperform",
                "downgrade",
                "weak",
                "decrease",
                "fall",
                "low",
                "risk",
                "fear",
            ],
        }

    def extract_text_features(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Extract basic text features.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with additional text features
        """
        df = df.copy()

        # Length features
        df["text_length"] = df[text_column].str.len()
        df["word_count"] = df[text_column].str.split().str.len()
        df["avg_word_length"] = df[text_column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )

        # Character features
        df["uppercase_ratio"] = df[text_column].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        df["digit_ratio"] = df[text_column].apply(
            lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0
        )
        df["punctuation_ratio"] = df[text_column].apply(
            lambda x: sum(1 for c in x if c in ".,!?;:") / len(x) if len(x) > 0 else 0
        )

        # Sentence features
        df["sentence_count"] = df[text_column].apply(
            lambda x: len(re.findall(r"[.!?]+", x))
        )

        return df

    def extract_sentiment_features(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Extract sentiment-related features using TextBlob.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()

        # TextBlob sentiment
        sentiments = df[text_column].apply(lambda x: TextBlob(x).sentiment)
        df["textblob_polarity"] = sentiments.apply(lambda x: x.polarity)
        df["textblob_subjectivity"] = sentiments.apply(lambda x: x.subjectivity)

        return df

    def extract_financial_features(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Extract financial domain-specific features.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with financial features
        """
        df = df.copy()

        # Ticker extraction
        df["tickers"] = df[text_column].apply(self._extract_tickers)
        df["ticker_count"] = df["tickers"].apply(len)
        df["has_ticker"] = df["ticker_count"] > 0

        # Financial keyword counts
        df["positive_keyword_count"] = df[text_column].apply(
            lambda x: self._count_keywords(x, self.financial_keywords["positive"])
        )
        df["negative_keyword_count"] = df[text_column].apply(
            lambda x: self._count_keywords(x, self.financial_keywords["negative"])
        )

        # Keyword ratio
        df["keyword_ratio"] = (
            df["positive_keyword_count"] - df["negative_keyword_count"]
        ) / (df["positive_keyword_count"] + df["negative_keyword_count"] + 1)

        # Number and currency detection
        df["has_numbers"] = df[text_column].str.contains(r"\d+", regex=True)
        df["has_currency"] = df[text_column].str.contains(
            r"\$|USD|EUR|GBP|bitcoin|btc", regex=True, case=False
        )
        df["has_percentage"] = df[text_column].str.contains(r"%|\bpercent\b", regex=True)

        return df

    def extract_source_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract source-specific features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with source features
        """
        df = df.copy()

        # Source encoding
        if "source" in df.columns:
            df["source_encoded"] = pd.Categorical(df["source"]).codes

        # Engagement features (for social media)
        if "score" in df.columns:
            df["score_log"] = np.log1p(df["score"])

        if "upvote_ratio" in df.columns:
            df["upvote_ratio_binned"] = pd.cut(
                df["upvote_ratio"], bins=[0, 0.5, 0.7, 0.9, 1.0], labels=[0, 1, 2, 3]
            )

        if "num_comments" in df.columns:
            df["num_comments_log"] = np.log1p(df["num_comments"])

        return df

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with temporal features
        """
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["month"] = df["timestamp"].dt.month
            df["is_market_hours"] = df["hour"].between(9, 16).astype(int)

        return df

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text."""
        tickers = re.findall(r"\$[A-Z]{1,5}\b", text)
        return [ticker.replace("$", "") for ticker in tickers]

    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text."""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)

    def create_features(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Create all features.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with all engineered features
        """
        print("Extracting text features...")
        df = self.extract_text_features(df, text_column)

        print("Extracting sentiment features...")
        df = self.extract_sentiment_features(df, text_column)

        print("Extracting financial features...")
        df = self.extract_financial_features(df, text_column)

        print("Extracting source features...")
        df = self.extract_source_features(df)

        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)

        print(f"Created {len(df.columns)} total features")

        return df
