"""Text cleaning and preprocessing utilities."""

import re
import unicodedata
from typing import List, Optional
import pandas as pd


class TextCleaner:
    """Clean and preprocess text data for sentiment analysis."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True,
        expand_contractions: bool = True,
    ):
        """
        Initialize text cleaner.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            remove_numbers: Remove numbers
            remove_extra_spaces: Remove extra whitespace
            expand_contractions: Expand contractions (e.g., don't -> do not)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces
        self.expand_contractions = expand_contractions

        # Contractions dictionary
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am",
        }

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)

        # Remove URLs
        if self.remove_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            text = re.sub(r"www\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", "", text)

        # Remove emails
        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)

        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # Remove hashtags
        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)

        # Expand contractions
        if self.expand_contractions:
            for contraction, expanded in self.contractions.items():
                text = re.sub(
                    contraction, expanded, text, flags=re.IGNORECASE
                )

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?;:\-']", "", text)

        # Remove extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r"\s+", " ", text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "text"
    ) -> pd.DataFrame:
        """
        Clean text in a DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of the text column

        Returns:
            DataFrame with cleaned text
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)

        # Remove empty texts
        df = df[df[text_column].str.len() > 0]

        return df


class FinancialTextProcessor:
    """Specialized processor for financial text."""

    # Financial terms to preserve
    FINANCIAL_TERMS = {
        "bull": "bullish",
        "bear": "bearish",
        "yolo": "high_risk_investment",
        "dd": "due_diligence",
        "ath": "all_time_high",
        "atl": "all_time_low",
        "hodl": "hold",
        "fud": "fear_uncertainty_doubt",
        "fomo": "fear_of_missing_out",
        "roi": "return_on_investment",
        "ipo": "initial_public_offering",
        "eps": "earnings_per_share",
        "p/e": "price_to_earnings",
    }

    def __init__(self):
        """Initialize financial text processor."""
        pass

    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text.

        Args:
            text: Input text

        Returns:
            List of extracted tickers
        """
        # Match patterns like $AAPL, AAPL, etc.
        tickers = re.findall(r"\$[A-Z]{1,5}\b|\b[A-Z]{2,5}\b", text)
        return list(set(tickers))

    def extract_sentiment_words(self, text: str) -> dict:
        """
        Extract positive and negative words from text.

        Args:
            text: Input text

        Returns:
            Dictionary with positive and negative word counts
        """
        positive_words = [
            "bull",
            "bullish",
            "up",
            "gain",
            "profit",
            "surge",
            "rally",
            "moon",
            "rocket",
        ]
        negative_words = [
            "bear",
            "bearish",
            "down",
            "loss",
            "crash",
            "drop",
            "fall",
            "dump",
        ]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        return {"positive_words": pos_count, "negative_words": neg_count}

    def normalize_financial_terms(self, text: str) -> str:
        """
        Normalize financial slang and abbreviations.

        Args:
            text: Input text

        Returns:
            Text with normalized financial terms
        """
        text_lower = text.lower()
        for term, normalized in self.FINANCIAL_TERMS.items():
            text_lower = re.sub(
                rf"\b{term}\b", normalized, text_lower
            )
        return text_lower
