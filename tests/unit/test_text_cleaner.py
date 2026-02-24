"""Unit tests for text cleaner."""

import pytest
from src.data_preprocessing.text_cleaner import TextCleaner, FinancialTextProcessor


class TestTextCleaner:
    """Test TextCleaner class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "This is a TEST message! @user #hashtag http://example.com"
        cleaned = self.cleaner.clean_text(text)

        assert "@user" not in cleaned
        assert "http" not in cleaned
        assert cleaned.islower()

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check this out: https://www.example.com/page"
        cleaned = self.cleaner.clean_text(text)

        assert "http" not in cleaned
        assert "www" not in cleaned

    def test_expand_contractions(self):
        """Test contraction expansion."""
        text = "I can't believe it won't work"
        cleaned = self.cleaner.clean_text(text)

        assert "cannot" in cleaned or "not" in cleaned

    def test_empty_text(self):
        """Test empty text handling."""
        text = ""
        cleaned = self.cleaner.clean_text(text)

        assert cleaned == ""

    def test_clean_batch(self):
        """Test batch cleaning."""
        texts = ["Text 1 @user", "Text 2 #hashtag", "Text 3 http://url.com"]
        cleaned = self.cleaner.clean_batch(texts)

        assert len(cleaned) == 3
        assert all("@" not in text for text in cleaned)


class TestFinancialTextProcessor:
    """Test FinancialTextProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FinancialTextProcessor()

    def test_extract_tickers(self):
        """Test ticker extraction."""
        text = "Buying $AAPL and $MSFT stocks today"
        tickers = self.processor.extract_tickers(text)

        assert "AAPL" in tickers or "$AAPL" in tickers
        assert "MSFT" in tickers or "$MSFT" in tickers

    def test_extract_sentiment_words(self):
        """Test sentiment word extraction."""
        text = "Stock is going up with great profit gains"
        result = self.processor.extract_sentiment_words(text)

        assert result["positive_words"] > 0

    def test_normalize_financial_terms(self):
        """Test financial term normalization."""
        text = "This is a bull market, not bear"
        normalized = self.processor.normalize_financial_terms(text)

        assert "bullish" in normalized or "bull" in normalized
