"""Unit tests for sentiment model."""

import pytest
import torch
from src.models.sentiment_model import FinancialSentimentModel


class TestFinancialSentimentModel:
    """Test FinancialSentimentModel class."""

    @pytest.fixture
    def model_config(self):
        """Model configuration fixture."""
        return {
            "base_model": "ProsusAI/finbert",
            "architecture": {"num_labels": 3, "dropout": 0.1},
            "labels": {0: "negative", 1: "neutral", 2: "positive"},
            "tokenizer": {
                "max_length": 128,
                "padding": "max_length",
                "truncation": True,
            },
            "inference": {"device": "cpu"},
        }

    @pytest.fixture
    def model(self, model_config):
        """Model fixture."""
        return FinancialSentimentModel(model_config)

    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model is not None
        assert model.num_labels == 3
        assert model.device.type == "cpu"

    def test_tokenize(self, model):
        """Test tokenization."""
        model.load_model()
        texts = ["This is a positive text", "This is negative"]
        encodings = model.tokenize(texts)

        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        assert encodings["input_ids"].shape[0] == 2

    def test_predict_single(self, model):
        """Test single prediction."""
        model.load_model()
        result = model.predict_single("Great earnings, stock going up!")

        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    def test_get_model_info(self, model):
        """Test getting model info."""
        model.load_model()
        info = model.get_model_info()

        assert "model_name" in info
        assert "num_labels" in info
        assert "device" in info
