"""Sentiment analysis model using transformers."""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from typing import Dict, List, Any, Tuple
import numpy as np


class FinancialSentimentModel:
    """Financial sentiment analysis model using FinBERT."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model_name = config.get("base_model", "ProsusAI/finbert")
        self.num_labels = config.get("architecture", {}).get("num_labels", 3)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("inference", {}).get("device") == "cuda" else "cpu"
        )

        # Label mapping
        self.id2label = config.get("labels", {0: "negative", 1: "neutral", 2: "positive"})
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None

    def load_model(self, model_path: str = None) -> None:
        """
        Load model and tokenizer.

        Args:
            model_path: Path to saved model, or use pretrained
        """
        print(f"Loading model: {model_path or self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path or self.model_name,
            cache_dir=self.config.get("pretrained", {}).get("cache_dir", None),
        )

        # Load model
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            config_model = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )

            # Update dropout rates
            arch_config = self.config.get("architecture", {})
            if "dropout" in arch_config:
                config_model.hidden_dropout_prob = arch_config["dropout"]
                config_model.attention_probs_dropout_prob = arch_config.get(
                    "attention_probs_dropout_prob", arch_config["dropout"]
                )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config_model,
                cache_dir=self.config.get("pretrained", {}).get("cache_dir", None),
            )

        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}")

    def tokenize(
        self, texts: List[str], max_length: int = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts.

        Args:
            texts: List of texts to tokenize
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments

        Returns:
            Dictionary of tokenized inputs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")

        tokenizer_config = self.config.get("tokenizer", {})
        max_length = max_length or tokenizer_config.get("max_length", 512)

        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=tokenizer_config.get("padding", "max_length"),
            truncation=tokenizer_config.get("truncation", True),
            return_attention_mask=tokenizer_config.get("return_attention_mask", True),
            return_tensors="pt",
            **kwargs,
        )

        return encodings

    def predict(
        self, texts: List[str], return_probabilities: bool = True
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        """
        Predict sentiment for texts.

        Args:
            texts: List of texts to predict
            return_probabilities: Whether to return probability scores

        Returns:
            Tuple of (predicted labels, confidence scores, probability distributions)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()

        # Tokenize
        encodings = self.tokenize(texts)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits

        # Get predictions
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predicted_classes = np.argmax(probabilities, axis=-1)
        confidence_scores = np.max(probabilities, axis=-1)

        # Convert to labels
        predicted_labels = [self.id2label[int(pred)] for pred in predicted_classes]

        if return_probabilities:
            return predicted_labels, confidence_scores.tolist(), probabilities.tolist()
        else:
            return predicted_labels, confidence_scores.tolist(), None

    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.

        Args:
            text: Text to predict

        Returns:
            Dictionary with prediction results
        """
        labels, confidences, probabilities = self.predict([text])

        return {
            "text": text,
            "sentiment": labels[0],
            "confidence": confidences[0],
            "probabilities": {
                label: prob
                for label, prob in zip(self.id2label.values(), probabilities[0])
            },
        }

    def save_model(self, save_path: str) -> None:
        """
        Save model and tokenizer.

        Args:
            save_path: Path to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "labels": self.id2label,
            "device": str(self.device),
            "parameters": (
                sum(p.numel() for p in self.model.parameters())
                if self.model
                else 0
            ),
            "trainable_parameters": (
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if self.model
                else 0
            ),
        }
