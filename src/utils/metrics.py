"""Metrics calculation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from typing import Dict, List, Union, Tuple
import pandas as pd


def calculate_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_proba: Union[List, np.ndarray] = None,
    labels: List[str] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for ROC-AUC)
        labels: Label names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1

    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = precision_macro
    metrics["recall_macro"] = recall_macro
    metrics["f1_macro"] = f1_macro

    # Per-class metrics
    if labels:
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
        )
        for i, label in enumerate(labels):
            metrics[f"precision_{label}"] = precision_per_class[i]
            metrics[f"recall_{label}"] = recall_per_class[i]
            metrics[f"f1_{label}"] = f1_per_class[i]

    # ROC-AUC (if probabilities provided)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) > 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")

    return metrics


def calculate_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: List[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names

    Returns:
        Tuple of (confusion matrix array, formatted dataframe)
    """
    cm = confusion_matrix(y_true, y_pred)

    if labels:
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    else:
        cm_df = pd.DataFrame(cm)

    return cm, cm_df


def get_classification_report(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: List[str] = None,
) -> str:
    """
    Generate classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names

    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)


def calculate_business_metrics(
    predictions: pd.DataFrame,
    actual_returns: pd.Series = None,
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics for financial sentiment.

    Args:
        predictions: DataFrame with sentiment predictions
        actual_returns: Actual market returns (for backtesting)

    Returns:
        Dictionary of business metrics
    """
    metrics = {}

    # Sentiment distribution
    if "sentiment" in predictions.columns:
        sentiment_dist = predictions["sentiment"].value_counts(normalize=True)
        for sentiment, ratio in sentiment_dist.items():
            metrics[f"sentiment_ratio_{sentiment}"] = ratio

    # Confidence metrics
    if "confidence" in predictions.columns:
        metrics["avg_confidence"] = predictions["confidence"].mean()
        metrics["min_confidence"] = predictions["confidence"].min()
        metrics["max_confidence"] = predictions["confidence"].max()

    # Prediction agreement (if multiple models)
    if "agreement_score" in predictions.columns:
        metrics["avg_agreement"] = predictions["agreement_score"].mean()

    # Backtesting metrics (if actual returns provided)
    if actual_returns is not None and "sentiment" in predictions.columns:
        # Calculate strategy returns based on sentiment
        predictions["strategy_return"] = 0
        predictions.loc[predictions["sentiment"] == "positive", "strategy_return"] = (
            actual_returns
        )
        predictions.loc[predictions["sentiment"] == "negative", "strategy_return"] = (
            -actual_returns
        )

        metrics["strategy_return"] = predictions["strategy_return"].sum()
        metrics["sharpe_ratio"] = (
            predictions["strategy_return"].mean()
            / predictions["strategy_return"].std()
            if predictions["strategy_return"].std() > 0
            else 0
        )

    return metrics
