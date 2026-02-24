"""Model evaluation and performance analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


class ModelEvaluator:
    """Comprehensive model evaluation for sentiment analysis."""

    def __init__(self, model, id2label: Dict[int, str]):
        """
        Initialize evaluator.

        Args:
            model: Trained sentiment model
            id2label: Label mapping dictionary
        """
        self.model = model
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

    def evaluate(
        self,
        eval_loader,
        save_path: str = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            eval_loader: DataLoader for evaluation data
            save_path: Path to save evaluation results

        Returns:
            Dictionary containing all evaluation metrics
        """
        # Get predictions
        y_true, y_pred, y_proba = self._get_predictions(eval_loader)

        # Calculate metrics
        metrics = {
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=list(self.id2label.values()),
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "roc_auc": self._calculate_roc_auc(y_true, y_proba),
            "per_class_metrics": self._calculate_per_class_metrics(y_true, y_pred),
            "error_analysis": self._analyze_errors(y_true, y_pred),
            "confidence_analysis": self._analyze_confidence(y_proba, y_true, y_pred),
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        # Save results
        if save_path:
            self._save_results(metrics, save_path)

        return metrics

    def _get_predictions(self, eval_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on evaluation data."""
        import torch

        self.model.model.eval()
        all_preds = []
        all_labels = []
        all_proba = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model.model(**batch)

                logits = outputs.logits
                proba = torch.softmax(logits, dim=-1)
                preds = torch.argmax(proba, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                all_proba.extend(proba.cpu().numpy())

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_proba),
        )

    def _calculate_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate ROC-AUC scores."""
        try:
            # One-vs-rest ROC AUC
            roc_auc_ovr = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )

            # Per-class ROC AUC
            roc_auc_per_class = {}
            for i, label in self.id2label.items():
                y_true_binary = (y_true == i).astype(int)
                roc_auc_per_class[label] = roc_auc_score(y_true_binary, y_proba[:, i])

            return {
                "weighted": roc_auc_ovr,
                "per_class": roc_auc_per_class,
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate detailed per-class metrics."""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        per_class = {}
        for i, label in self.id2label.items():
            per_class[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }

        return per_class

    def _analyze_errors(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = y_true != y_pred
        error_count = errors.sum()
        total = len(y_true)

        # Error matrix (misclassification patterns)
        error_matrix = {}
        for true_label in self.id2label.values():
            error_matrix[true_label] = {}
            for pred_label in self.id2label.values():
                mask = (y_true == self.label2id[true_label]) & (
                    y_pred == self.label2id[pred_label]
                )
                error_matrix[true_label][pred_label] = int(mask.sum())

        return {
            "total_errors": int(error_count),
            "error_rate": float(error_count / total),
            "accuracy": float((total - error_count) / total),
            "error_matrix": error_matrix,
        }

    def _analyze_confidence(
        self, y_proba: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction confidence."""
        max_proba = y_proba.max(axis=1)
        correct = y_true == y_pred

        return {
            "avg_confidence_correct": float(max_proba[correct].mean()),
            "avg_confidence_incorrect": float(max_proba[~correct].mean()),
            "low_confidence_threshold": 0.6,
            "low_confidence_predictions": int((max_proba < 0.6).sum()),
            "high_confidence_threshold": 0.9,
            "high_confidence_predictions": int((max_proba > 0.9).sum()),
        }

    def _save_results(self, metrics: Dict[str, Any], save_path: str):
        """Save evaluation results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Evaluation results saved to {save_path}")

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(self.id2label.values()),
            yticklabels=list(self.id2label.values()),
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_evaluation_report(
        self, eval_loader, output_dir: str = "evaluation_results"
    ):
        """
        Generate comprehensive evaluation report with visualizations.

        Args:
            eval_loader: DataLoader for evaluation
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        metrics = self.evaluate(
            eval_loader,
            save_path=str(output_dir / "metrics.json"),
        )

        # Generate plots
        y_true, y_pred, y_proba = self._get_predictions(eval_loader)

        self.plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=str(output_dir / "confusion_matrix.png"),
        )

        # Create summary report
        self._create_summary_report(metrics, output_dir / "summary_report.txt")

        print(f"\nEvaluation report generated in {output_dir}/")

    def _create_summary_report(self, metrics: Dict[str, Any], save_path: Path):
        """Create human-readable summary report."""
        with open(save_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Evaluation Date: {metrics['evaluation_timestamp']}\n\n")

            # Overall metrics
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 70 + "\n")
            report = metrics["classification_report"]
            f.write(f"Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Precision: {report['weighted avg']['precision']:.4f}\n")
            f.write(f"Weighted Recall: {report['weighted avg']['recall']:.4f}\n\n")

            # Per-class performance
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 70 + "\n")
            for label, metrics_dict in metrics["per_class_metrics"].items():
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  Precision: {metrics_dict['precision']:.4f}\n")
                f.write(f"  Recall: {metrics_dict['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics_dict['f1_score']:.4f}\n")
                f.write(f"  Support: {metrics_dict['support']}\n")

            # Error analysis
            f.write("\nERROR ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            error_analysis = metrics["error_analysis"]
            f.write(f"Total Errors: {error_analysis['total_errors']}\n")
            f.write(f"Error Rate: {error_analysis['error_rate']:.4f}\n\n")

            # Confidence analysis
            f.write("CONFIDENCE ANALYSIS:\n")
            f.write("-" * 70 + "\n")
            conf_analysis = metrics["confidence_analysis"]
            f.write(
                f"Avg Confidence (Correct): {conf_analysis['avg_confidence_correct']:.4f}\n"
            )
            f.write(
                f"Avg Confidence (Incorrect): {conf_analysis['avg_confidence_incorrect']:.4f}\n"
            )
            f.write(
                f"Low Confidence Predictions: {conf_analysis['low_confidence_predictions']}\n"
            )
            f.write(
                f"High Confidence Predictions: {conf_analysis['high_confidence_predictions']}\n"
            )

            f.write("\n" + "=" * 70 + "\n")

        print(f"Summary report saved to {save_path}")
