"""Data drift detection using Evidently."""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime


class DataDriftDetector:
    """Detect data drift in production data."""

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector.

        Args:
            reference_data: Reference dataset (training data)
        """
        self.reference_data = reference_data

    def detect_drift(
        self, current_data: pd.DataFrame, save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data: Current production data
            save_report: Whether to save the report

        Returns:
            Drift detection results
        """
        # Create data drift report
        data_drift_report = Report(metrics=[DataDriftPreset()])

        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )

        # Get results
        results = data_drift_report.as_dict()

        # Extract key metrics
        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": results["metrics"][0]["result"]["dataset_drift"],
            "number_of_drifted_columns": results["metrics"][0]["result"][
                "number_of_drifted_columns"
            ],
            "share_of_drifted_columns": results["metrics"][0]["result"][
                "share_of_drifted_columns"
            ],
        }

        # Save report
        if save_report:
            report_dir = Path("reports/drift")
            report_dir.mkdir(parents=True, exist_ok=True)

            # Save HTML report
            html_path = report_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            data_drift_report.save_html(str(html_path))

            # Save JSON summary
            json_path = report_dir / f"drift_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, "w") as f:
                json.dump(drift_summary, f, indent=2)

            print(f"Drift report saved to {html_path}")

        return drift_summary

    def test_drift(
        self, current_data: pd.DataFrame, max_drifted_columns: int = 3
    ) -> Dict[str, Any]:
        """
        Run drift tests.

        Args:
            current_data: Current production data
            max_drifted_columns: Maximum allowed drifted columns

        Returns:
            Test results
        """
        # Create test suite
        tests = TestSuite(
            tests=[TestNumberOfDriftedColumns(lt=max_drifted_columns)],
        )

        tests.run(reference_data=self.reference_data, current_data=current_data)

        # Get results
        results = tests.as_dict()

        test_summary = {
            "timestamp": datetime.now().isoformat(),
            "passed": results["summary"]["all_passed"],
            "failed_tests": results["summary"]["failed_tests"],
        }

        return test_summary


class ModelPerformanceMonitor:
    """Monitor model performance in production."""

    def __init__(self):
        """Initialize performance monitor."""
        self.predictions = []
        self.metrics_history = []

    def log_prediction(
        self,
        text: str,
        prediction: str,
        confidence: float,
        latency: float,
        timestamp: datetime = None,
    ) -> None:
        """
        Log a prediction.

        Args:
            text: Input text
            prediction: Model prediction
            confidence: Prediction confidence
            latency: Prediction latency in ms
            timestamp: Timestamp of prediction
        """
        self.predictions.append(
            {
                "text": text,
                "prediction": prediction,
                "confidence": confidence,
                "latency": latency,
                "timestamp": timestamp or datetime.now(),
            }
        )

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary of metrics
        """
        if not self.predictions:
            return {}

        df = pd.DataFrame(self.predictions)

        metrics = {
            "avg_confidence": df["confidence"].mean(),
            "min_confidence": df["confidence"].min(),
            "max_confidence": df["confidence"].max(),
            "avg_latency": df["latency"].mean(),
            "p95_latency": df["latency"].quantile(0.95),
            "p99_latency": df["latency"].quantile(0.99),
            "total_predictions": len(df),
        }

        # Sentiment distribution
        sentiment_dist = df["prediction"].value_counts(normalize=True)
        for sentiment, ratio in sentiment_dist.items():
            metrics[f"ratio_{sentiment}"] = ratio

        return metrics

    def save_metrics(self, output_path: str) -> None:
        """
        Save metrics to file.

        Args:
            output_path: Path to save metrics
        """
        metrics = self.calculate_metrics()
        metrics["timestamp"] = datetime.now().isoformat()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {output_path}")
