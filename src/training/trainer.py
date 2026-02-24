"""Model training with MLflow integration."""

import torch
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import json


class SentimentDataset(Dataset):
    """Dataset for sentiment analysis."""

    def __init__(self, encodings, labels=None):
        """
        Initialize dataset.

        Args:
            encodings: Tokenized encodings
            labels: Optional labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


class SentimentTrainer:
    """Trainer for sentiment analysis model with MLflow tracking."""

    def __init__(self, model, config: Dict[str, Any], mlflow_config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            model: Model to train
            config: Training configuration
            mlflow_config: MLflow configuration
        """
        self.model = model
        self.config = config
        self.mlflow_config = mlflow_config
        self.device = model.device

        # Training parameters
        self.num_epochs = config.get("num_epochs", 5)
        self.batch_size = config.get("batch_size", 16)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Early stopping
        self.early_stopping_config = config.get("early_stopping", {})
        self.early_stopping_enabled = self.early_stopping_config.get("enabled", False)
        self.patience = self.early_stopping_config.get("patience", 3)
        self.best_metric = None
        self.patience_counter = 0

        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

    def _setup_optimizer(self, num_training_steps: int) -> None:
        """Set up optimizer and learning rate scheduler."""
        optimizer_config = self.config.get("optimizer", {})

        # AdamW optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=optimizer_config.get("lr", 2e-5),
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            eps=optimizer_config.get("eps", 1e-8),
        )

        # Learning rate scheduler
        scheduler_config = self.config.get("scheduler", {})
        num_warmup_steps = scheduler_config.get("num_warmup_steps", 500)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model.model(**batch)
            loss = outputs.loss

            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item()})

        return {"train_loss": total_loss / len(train_loader)}

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "eval_loss": total_loss / len(eval_loader),
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1_score": f1_score(all_labels, all_predictions, average="weighted"),
            "precision": precision_score(all_labels, all_predictions, average="weighted"),
            "recall": recall_score(all_labels, all_predictions, average="weighted"),
        }

        return metrics

    def train(
        self,
        train_encodings,
        train_labels,
        val_encodings,
        val_labels,
        experiment_name: str = None,
    ) -> Dict[str, Any]:
        """
        Train the model with MLflow tracking.

        Args:
            train_encodings: Training encodings
            train_labels: Training labels
            val_encodings: Validation encodings
            val_labels: Validation labels
            experiment_name: MLflow experiment name

        Returns:
            Training history and metrics
        """
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_config.get("tracking_uri"))
        experiment_name = experiment_name or self.mlflow_config.get("experiment_name")
        mlflow.set_experiment(experiment_name)

        # Create datasets
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Calculate training steps
        num_training_steps = (
            len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        )
        self._setup_optimizer(num_training_steps)

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": self.model.model_name,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.config.get("optimizer", {}).get("lr"),
                    "num_training_samples": len(train_dataset),
                    "num_val_samples": len(val_dataset),
                }
            )

            # Training loop
            history = {"train_loss": [], "val_loss": [], "val_f1": []}
            best_val_f1 = 0

            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

                # Train
                train_metrics = self.train_epoch(train_loader)
                print(f"Train Loss: {train_metrics['train_loss']:.4f}")

                # Evaluate
                val_metrics = self.evaluate(val_loader)
                print(f"Val Loss: {val_metrics['eval_loss']:.4f}")
                print(f"Val F1: {val_metrics['f1_score']:.4f}")
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

                # Log metrics
                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["train_loss"],
                        "val_loss": val_metrics["eval_loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1_score"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                    },
                    step=epoch,
                )

                # Update history
                history["train_loss"].append(train_metrics["train_loss"])
                history["val_loss"].append(val_metrics["eval_loss"])
                history["val_f1"].append(val_metrics["f1_score"])

                # Save best model
                if val_metrics["f1_score"] > best_val_f1:
                    best_val_f1 = val_metrics["f1_score"]
                    best_model_path = Path("models") / "best_model"
                    best_model_path.mkdir(parents=True, exist_ok=True)
                    self.model.save_model(str(best_model_path))
                    print(f"Saved best model with F1: {best_val_f1:.4f}")

                # Early stopping
                if self.early_stopping_enabled:
                    if self._check_early_stopping(val_metrics["f1_score"]):
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            # Log final model
            mlflow.pytorch.log_model(self.model.model, "model")

            # Log best metrics
            mlflow.log_metric("best_val_f1", best_val_f1)

            print(f"\nTraining completed. Best Val F1: {best_val_f1:.4f}")

            return {"history": history, "best_val_f1": best_val_f1}

    def _check_early_stopping(self, metric_value: float) -> bool:
        """Check if early stopping criteria is met."""
        if self.best_metric is None or metric_value > self.best_metric:
            self.best_metric = metric_value
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            return False
