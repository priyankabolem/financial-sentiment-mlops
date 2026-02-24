"""Training script for sentiment analysis model."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.sentiment_model import FinancialSentimentModel
from src.training.trainer import SentimentTrainer
from src.utils.logger import setup_logger, get_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Setup logger
    setup_logger(cfg)
    logger = get_logger(__name__)

    logger.info("Starting training pipeline")
    logger.info(f"Config: {cfg}")

    # Load processed data
    data_path = Path(cfg.paths.processed_data) / "processed_data.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        logger.error("Please run data ingestion and preprocessing first")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # For this example, we'll create synthetic labels
    # In production, you'd have labeled data
    logger.warning("Creating synthetic labels for demonstration")
    import numpy as np
    df["label"] = np.random.randint(0, 3, len(df))  # 0: negative, 1: neutral, 2: positive

    # Split data
    logger.info("Splitting data")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=cfg.data.splits.validation + cfg.data.splits.test,
        random_state=cfg.seed,
        stratify=df["label"] if cfg.data.splits.stratify else None,
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Initialize model
    logger.info("Initializing model")
    model = FinancialSentimentModel(cfg.model)
    model.load_model()

    # Tokenize data
    logger.info("Tokenizing data")
    train_encodings = model.tokenize(train_texts)
    val_encodings = model.tokenize(val_texts)

    # Initialize trainer
    logger.info("Initializing trainer")
    trainer = SentimentTrainer(model, cfg.training, cfg.mlflow)

    # Train model
    logger.info("Starting training")
    results = trainer.train(
        train_encodings=train_encodings,
        train_labels=train_labels,
        val_encodings=val_encodings,
        val_labels=val_labels,
    )

    logger.info("Training completed")
    logger.info(f"Best validation F1: {results['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
