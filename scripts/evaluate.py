"""Model evaluation script."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import pandas as pd
from torch.utils.data import DataLoader
from src.models.sentiment_model import FinancialSentimentModel
from src.training.trainer import SentimentDataset
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger, get_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main evaluation pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Setup logger
    setup_logger(cfg)
    logger = get_logger(__name__)

    logger.info("Starting model evaluation")

    # Load test data
    data_path = Path(cfg.paths.processed_data) / "processed_data.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # For demonstration - create test split
    # In production, you'd have a separate test set
    test_df = df.sample(frac=0.15, random_state=cfg.seed)
    test_texts = test_df["text"].tolist()

    # Create synthetic labels for demonstration
    import numpy as np
    test_labels = np.random.randint(0, 3, len(test_texts))

    logger.info(f"Test samples: {len(test_texts)}")

    # Load model
    logger.info("Loading model")
    model = FinancialSentimentModel(cfg.model)

    # Try to load best model
    best_model_path = Path("models/best_model")
    if best_model_path.exists():
        model.load_model(str(best_model_path))
        logger.info("Loaded best trained model")
    else:
        model.load_model()
        logger.info("Loaded pretrained model")

    # Tokenize test data
    logger.info("Tokenizing test data")
    test_encodings = model.tokenize(test_texts)

    # Create test dataset
    test_dataset = SentimentDataset(test_encodings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size)

    # Initialize evaluator
    evaluator = ModelEvaluator(model, model.id2label)

    # Generate comprehensive evaluation report
    logger.info("Generating evaluation report")
    output_dir = Path("evaluation_results")
    evaluator.generate_evaluation_report(test_loader, output_dir=str(output_dir))

    logger.info(f"Evaluation completed. Results saved to {output_dir}/")

    # Print summary
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        import json
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        logger.info("\n" + "="*70)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*70)
        report = metrics["classification_report"]
        logger.info(f"Accuracy: {report['accuracy']:.4f}")
        logger.info(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        logger.info(f"Weighted Precision: {report['weighted avg']['precision']:.4f}")
        logger.info(f"Weighted Recall: {report['weighted avg']['recall']:.4f}")

        for label in model.id2label.values():
            if label in report:
                logger.info(f"\n{label.upper()}:")
                logger.info(f"  F1-Score: {report[label]['f1-score']:.4f}")
                logger.info(f"  Precision: {report[label]['precision']:.4f}")
                logger.info(f"  Recall: {report[label]['recall']:.4f}")


if __name__ == "__main__":
    main()
