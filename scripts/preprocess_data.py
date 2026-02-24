"""Data preprocessing script."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
import pandas as pd
from src.data_preprocessing.text_cleaner import TextCleaner
from src.feature_engineering.features import FeatureEngineer
from src.utils.logger import setup_logger, get_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main data preprocessing pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Setup logger
    setup_logger(cfg)
    logger = get_logger(__name__)

    logger.info("Starting data preprocessing pipeline")

    # Load raw data
    raw_data_path = Path(cfg.paths.raw_data) / "raw_data.parquet"
    if not raw_data_path.exists():
        logger.error(f"Raw data not found at {raw_data_path}")
        logger.error("Please run data ingestion first")
        return

    logger.info(f"Loading raw data from {raw_data_path}")
    df = pd.read_parquet(raw_data_path)
    logger.info(f"Loaded {len(df)} records")

    # Initialize text cleaner
    logger.info("Cleaning text data")
    cleaner = TextCleaner(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_mentions=True,
        remove_hashtags=False,
        remove_numbers=False,
    )

    df = cleaner.clean_dataframe(df, text_column="text")
    logger.info(f"After cleaning: {len(df)} records")

    # Engineer features
    logger.info("Engineering features")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.create_features(df, text_column="text")

    # Save processed data
    output_path = Path(cfg.paths.processed_data) / "processed_data.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    # Save feature names
    feature_columns = [col for col in df.columns if col not in ["text", "timestamp", "source"]]
    features_path = Path(cfg.paths.processed_data) / "feature_names.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(feature_columns))
    logger.info(f"Saved feature names to {features_path}")

    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
