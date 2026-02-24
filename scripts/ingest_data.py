"""Data ingestion script."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig
from src.data_ingestion.pipeline import DataIngestionPipeline
from src.utils.logger import setup_logger, get_logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main data ingestion pipeline.

    Args:
        cfg: Hydra configuration
    """
    # Setup logger
    setup_logger(cfg)
    logger = get_logger(__name__)

    logger.info("Starting data ingestion pipeline")

    # Initialize pipeline
    pipeline = DataIngestionPipeline(cfg.data)

    # Run pipeline
    output_path = Path(cfg.paths.raw_data) / "raw_data.parquet"
    df = pipeline.run(output_path=str(output_path))

    if not df.empty:
        logger.info("Data ingestion completed successfully")
        logger.info(f"Total records: {len(df)}")
    else:
        logger.warning("No data collected")


if __name__ == "__main__":
    main()
