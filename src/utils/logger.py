"""Logging configuration."""

import sys
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig


def setup_logger(config: DictConfig) -> None:
    """
    Set up logging configuration.

    Args:
        config: Hydra configuration object
    """
    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        format=config.logging.format,
        level=config.logging.level,
        colorize=True,
    )

    # Add file handler
    log_dir = Path(config.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app_{time}.log",
        format=config.logging.format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression="zip",
    )

    logger.info("Logger initialized successfully")


def get_logger(name: str):
    """Get a logger instance."""
    return logger.bind(name=name)
