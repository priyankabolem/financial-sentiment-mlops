"""Configuration utilities."""

import os
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from dotenv import load_dotenv


def load_env_variables(env_file: str = ".env") -> None:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"Warning: {env_file} not found. Using system environment variables.")


def get_config(config_path: str = "configs", config_name: str = "config") -> DictConfig:
    """
    Load Hydra configuration.

    Args:
        config_path: Path to config directory
        config_name: Name of config file (without .yaml)

    Returns:
        Hydra configuration object
    """
    with hydra.initialize(version_base=None, config_path=f"../{config_path}"):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object
        save_path: Path to save the config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_path)


def validate_config(config: DictConfig) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration object

    Returns:
        True if valid, raises exception otherwise
    """
    # Validate required fields
    required_fields = ["project", "paths", "mlflow"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate paths exist
    for path_key in ["data_dir", "models_dir", "logs_dir"]:
        path = Path(config.paths[path_key])
        path.mkdir(parents=True, exist_ok=True)

    return True
