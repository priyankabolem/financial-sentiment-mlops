"""Base class for data ingestion."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class BaseDataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data source.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.source_name = self.__class__.__name__.replace("DataSource", "").lower()

    @abstractmethod
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from the source.

        Returns:
            DataFrame with fetched data
        """
        pass

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate fetched data.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df

        # Add source column if not present
        if "source" not in df.columns:
            df["source"] = self.source_name

        # Add timestamp if not present
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now()

        # Ensure required columns exist
        required_columns = ["text", "source", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean fetched data.

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        # Remove duplicates
        if self.config.get("processing", {}).get("remove_duplicates", True):
            df = df.drop_duplicates(subset=["text"], keep="first")

        # Remove rows with missing text
        df = df.dropna(subset=["text"])

        # Filter by text length
        min_length = self.config.get("processing", {}).get("min_text_length", 10)
        max_length = self.config.get("processing", {}).get("max_text_length", 512)
        df = df[df["text"].str.len().between(min_length, max_length)]

        return df

    def run(self) -> pd.DataFrame:
        """
        Run the complete data ingestion pipeline.

        Returns:
            Processed DataFrame
        """
        if not self.enabled:
            print(f"{self.source_name} is disabled. Skipping...")
            return pd.DataFrame()

        print(f"Fetching data from {self.source_name}...")
        df = self.fetch_data()

        if df.empty:
            print(f"No data fetched from {self.source_name}")
            return df

        print(f"Fetched {len(df)} records from {self.source_name}")

        df = self.validate_data(df)
        df = self.clean_data(df)

        print(f"Processed {len(df)} records from {self.source_name}")

        return df
