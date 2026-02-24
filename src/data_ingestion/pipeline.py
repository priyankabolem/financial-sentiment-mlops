"""Data ingestion pipeline orchestrator."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from .news_api import NewsAPIDataSource
from .alpha_vantage import AlphaVantageDataSource
from .reddit import RedditDataSource


class DataIngestionPipeline:
    """Orchestrate data ingestion from multiple sources."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data ingestion pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_sources = self._initialize_sources()

    def _initialize_sources(self) -> List[Any]:
        """
        Initialize all data sources based on configuration.

        Returns:
            List of data source instances
        """
        sources = []

        # News API
        if "news_api" in self.config and self.config["news_api"].get("enabled", False):
            sources.append(NewsAPIDataSource(self.config["news_api"]))

        # Alpha Vantage
        if "alpha_vantage" in self.config and self.config["alpha_vantage"].get(
            "enabled", False
        ):
            sources.append(AlphaVantageDataSource(self.config["alpha_vantage"]))

        # Reddit
        if "reddit" in self.config and self.config["reddit"].get("enabled", False):
            sources.append(RedditDataSource(self.config["reddit"]))

        return sources

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Fetch data from all enabled sources.

        Returns:
            Combined DataFrame with data from all sources
        """
        all_data = []

        for source in self.data_sources:
            try:
                df = source.run()
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"Error running {source.source_name}: {e}")
                continue

        if not all_data:
            print("No data fetched from any source")
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Add ingestion timestamp
        combined_df["ingestion_timestamp"] = datetime.now()

        return combined_df

    def save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save fetched data to disk.

        Args:
            df: DataFrame to save
            output_path: Path to save the data
        """
        if df.empty:
            print("No data to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet for efficiency
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")

        # Also save metadata
        metadata = {
            "num_records": len(df),
            "sources": df["source"].value_counts().to_dict(),
            "date_range": {
                "min": str(df["timestamp"].min()),
                "max": str(df["timestamp"].max()),
            },
            "ingestion_timestamp": str(datetime.now()),
        }

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    def run(self, output_path: str = None) -> pd.DataFrame:
        """
        Run the complete data ingestion pipeline.

        Args:
            output_path: Optional path to save the data

        Returns:
            Combined DataFrame with all fetched data
        """
        print("Starting data ingestion pipeline...")
        df = self.fetch_all_data()

        if not df.empty:
            print(f"\nTotal records fetched: {len(df)}")
            print(f"Sources: {df['source'].value_counts().to_dict()}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            if output_path:
                self.save_data(df, output_path)

        return df
