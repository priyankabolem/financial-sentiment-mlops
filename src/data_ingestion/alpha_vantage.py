"""Alpha Vantage data source for market news and sentiment."""

import os
import requests
from typing import Dict, Any
import pandas as pd
from .base import BaseDataSource


class AlphaVantageDataSource(BaseDataSource):
    """Fetch financial news and sentiment from Alpha Vantage."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpha Vantage data source.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            print(
                "Warning: ALPHA_VANTAGE_API_KEY not found. Alpha Vantage will be disabled."
            )
            self.enabled = False
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch news and sentiment from Alpha Vantage.

        Returns:
            DataFrame with news and sentiment data
        """
        if not self.api_key:
            return pd.DataFrame()

        articles = []

        try:
            # Get configuration
            tickers = self.config.get("tickers", "").split(",")
            topics = self.config.get("topics", "")

            # Fetch news sentiment
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": 200,
            }

            # Add tickers if specified
            if tickers and tickers[0]:
                params["tickers"] = ",".join(tickers)

            # Add topics if specified
            if topics:
                params["topics"] = topics

            response = requests.get(self.base_url, params=params)
            data = response.json()

            if "feed" in data:
                for item in data["feed"]:
                    # Extract ticker sentiments
                    ticker_sentiments = []
                    for ticker_data in item.get("ticker_sentiment", []):
                        ticker_sentiments.append(
                            {
                                "ticker": ticker_data.get("ticker"),
                                "relevance_score": float(
                                    ticker_data.get("relevance_score", 0)
                                ),
                                "ticker_sentiment_score": float(
                                    ticker_data.get("ticker_sentiment_score", 0)
                                ),
                                "ticker_sentiment_label": ticker_data.get(
                                    "ticker_sentiment_label"
                                ),
                            }
                        )

                    articles.append(
                        {
                            "text": f"{item.get('title', '')} {item.get('summary', '')}",
                            "title": item.get("title", ""),
                            "summary": item.get("summary", ""),
                            "url": item.get("url", ""),
                            "source": "alpha_vantage",
                            "source_name": item.get("source", ""),
                            "timestamp": pd.to_datetime(item.get("time_published")),
                            "overall_sentiment_score": float(
                                item.get("overall_sentiment_score", 0)
                            ),
                            "overall_sentiment_label": item.get(
                                "overall_sentiment_label", ""
                            ),
                            "ticker_sentiments": ticker_sentiments,
                        }
                    )

        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {e}")

        return pd.DataFrame(articles)
