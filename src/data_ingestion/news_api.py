"""News API data source."""

import os
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
from newsapi import NewsApiClient
from .base import BaseDataSource


class NewsAPIDataSource(BaseDataSource):
    """Fetch financial news from News API."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize News API data source.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            print("Warning: NEWS_API_KEY not found. News API will be disabled.")
            self.enabled = False
            self.client = None
        else:
            self.client = NewsApiClient(api_key=api_key)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch news articles from News API.

        Returns:
            DataFrame with news articles
        """
        if not self.client:
            return pd.DataFrame()

        # Calculate date range
        lookback_days = self.config.get("lookback_days", 7)
        from_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        articles = []

        try:
            # Fetch articles based on keywords
            keywords = self.config.get("keywords", "")
            sources = self.config.get("sources", None)
            language = self.config.get("language", "en")
            page_size = self.config.get("page_size", 100)

            response = self.client.get_everything(
                q=keywords,
                sources=sources,
                language=language,
                from_param=from_date,
                sort_by="publishedAt",
                page_size=page_size,
            )

            if response["status"] == "ok":
                for article in response.get("articles", []):
                    articles.append(
                        {
                            "text": f"{article.get('title', '')} {article.get('description', '')}",
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "url": article.get("url", ""),
                            "source": "news_api",
                            "source_name": article.get("source", {}).get("name", ""),
                            "author": article.get("author", ""),
                            "timestamp": pd.to_datetime(article.get("publishedAt")),
                            "content": article.get("content", ""),
                        }
                    )

        except Exception as e:
            print(f"Error fetching data from News API: {e}")

        return pd.DataFrame(articles)
