"""Reddit data source for financial discussions."""

import os
from typing import Dict, Any
import pandas as pd
import praw
from .base import BaseDataSource


class RedditDataSource(BaseDataSource):
    """Fetch financial discussions from Reddit."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Reddit data source.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "financial-sentiment-bot")

        if not all([client_id, client_secret]):
            print("Warning: Reddit API credentials not found. Reddit will be disabled.")
            self.enabled = False
            self.reddit = None
        else:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch posts and comments from Reddit.

        Returns:
            DataFrame with Reddit posts and comments
        """
        if not self.reddit:
            return pd.DataFrame()

        posts = []

        try:
            subreddits = self.config.get("subreddits", ["wallstreetbets", "stocks"])
            limit = self.config.get("limit", 100)
            time_filter = self.config.get("time_filter", "week")

            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)

                # Fetch hot posts
                for submission in subreddit.hot(limit=limit):
                    # Skip stickied posts
                    if submission.stickied:
                        continue

                    posts.append(
                        {
                            "text": f"{submission.title} {submission.selftext}",
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "url": submission.url,
                            "source": "reddit",
                            "source_name": f"r/{subreddit_name}",
                            "author": str(submission.author),
                            "timestamp": pd.to_datetime(
                                submission.created_utc, unit="s"
                            ),
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "post_id": submission.id,
                        }
                    )

                # Fetch top posts for the time period
                for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    if submission.stickied:
                        continue

                    posts.append(
                        {
                            "text": f"{submission.title} {submission.selftext}",
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "url": submission.url,
                            "source": "reddit",
                            "source_name": f"r/{subreddit_name}",
                            "author": str(submission.author),
                            "timestamp": pd.to_datetime(
                                submission.created_utc, unit="s"
                            ),
                            "score": submission.score,
                            "upvote_ratio": submission.upvote_ratio,
                            "num_comments": submission.num_comments,
                            "post_id": submission.id,
                        }
                    )

        except Exception as e:
            print(f"Error fetching data from Reddit: {e}")

        # Remove duplicates based on post_id
        df = pd.DataFrame(posts)
        if not df.empty and "post_id" in df.columns:
            df = df.drop_duplicates(subset=["post_id"], keep="first")

        return df
