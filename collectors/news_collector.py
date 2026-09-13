"""
Financial News & Macro Sentiment Collector for Gold (XAUUSD).
Fetches live and historical financial news headlines, economic feeds,
and computes sentiment polarity scores (Bullish vs Bearish) for model ingestion.
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import urllib.request
import re
import sys

import pandas as pd
import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig


class NewsCollector:
    """
    Collects real-time and historical news headlines for Gold & USD,
    computes sentiment polarity, and maps headlines to candle timestamps.
    """

    # Free public financial RSS feeds (Zero API key required)
    RSS_FEEDS = {
        "yahoo_gold": "https://finance.yahoo.com/news/rssindex",
        "forexlive": "https://www.forexlive.com/feed/news",
        "marketwatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
    }

    # Gold-specific keywords for relevance filtering
    RELEVANT_KEYWORDS = [
        "gold", "xauusd", "xau", "precious metals", "bullion",
        "federal reserve", "fed", "powell", "inflation", "cpi",
        "interest rate", "dollar", "dxy", "treasury yield", "fomc",
        "nonfarm", "nfp", "unemployment", "ppi"
    ]

    # Keyword-based sentiment dictionary for financial domain
    BULLISH_WORDS = {
        "surge", "rally", "jump", "gains", "bullish", "soar", "record high",
        "breakout", "cut rates", "rate cut", "dovish", "inflation high", "safe haven",
        "weak dollar", "dollar drops", "dollar slides", "stimulus", "buying", "climb"
    }

    BEARISH_WORDS = {
        "drop", "fall", "plunge", "decline", "bearish", "tumble", "slump",
        "hike rates", "rate hike", "hawkish", "strong dollar", "dollar rallies",
        "dollar surges", "yields rise", "yields surge", "selloff", "losses", "retreat"
    }

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.news_df: Optional[pd.DataFrame] = None
        self._load_cached_news()

    def _load_cached_news(self) -> None:
        """Load cached news dataset from disk if available."""
        news_path = self.config.paths.DATA_DIR / "news_sentiment.csv"
        if news_path.exists():
            self.news_df = pd.read_csv(news_path)
            self.news_df["time"] = pd.to_datetime(self.news_df["time"], utc=True)
            self.news_df.sort_values("time", inplace=True)
            print(f"[INFO] Loaded {len(self.news_df)} cached news items.")
        else:
            self.news_df = pd.DataFrame(columns=["time", "title", "source", "sentiment_score", "relevance_score"])

    def fetch_live_rss_news(self) -> pd.DataFrame:
        """
        Fetches the latest headlines from public financial RSS feeds without needing API keys.
        """
        articles = []
        now = datetime.now(timezone.utc)

        for source_name, url in self.RSS_FEEDS.items():
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    xml_content = response.read()
                    root = ET.fromstring(xml_content)

                    # Parse standard RSS items
                    for item in root.findall(".//item"):
                        title = item.find("title")
                        pub_date = item.find("pubDate")
                        
                        title_text = title.text if title is not None else ""
                        pub_text = pub_date.text if pub_date is not None else ""

                        # Filter for relevance to Gold & USD Macro
                        relevance = self._calculate_relevance(title_text)
                        if relevance > 0:
                            sentiment = self._calculate_sentiment(title_text)
                            articles.append({
                                "time": now,
                                "title": title_text,
                                "source": source_name,
                                "sentiment_score": sentiment,
                                "relevance_score": relevance,
                            })

            except Exception as e:
                print(f"[WARN] Failed to fetch RSS feed from {source_name}: {e}")

        if articles:
            new_df = pd.DataFrame(articles)
            new_df["time"] = pd.to_datetime(new_df["time"], utc=True)
            self.news_df = pd.concat([self.news_df, new_df]).drop_duplicates(subset=["title"])
            self.news_df.sort_values("time", inplace=True)
            
            # Save cache
            news_path = self.config.paths.DATA_DIR / "news_sentiment.csv"
            self.news_df.to_csv(news_path, index=False)
            print(f"[SUCCESS] Fetched & saved {len(articles)} relevant market headlines.")
            return new_df

        return pd.DataFrame()

    def _calculate_relevance(self, text: str) -> float:
        """Score relevance to Gold and Macro factors between [0.0, 1.0]."""
        text_lower = text.lower()
        matches = sum(1 for kw in self.RELEVANT_KEYWORDS if kw in text_lower)
        return min(1.0, matches * 0.35)

    def _calculate_sentiment(self, text: str) -> float:
        """
        Computes financial sentiment polarity score:
        +1.0 = Strong Bullish Gold
        -1.0 = Strong Bearish Gold
         0.0 = Neutral
        """
        text_lower = text.lower()
        bull_score = sum(1 for w in self.BULLISH_WORDS if w in text_lower)
        bear_score = sum(1 for w in self.BEARISH_WORDS if w in text_lower)

        total = bull_score + bear_score
        if total == 0:
            return 0.0

        polarity = (bull_score - bear_score) / total
        return float(np.clip(polarity, -1.0, 1.0))

    def map_sentiment_to_market_candles(
        self, df_timestamps: pd.DatetimeIndex, rolling_hours: int = 4
    ) -> pd.DataFrame:
        """
        Computes rolling news sentiment features aligned with candle timestamps:
        1. rolling_news_sentiment: Mean sentiment over trailing window (e.g. 4 hours)
        2. news_volume_intensity: Number of relevant news stories in trailing window
        """
        feats = pd.DataFrame(index=df_timestamps)
        feats["news_sentiment_mean_4h"] = 0.0
        feats["news_count_4h"] = 0.0

        if self.news_df is None or self.news_df.empty:
            return feats

        news_times = self.news_df["time"].values
        news_sentiments = self.news_df["sentiment_score"].values
        timestamps = df_timestamps.values

        # Rolling window aggregation for each candle
        delta_window = np.timedelta64(rolling_hours, "h")

        sentiment_means = []
        news_counts = []

        for ts in timestamps:
            start_window = ts - delta_window
            # Mask news occurring in [ts - 4h, ts]
            mask = (news_times >= start_window) & (news_times <= ts)
            if np.any(mask):
                sentiment_means.append(float(np.mean(news_sentiments[mask])))
                news_counts.append(int(np.sum(mask)))
            else:
                sentiment_means.append(0.0)
                news_counts.append(0)

        feats["news_sentiment_mean_4h"] = sentiment_means
        feats["news_count_4h"] = news_counts
        return feats


if __name__ == "__main__":
    collector = NewsCollector()
    print("\n==========================================")
    print("      TESTING LIVE FINANCIAL NEWS FEED    ")
    print("==========================================")
    df = collector.fetch_live_rss_news()
    if not df.empty:
        print(df[["time", "source", "sentiment_score", "title"]].head(10))
    else:
        print("[INFO] No new items fetched or network unavailable.")
