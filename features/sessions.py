"""
Market Session Timing, Cyclical Time Embeddings & Macro News Distance Features.
Captures Asian consolidation, London expansion, and NY overlap regimes for Gold.
"""

from typing import Optional
import numpy as np
import pandas as pd

from collectors.calendar_collector import CalendarCollector
from collectors.news_collector import NewsCollector


class SessionFeatureExtractor:
    """
    Encodes cyclical time patterns, trading sessions, macroeconomic event proximities,
    and rolling news sentiment scores.
    """

    def __init__(
        self,
        calendar_collector: Optional[CalendarCollector] = None,
        news_collector: Optional[NewsCollector] = None,
    ):
        self.calendar = calendar_collector or CalendarCollector()
        self.news = news_collector or NewsCollector()

    def compute_session_features(self, df_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Computes sine/cosine cyclical time representations, session binary flags,
        macro event distance features, and rolling news sentiment metrics.
        """
        feats = pd.DataFrame(index=df_index)

        # 1. Cyclical Time Embeddings (Continuous Representation of 24h & 7d Cycles)
        hours = df_index.hour + (df_index.minute / 60.0)
        feats["sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
        feats["cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)

        day_of_week = df_index.dayofweek
        feats["sin_day"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        feats["cos_day"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

        # 2. Institutional Trading Session Flags (UTC)
        # Asian Session: 00:00 - 08:00 UTC (Tokyo/Sydney)
        feats["is_asian_session"] = ((df_index.hour >= 0) & (df_index.hour < 8)).astype(float)

        # London Session: 07:00 - 16:00 UTC
        feats["is_london_session"] = ((df_index.hour >= 7) & (df_index.hour < 16)).astype(float)

        # New York Session: 12:00 - 21:00 UTC
        feats["is_ny_session"] = ((df_index.hour >= 12) & (df_index.hour < 21)).astype(float)

        # London + NY Overlap (Peak Gold Liquidity & Volatility): 12:00 - 16:00 UTC
        feats["is_london_ny_overlap"] = ((df_index.hour >= 12) & (df_index.hour < 16)).astype(float)

        # Friday Close Risk Flag (Avoid holding weekend gap risk)
        feats["is_friday_late"] = ((day_of_week == 4) & (df_index.hour >= 19)).astype(float)

        # 3. Strict Point-in-Time Macroeconomic Event Context & Decayed Surprise
        macro_feats = self.calendar.compute_point_in_time_macro_features(df_index)
        if not macro_feats.empty:
            for col in macro_feats.columns:
                feats[col] = macro_feats[col]

        # 4. Financial News Headline Sentiment Features (Bullish vs Bearish)
        sentiment_feats = self.news.map_sentiment_to_market_candles(df_index, rolling_hours=4)
        if not sentiment_feats.empty:
            feats["news_sentiment_mean_4h"] = sentiment_feats["news_sentiment_mean_4h"]
            feats["news_count_4h"] = sentiment_feats["news_count_4h"]

        return feats
