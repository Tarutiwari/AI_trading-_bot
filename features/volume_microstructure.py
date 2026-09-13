"""
Broker-Invariant Volume Normalization & Market Microstructure Features.
Handles tick volume, real volume, candle anatomy, and spread dynamics for Gold (XAUUSD).
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class VolumeMicrostructureExtractor:
    """
    Computes broker-agnostic volume signals and candle microstructure features.
    Transforms raw broker tick volume into standardized statistical metrics.
    """

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        self.rolling_windows = rolling_windows or [20, 50, 200]

    def compute_all_microstructure_features(
        self, df: pd.DataFrame, atr_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Computes volume normalizations, candle anatomy, and spread metrics.
        """
        feats = pd.DataFrame(index=df.index)
        eps = 1e-8

        # 1. Choose Best Available Volume Column
        vol_col = "tick_volume" if "tick_volume" in df.columns else ("real_volume" if "real_volume" in df.columns else None)
        
        if vol_col is not None:
            vol = df[vol_col].astype(float)

            # Broker-Agnostic Rolling Z-Scores & Relative Volume (RVOL)
            for w in self.rolling_windows:
                mean_vol = vol.rolling(window=w).mean()
                std_vol = vol.rolling(window=w).std()
                feats[f"vol_zscore_{w}"] = (vol - mean_vol) / (std_vol + eps)
                feats[f"rvol_{w}"] = vol / (mean_vol + eps)

            # Rolling Quantile Rank [0.0 to 1.0] (Completely broker scale invariant)
            # raw=True passes numpy array directly — ~100x faster than raw=False on large datasets
            feats["vol_quantile_50"] = (
                vol.rolling(window=50)
                .apply(lambda x: np.searchsorted(np.sort(x), x[-1]) / len(x), raw=True)
            )

            # Volume Pressure / Directional Flow (Tick Volume * Normalized Price Direction)
            candle_range = df["high"] - df["low"] + eps
            direction = (df["close"] - df["open"]) / candle_range
            feats["vol_pressure"] = (vol / (vol.rolling(window=20).mean() + eps)) * direction

            # Normalized On-Balance Volume (OBV)
            direction_sign = np.sign(df["close"].diff()).fillna(0)
            obv = (direction_sign * vol).cumsum()
            feats["obv_zscore_50"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + eps)

            # Normalized Volume-Price Trend (VPT)
            pct_change = df["close"].pct_change().fillna(0)
            vpt = (vol * pct_change).cumsum()
            feats["vpt_zscore_50"] = (vpt - vpt.rolling(50).mean()) / (vpt.rolling(50).std() + eps)

        # 2. Candle Anatomy & Price Action Rejection Geometry
        candle_range = df["high"] - df["low"] + eps
        max_oc = np.maximum(df["open"], df["close"])
        min_oc = np.minimum(df["open"], df["close"])

        # Upper rejection wick (Pin bar up)
        feats["upper_wick_ratio"] = (df["high"] - max_oc) / candle_range
        # Lower rejection wick (Pin bar down)
        feats["lower_wick_ratio"] = (min_oc - df["low"]) / candle_range
        # Candle Body Ratio (Momentum vs Indecision/Doji)
        feats["body_ratio"] = (max_oc - min_oc) / candle_range
        # Directional Sign
        feats["candle_direction"] = np.where(df["close"] >= df["open"], 1.0, -1.0)

        # 3. Spread Dynamics (Transaction Cost & Liquidity Volatility)
        if "spread" in df.columns:
            spread = df["spread"].astype(float)
            feats["spread_points"] = spread
            
            # Spread Z-Score: Identifies liquidity dry-ups or news release spread spikes
            spread_mean = spread.rolling(window=50).mean()
            spread_std = spread.rolling(window=50).std()
            feats["spread_zscore_50"] = (spread - spread_mean) / (spread_std + eps)

            # Spread-to-ATR ratio (How much volatility is eaten by spread)
            if atr_series is not None:
                # Convert ATR to points (e.g., $1.00 ATR on Gold = 100 points)
                atr_points = atr_series * 100.0
                feats["spread_to_atr_ratio"] = spread / (atr_points + eps)

        return feats
