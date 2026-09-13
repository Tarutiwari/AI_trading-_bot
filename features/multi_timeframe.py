"""
Zero-Leakage Multi-Timeframe Alignment Engine.
Computes technical and volume features on higher timeframes (M15, H1, H4),
strictly lags them by 1 completed bar, and backward-merges them into the primary M5 timeframe.
"""

from typing import Dict, List, Optional
import pandas as pd

from features.technical import TechnicalFeatureExtractor
from features.volume_microstructure import VolumeMicrostructureExtractor


class MultiTimeframeAligner:
    """
    Guarantees mathematically rigorous, lookahead-free multi-timeframe feature alignment.
    Higher-timeframe candles are strictly lagged by 1 full bar (T - 1) before merging.
    """

    def __init__(self):
        self.tech_extractor = TechnicalFeatureExtractor()
        self.vol_extractor = VolumeMicrostructureExtractor()

    def extract_and_align(
        self,
        primary_df: pd.DataFrame,
        higher_tf_dfs: Dict[str, pd.DataFrame],
        primary_tf_name: str = "M5",
    ) -> pd.DataFrame:
        """
        Extracts features on primary and higher timeframes, enforces 1-bar lagging
        on higher timeframes, and merges safely via backward merge_asof.
        """
        # 1. Primary Timeframe Features
        print(f"[INFO] Computing features for primary timeframe: [{primary_tf_name}]...")
        primary_tech = self.tech_extractor.compute_all_technical_features(primary_df)
        primary_vol = self.vol_extractor.compute_all_microstructure_features(
            primary_df, atr_series=primary_tech["atr"]
        )
        base_features = pd.concat([primary_tech, primary_vol], axis=1)

        # 2. Process and Lag Each Higher Timeframe
        aligned_df = base_features.copy()

        for tf_name, htf_df in higher_tf_dfs.items():
            if htf_df.empty:
                continue
            
            print(f"[INFO] Processing and lagging higher timeframe: [{tf_name}] (Zero-Leakage)...")
            # Compute indicators strictly on the higher timeframe's historical bars
            htf_tech = self.tech_extractor.compute_all_technical_features(htf_df)
            htf_vol = self.vol_extractor.compute_all_microstructure_features(
                htf_df, atr_series=htf_tech["atr"]
            )
            htf_combined = pd.concat([htf_tech, htf_vol], axis=1)

            # Prefix column names with the timeframe name (e.g. H1_rsi, H1_parkinson_vol)
            htf_combined = htf_combined.add_prefix(f"{tf_name}_")

            # CRITICAL: Shift by 1 completed bar to prevent looking into the unclosed candle!
            htf_lagged = htf_combined.shift(1)

            # Ensure both dataframes have sorted DatetimeIndexes for merge_asof
            aligned_df.sort_index(inplace=True)
            htf_lagged.sort_index(inplace=True)

            # Backward merge: each M5 candle receives the most recently COMPLETED higher-TF bar
            aligned_df = pd.merge_asof(
                aligned_df,
                htf_lagged,
                left_index=True,
                right_index=True,
                direction="backward",
            )

        print(f"[SUCCESS] Multi-timeframe alignment completed. Total features: {aligned_df.shape[1]}")
        return aligned_df
