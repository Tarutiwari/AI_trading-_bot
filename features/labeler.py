"""
Institutional Triple Barrier Labeling Engine (Marcos Lopez de Prado Methodology).
Generates realistic dynamic ATR-based Take-Profit, Stop-Loss, and Time Barrier labels
with realized return and volatility targets for Multi-Task Mamba learning.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


class TripleBarrierLabeler:
    """
    Applies dynamic ATR-based upper, lower, and vertical time barriers
    to generate realistic trade outcome targets.
    """

    def __init__(
        self,
        tp_multiplier: float = 1.5,
        sl_multiplier: float = 1.5,
        max_holding_bars: int = 24,
        min_return: float = 0.0005,
    ):
        self.tp_mult = tp_multiplier
        self.sl_mult = sl_multiplier
        self.max_bars = max_holding_bars
        self.min_ret = min_return

    def generate_labels(
        self,
        df: pd.DataFrame,
        atr_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Calculates Triple Barrier labels for each historical bar.
        
        Returns DataFrame with:
        - 'target_direction': 0 (Sell / Bearish), 1 (Neutral / Chop), 2 (Buy / Bullish)
        - 'target_return': Price return at barrier hit
        - 'target_volatility': Maximum price excursion within the window
        - 'target_regime': 0 (Low Vol Trend), 1 (Low Vol Chop), 2 (High Vol Trend), 3 (High Vol Chop)
        - 'barrier_hit_bar': Number of bars until barrier was touched
        """
        n = len(df)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        atrs = atr_series.values

        labels = np.full(n, -1, dtype=int)  # -1 = undefined sentinel (pipeline drops these via valid_mask)
        # NOTE: Do NOT default to 1 (Neutral) — ATR warmup bars (~200 bars with NaN ATR)
        # were silently becoming Neutral and artificially inflating its class count.
        realized_returns = np.zeros(n, dtype=float)
        realized_vols = np.zeros(n, dtype=float)
        barrier_bars = np.full(n, self.max_bars, dtype=int)

        print(f"[INFO] Computing Triple Barrier labels across {n:,} bars (TP={self.tp_mult}x ATR, SL={self.sl_mult}x ATR, Window={self.max_bars} bars)...")

        for i in range(n - self.max_bars):
            p0 = closes[i]
            atr = atrs[i]
            if np.isnan(atr) or atr <= 0:
                continue

            upper_barrier = p0 + (self.tp_mult * atr)
            lower_barrier = p0 - (self.sl_mult * atr)

            future_highs = highs[i + 1 : i + 1 + self.max_bars]
            future_lows = lows[i + 1 : i + 1 + self.max_bars]
            future_closes = closes[i + 1 : i + 1 + self.max_bars]

            tp_hit_idx = np.where(future_highs >= upper_barrier)[0]
            sl_hit_idx = np.where(future_lows <= lower_barrier)[0]

            first_tp = tp_hit_idx[0] if len(tp_hit_idx) > 0 else 9999
            first_sl = sl_hit_idx[0] if len(sl_hit_idx) > 0 else 9999

            if first_tp < first_sl and first_tp < self.max_bars:
                # Upper Take-Profit Barrier Hit First -> Bullish Setup (+1 -> Class 2)
                labels[i] = 2
                realized_returns[i] = (upper_barrier - p0) / p0
                barrier_bars[i] = first_tp + 1
            elif first_sl < first_tp and first_sl < self.max_bars:
                # Lower Stop-Loss Barrier Hit First -> Bearish Setup (-1 -> Class 0)
                labels[i] = 0
                realized_returns[i] = (lower_barrier - p0) / p0
                barrier_bars[i] = first_sl + 1
            else:
                # Vertical Time Barrier Hit First -> Neutral / Chop (0 -> Class 1)
                labels[i] = 1
                end_price = future_closes[-1]
                realized_returns[i] = (end_price - p0) / p0
                barrier_bars[i] = self.max_bars

            # Compute realized max excursion (volatility magnitude)
            max_dev = np.max(np.abs(future_closes - p0))
            realized_vols[i] = max_dev / p0

        # Market Regime Classification (Pure Volatility & Excursion based - decoupled from direction)
        # 0 = Low Vol Trend, 1 = Low Vol Chop, 2 = High Vol Trend, 3 = High Vol Chop
        non_zero_vols = realized_vols[realized_vols > 0]
        non_zero_rets = np.abs(realized_returns[realized_returns != 0])
        
        vol_median = np.nanpercentile(non_zero_vols, 50) if len(non_zero_vols) > 0 else 0.002
        ret_median = np.nanpercentile(non_zero_rets, 50) if len(non_zero_rets) > 0 else 0.001
        
        regimes = np.zeros(n, dtype=int)
        for i in range(n):
            v_high = realized_vols[i] >= vol_median
            r_trend = np.abs(realized_returns[i]) >= ret_median
            
            if not v_high and r_trend:
                regimes[i] = 0  # Low Vol, Trending
            elif not v_high and not r_trend:
                regimes[i] = 1  # Low Vol, Range/Chop
            elif v_high and r_trend:
                regimes[i] = 2  # High Vol, Trending
            else:
                regimes[i] = 3  # High Vol, Spiky/Chop

        target_df = pd.DataFrame(
            {
                "target_direction": labels,
                "target_return": realized_returns,
                "target_volatility": realized_vols,
                "target_regime": regimes,
                "barrier_hit_bar": barrier_bars,
            },
            index=df.index,
        )

        # Summary statistics
        class_counts = target_df["target_direction"].iloc[:-self.max_bars].value_counts().to_dict()
        print(f"[SUCCESS] Label distribution (0=Bearish, 1=Neutral, 2=Bullish): {class_counts}")
        return target_df
