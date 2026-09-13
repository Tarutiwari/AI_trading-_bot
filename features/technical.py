"""
Technical Indicators & Stationary Price Transformations for Gold (XAUUSD).
Calculates volatility, momentum, trend alignment, and stationary returns.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class TechnicalFeatureExtractor:
    """
    Computes stationary, normalized technical features for time series models.
    All outputs are strictly stationary to prevent regime-shift degradation.
    """

    def __init__(self, rsi_period: int = 14, atr_period: int = 14):
        self.rsi_period = rsi_period
        self.atr_period = atr_period

    def compute_all_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the complete suite of technical indicators on an OHLCV DataFrame.
        """
        feats = pd.DataFrame(index=df.index)

        # 1. Stationary Price Returns
        feats["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
        feats["log_ret_3"] = np.log(df["close"] / df["close"].shift(3))
        feats["log_ret_6"] = np.log(df["close"] / df["close"].shift(6))
        feats["log_ret_12"] = np.log(df["close"] / df["close"].shift(12))

        # 2. Average True Range (ATR) & Normalized Volatility
        atr = self._calculate_atr(df, period=self.atr_period)
        feats["atr"] = atr
        feats["norm_atr"] = atr / (df["close"] + 1e-8)  # Relative volatility percentage

        # 3. Parkinson & Garman-Klass Microstructure Volatility
        feats["parkinson_vol"] = self._calculate_parkinson_volatility(df, window=self.atr_period)
        feats["garman_klass_vol"] = self._calculate_garman_klass_volatility(df, window=self.atr_period)

        # 4. Momentum Oscillators: RSI & RSI Velocity
        rsi = self._calculate_rsi(df["close"], period=self.rsi_period)
        feats["rsi"] = (rsi - 50.0) / 50.0  # Center RSI to [-1.0, +1.0]
        feats["rsi_slope_3"] = (rsi - rsi.shift(3)) / 50.0  # Rate of change of momentum
        feats["rsi_slope_5"] = (rsi - rsi.shift(5)) / 50.0

        # Stochastic RSI
        stoch_k, stoch_d = self._calculate_stoch_rsi(rsi, period=14, smooth_k=3, smooth_d=3)
        feats["stoch_rsi_k"] = (stoch_k - 50.0) / 50.0
        feats["stoch_rsi_d"] = (stoch_d - 50.0) / 50.0

        # 5. Trend Alignment: Normalized EMA Deviations
        for span in [9, 21, 50, 200]:
            ema = df["close"].ewm(span=span, adjust=False).mean()
            feats[f"ema_dev_{span}"] = (df["close"] - ema) / (atr + 1e-8)  # Expressed in ATR units

        # EMA Slope (Trend velocity)
        ema_21 = df["close"].ewm(span=21, adjust=False).mean()
        feats["ema_21_slope_5"] = (ema_21 - ema_21.shift(5)) / (atr + 1e-8)

        # 6. Bollinger Bands Bandwidth & %B
        bb_mid = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        bb_upper = bb_mid + (2.0 * bb_std)
        bb_lower = bb_mid - (2.0 * bb_std)

        feats["bb_bandwidth"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        feats["bb_percent_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8) - 0.5  # [-0.5, +0.5]

        # 7. MACD Normalized by ATR
        ema_fast = df["close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        feats["macd_hist_norm"] = macd_hist / (atr + 1e-8)

        return feats

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _calculate_stoch_rsi(
        rsi_series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
    ) -> tuple:
        """Calculate Stochastic RSI (%K and %D)."""
        rsi_min = rsi_series.rolling(window=period).min()
        rsi_max = rsi_series.rolling(window=period).max()
        stoch = 100.0 * (rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-8)
        stoch_k = stoch.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        return stoch_k, stoch_d

    @staticmethod
    def _calculate_parkinson_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Parkinson volatility based on High-Low range (more efficient than Close-to-Close)."""
        hl_ratio = np.log(df["high"] / (df["low"] + 1e-8))
        factor = 1.0 / (4.0 * np.log(2.0))
        parkinson = np.sqrt(factor * (hl_ratio ** 2).rolling(window=window).mean())
        return parkinson

    @staticmethod
    def _calculate_garman_klass_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Garman-Klass volatility incorporating Open, High, Low, and Close."""
        log_hl = np.log(df["high"] / (df["low"] + 1e-8))
        log_co = np.log(df["close"] / (df["open"] + 1e-8))
        rs = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        return np.sqrt(rs.rolling(window=window).mean().clip(lower=0))
