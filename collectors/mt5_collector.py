"""
MT5 Multi-Timeframe Data Collector for Gold (XAUUSD).
Connects to MetaTrader 5, resolves the broker-specific Gold symbol,
and fetches historical OHLCV data with tick volume, real volume, and spreads.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig


class MT5DataCollector:
    """
    Production-grade historical data extractor for MetaTrader 5.
    Downloads multi-timeframe candles, preserves tick/real volume and spreads,
    and formats timestamps to standard UTC.
    """

    TIMEFRAME_LOOKUP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.symbol: Optional[str] = None
        self._connected = False

    def connect(self) -> bool:
        """Initialize connection to MetaTrader 5 terminal."""
        if not mt5.initialize():
            print(f"[ERROR] MT5 Initialization failed: {mt5.last_error()}")
            return False
        
        terminal_info = mt5.terminal_info()
        version_info = mt5.version()
        print(f"[INFO] Connected to MT5: Build {version_info[0]}, Terminal: {terminal_info.name}")
        self._connected = True
        return True

    def resolve_symbol(self) -> Optional[str]:
        """
        Auto-detect the active Gold symbol on the broker account.
        Checks priority candidates first, then searches for 'XAU' or 'GOLD'.
        """
        if not self._connected and not self.connect():
            return None

        # 1. Check priority candidates
        for candidate in self.config.instrument.SYMBOL_CANDIDATES:
            info = mt5.symbol_info(candidate)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(candidate, True)
                self.symbol = candidate
                print(f"[INFO] Resolved Gold Symbol: '{self.symbol}' (Spread: {info.spread} pts, Digits: {info.digits})")
                return self.symbol

        # 2. Fallback: Search all symbols in broker terminal
        all_symbols = mt5.symbols_get()
        if all_symbols:
            gold_matches = [
                s.name for s in all_symbols 
                if "XAU" in s.name.upper() or "GOLD" in s.name.upper()
            ]
            if gold_matches:
                chosen = gold_matches[0]
                mt5.symbol_select(chosen, True)
                self.symbol = chosen
                print(f"[INFO] Auto-detected Gold Symbol from market watch: '{self.symbol}'")
                return self.symbol

        print("[ERROR] No Gold symbol found on your MT5 broker account.")
        return None

    def fetch_rates(
        self,
        timeframe_str: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bars for a given timeframe between start_date and end_date.
        Preserves Open, High, Low, Close, Tick Volume, Real Volume, and Spread.
        """
        if not self.symbol and not self.resolve_symbol():
            return None

        if timeframe_str not in self.TIMEFRAME_LOOKUP:
            raise ValueError(f"Unsupported timeframe '{timeframe_str}'. Supported: {list(self.TIMEFRAME_LOOKUP.keys())}")

        tf_constant = self.TIMEFRAME_LOOKUP[timeframe_str]
        end_dt = end_date or datetime.now(timezone.utc)

        # Ping symbol tick to force terminal history synchronization
        mt5.symbol_info_tick(self.symbol)
        time.sleep(0.5)

        print(f"[INFO] Fetching '{self.symbol}' [{timeframe_str}] from {start_date.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")

        rates = mt5.copy_rates_range(
            self.symbol,
            tf_constant,
            start_date,
            end_dt,
        )

        if rates is None or len(rates) == 0:
            print(f"[WARN] No rates returned for {self.symbol} [{timeframe_str}]. MT5 Error: {mt5.last_error()}")
            return None

        # Convert structured numpy array to Pandas DataFrame
        df = pd.DataFrame(rates)
        
        # Convert unix timestamp (seconds) to readable UTC datetime
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)

        # Reorder and rename columns cleanly
        columns_to_keep = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        available_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[available_cols]

        print(f"[SUCCESS] Downloaded {len(df):,} candles for {self.symbol} [{timeframe_str}] "
              f"({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        return df

    def collect_all_timeframes(
        self,
        timeframes: Optional[List[str]] = None,
        start_date_str: Optional[str] = None,
        save_to_disk: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all specified timeframes and optionally save raw files to disk.
        """
        if timeframes is None:
            timeframes = [self.config.instrument.PRIMARY_TIMEFRAME] + self.config.instrument.CONTEXT_TIMEFRAMES

        start_str = start_date_str or self.config.data.START_DATE
        start_date = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        collected_data = {}
        for tf in timeframes:
            df = self.fetch_rates(timeframe_str=tf, start_date=start_date)
            if df is not None and not df.empty:
                collected_data[tf] = df
                if save_to_disk:
                    self._save_raw(df, tf)

        return collected_data

    def _save_raw(self, df: pd.DataFrame, timeframe_str: str) -> None:
        """Save dataframe to raw data directory as CSV and Parquet."""
        raw_dir = self.config.paths.RAW_DATA_DIR
        clean_symbol = self.symbol.replace(".", "_").replace("/", "_") if self.symbol else "XAUUSD"
        
        csv_path = raw_dir / f"{clean_symbol}_{timeframe_str}_raw.csv"
        parquet_path = raw_dir / f"{clean_symbol}_{timeframe_str}_raw.parquet"

        df.to_csv(csv_path)
        df.to_parquet(parquet_path)
        print(f"[SAVED] Saved raw {timeframe_str} data -> {csv_path.name} & {parquet_path.name}")

    def shutdown(self) -> None:
        """Disconnect and release MT5 resources."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            print("[INFO] MetaTrader 5 disconnected cleanly.")


if __name__ == "__main__":
    collector = MT5DataCollector()
    try:
        if collector.connect():
            symbol = collector.resolve_symbol()
            if symbol:
                print(f"\n==========================================")
                print(f"  Starting Multi-Timeframe Data Pull: {symbol}")
                print(f"==========================================")
                data = collector.collect_all_timeframes()
                print(f"\n[DONE] Successfully pulled {len(data)} timeframes.")
    finally:
        collector.shutdown()
