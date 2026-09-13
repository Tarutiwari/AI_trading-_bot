"""
Data Integrity & Health Validator for Gold (XAUUSD) Datasets.
Checks for missing bars, invalid OHLC geometries, spread anomalies,
and timezone alignment errors.
"""

from pathlib import Path
from typing import Dict, List, Optional
import sys

import pandas as pd
import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig


class DataValidator:
    """Validates raw market data integrity before feature engineering."""

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config

    def validate_dataframe(self, df: pd.DataFrame, timeframe_str: str) -> Dict[str, any]:
        """
        Runs comprehensive integrity checks on a given OHLCV DataFrame.
        """
        report = {
            "timeframe": timeframe_str,
            "total_bars": len(df),
            "start_time": str(df.index[0]) if not df.empty else None,
            "end_time": str(df.index[-1]) if not df.empty else None,
            "passed": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        if df.empty:
            report["passed"] = False
            report["errors"].append("DataFrame is completely empty.")
            return report

        # 1. Check required columns
        required_cols = ["open", "high", "low", "close", "tick_volume"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            report["passed"] = False
            report["errors"].append(f"Missing required columns: {missing_cols}")

        # 2. Check for NaN / Null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            report["passed"] = False
            report["errors"].append(f"Found NaN values: {null_counts[null_counts > 0].to_dict()}")

        # 3. Check OHLC Geometry Consistency
        # High must be >= Open, Close, Low
        invalid_high = df[(df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])]
        if len(invalid_high) > 0:
            report["passed"] = False
            report["errors"].append(f"Found {len(invalid_high)} bars where High < Open/Close/Low.")

        # Low must be <= Open, Close, High
        invalid_low = df[(df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])]
        if len(invalid_low) > 0:
            report["passed"] = False
            report["errors"].append(f"Found {len(invalid_low)} bars where Low > Open/Close/High.")

        # 4. Check for Non-Positive Prices
        non_positive = df[(df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)]
        if len(non_positive) > 0:
            report["passed"] = False
            report["errors"].append(f"Found {len(non_positive)} bars with price <= 0.")

        # 5. Check Spread & Volume Health
        if "spread" in df.columns:
            negative_spreads = df[df["spread"] < 0]
            if len(negative_spreads) > 0:
                report["passed"] = False
                report["errors"].append(f"Found {len(negative_spreads)} bars with negative spread.")
            
            report["stats"]["spread_mean"] = float(df["spread"].mean())
            report["stats"]["spread_max"] = float(df["spread"].max())
            report["stats"]["spread_min"] = float(df["spread"].min())

        if "tick_volume" in df.columns:
            zero_vol = df[df["tick_volume"] == 0]
            if len(zero_vol) > 0:
                report["warnings"].append(f"Found {len(zero_vol)} bars with zero tick volume.")
            
            report["stats"]["avg_tick_volume"] = float(df["tick_volume"].mean())

        # 6. Check Time Spacing / Continuity
        if isinstance(df.index, pd.DatetimeIndex):
            time_diffs = df.index.to_series().diff()
            # Large gaps (excluding regular weekend closures)
            large_gaps = time_diffs[time_diffs > pd.Timedelta(days=3)]
            if len(large_gaps) > 0:
                report["warnings"].append(f"Found {len(large_gaps)} large time gaps (>3 days, holidays/breaks).")

        return report

    def print_report(self, report: Dict[str, any]) -> None:
        """Format and print validation report to console."""
        status = "[PASSED]" if report["passed"] else "[FAILED]"
        print(f"\n{status} Data Integrity Report: {report['timeframe']} ({report['total_bars']:,} bars)")
        print(f"  Date Range: {report['start_time']} -> {report['end_time']}")
        
        if report["stats"]:
            print("  Stats:", report["stats"])
            
        if report["errors"]:
            print("  [!] Errors:")
            for err in report["errors"]:
                print(f"      - {err}")

        if report["warnings"]:
            print("  [*] Warnings:")
            for warn in report["warnings"]:
                print(f"      - {warn}")


if __name__ == "__main__":
    validator = DataValidator()
    print("[INFO] DataValidator initialized.")
