"""
Event-Driven, Transaction-Cost-Aware Backtesting Engine.
Simulates realistic trade lifecycles on Gold (XAUUSD) including dynamic spreads,
commissions, slippage, trailing stops, and breakeven adjustments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import sys

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig
from models.mamba_model import MambaTradingModel
from execution.risk_manager import InstitutionalRiskManager
from backtest.metrics import PerformanceMetrics


class CostAwareBacktester:
    """
    Simulates institutional execution mechanics with realistic friction:
    - Dynamic broker spread
    - Commission ($7/standard lot round-turn)
    - Slippage (simulated execution delay)
    - Trailing stop-loss & breakeven rules
    """

    def __init__(
        self,
        config: BotConfig = CONFIG,
        initial_capital: float = 10_000.0,
        commission_per_lot: float = 7.0,  # $7 per standard lot
        slippage_points: float = 10.0,    # 10 points = $0.10
    ):
        self.config = config
        self.initial_capital = initial_capital
        self.commission_per_lot = commission_per_lot
        self.slippage_points = slippage_points
        self.risk_manager = InstitutionalRiskManager(config)

    def run_backtest(
        self,
        df_market: pd.DataFrame,
        X_features: np.ndarray,
        model: MambaTradingModel,
        device: torch.device,
    ) -> Tuple[pd.Series, List[Dict[str, any]], Dict[str, any]]:
        """
        Executes event-driven backtesting across the provided market dataset.
        """
        model.eval()
        seq_len = self.config.features.LOOKBACK_SEQUENCE_LENGTH
        n_bars = len(df_market)

        equity = self.initial_capital
        equity_curve = []
        timestamps = []
        trades = []

        open_position: Optional[Dict[str, any]] = None

        print(f"\n=======================================================")
        print(f"  Starting Cost-Aware Backtest ({n_bars:,} bars | Capital: ${self.initial_capital:,.2f})")
        print(f"=======================================================")

        for t in range(seq_len, n_bars):
            current_bar = df_market.iloc[t]
            current_time = df_market.index[t]
            open_p = current_bar["open"]
            high_p = current_bar["high"]
            low_p = current_bar["low"]
            close_p = current_bar["close"]
            spread_pts = current_bar.get("spread", 20.0)
            atr = current_bar.get("atr", 1.50)
            is_news = bool(current_bar.get("is_news_freeze", 0))

            # Reset daily equity tracker for drawdown monitoring
            self.risk_manager.reset_daily_tracking_if_new_day(equity, current_time)

            # ----------------------------------------------------
            # 1. Manage Active Open Position (SL, TP, Trailing)
            # ----------------------------------------------------
            if open_position is not None:
                pos = open_position
                is_buy = pos["type"] == "BUY"
                lot = pos["lot"]
                entry_p = pos["entry_price"]
                sl = pos["sl"]
                tp = pos["tp"]
                bars_held = t - pos["entry_bar"]

                closed = False
                exit_price = 0.0
                exit_reason = ""

                # Check Stop-Loss
                if is_buy and low_p <= sl:
                    exit_price = sl - (self.slippage_points * 0.01)
                    exit_reason = "Stop Loss"
                    closed = True
                elif not is_buy and high_p >= sl:
                    exit_price = sl + (self.slippage_points * 0.01)
                    exit_reason = "Stop Loss"
                    closed = True

                # Check Take-Profit
                elif is_buy and high_p >= tp:
                    exit_price = tp
                    exit_reason = "Take Profit"
                    closed = True
                elif not is_buy and low_p <= tp:
                    exit_price = tp
                    exit_reason = "Take Profit"
                    closed = True

                # Check Max Holding Bars Limit (Time Barrier)
                elif bars_held >= self.config.labeling.MAX_HOLDING_BARS:
                    exit_price = close_p
                    exit_reason = "Time Horizon Exit"
                    closed = True

                # Trailing Stop & Breakeven Updates
                if not closed:
                    initial_risk = abs(entry_p - sl)
                    if is_buy:
                        profit_dist = close_p - entry_p
                        # Breakeven
                        if profit_dist >= initial_risk and sl < entry_p:
                            pos["sl"] = round(entry_p + 0.10, 2)
                        # Trailing
                        if self.config.risk.ENABLE_TRAILING_STOP and pos["sl"] >= entry_p:
                            proposed_sl = round(close_p - (atr * self.config.risk.TRAILING_ATR_MULTIPLE), 2)
                            if proposed_sl > pos["sl"]:
                                pos["sl"] = proposed_sl
                    else:
                        profit_dist = entry_p - close_p
                        if profit_dist >= initial_risk and sl > entry_p:
                            pos["sl"] = round(entry_p - 0.10, 2)
                        if self.config.risk.ENABLE_TRAILING_STOP and pos["sl"] <= entry_p:
                            proposed_sl = round(close_p + (atr * self.config.risk.TRAILING_ATR_MULTIPLE), 2)
                            if proposed_sl < pos["sl"]:
                                pos["sl"] = proposed_sl

                # Finalize closed trade
                if closed:
                    # Calculate PnL: ($1.00 per point per standard lot = $100 per $1.00 price change per lot)
                    price_diff = (exit_price - entry_p) if is_buy else (entry_p - exit_price)
                    gross_pnl = price_diff * 100.0 * lot
                    commission = self.commission_per_lot * lot
                    net_pnl = gross_pnl - commission

                    equity += net_pnl
                    trades.append({
                        "entry_time": pos["entry_time"],
                        "exit_time": current_time,
                        "type": pos["type"],
                        "lot": lot,
                        "entry_price": entry_p,
                        "exit_price": exit_price,
                        "pnl": net_pnl,
                        "exit_reason": exit_reason,
                        "bars_held": bars_held,
                    })
                    open_position = None

            # ----------------------------------------------------
            # 2. Check for New Trade Entry Signals (If Flat)
            # ----------------------------------------------------
            if open_position is None:
                # Extract sequence window
                window_x = X_features[t - seq_len : t]
                tensor_x = torch.tensor(window_x, dtype=torch.float32).unsqueeze(0).to(device)

                # Mamba Model Inference
                pred = model.predict_trade_action(
                    tensor_x, confidence_threshold=self.config.risk.MIN_CONFIDENCE_THRESHOLD
                )
                action = pred["action"]
                confidence = pred["confidence"]

                if action in ["BUY", "SELL"]:
                    # Pre-flight institutional risk validation
                    valid, reason = self.risk_manager.validate_trade_entry(
                        current_spread_points=spread_pts,
                        current_equity=equity,
                        is_news_freeze=is_news,
                        model_confidence=confidence,
                    )

                    if valid:
                        # Slippage + Spread adjusted entry price
                        spread_cost = (spread_pts * 0.01)
                        slippage_cost = (self.slippage_points * 0.01)
                        entry_price = (open_p + spread_cost + slippage_cost) if action == "BUY" else (open_p - slippage_cost)

                        sl_price, tp_price = self.risk_manager.compute_sl_tp_prices(
                            action=action, entry_price=entry_price, atr=atr
                        )
                        lot = self.risk_manager.calculate_lot_size(
                            account_equity=equity,
                            entry_price=entry_price,
                            stop_loss_price=sl_price,
                        )

                        open_position = {
                            "type": action,
                            "lot": lot,
                            "entry_price": entry_price,
                            "sl": sl_price,
                            "tp": tp_price,
                            "entry_time": current_time,
                            "entry_bar": t,
                        }

            # Record current equity
            equity_curve.append(equity)
            timestamps.append(current_time)

        eq_series = pd.Series(equity_curve, index=timestamps)
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_series=eq_series,
            trade_records=trades,
            initial_capital=self.initial_capital,
        )

        PerformanceMetrics.print_tearsheet(metrics)
        return eq_series, trades, metrics
