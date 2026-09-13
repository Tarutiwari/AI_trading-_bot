"""
Institutional Risk Management & Position Sizing Engine.
Calculates volatility-adjusted lot sizing, enforces daily drawdown kill-switches,
macro news circuit breakers, and spread filters for Gold (XAUUSD).
"""

from datetime import datetime, date
from typing import Dict, Optional, Tuple
import math
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig


class InstitutionalRiskManager:
    """
    Mathematical risk management engine that enforces strict capital preservation.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.daily_starting_equity: Optional[float] = None
        self.current_date: Optional[date] = None
        self.is_circuit_breaker_active: bool = False
        self.consecutive_losses: int = 0
        self.daily_trade_count: int = 0
        self.daily_pnl: float = 0.0

    def reset_daily_tracking_if_new_day(self, current_equity: float, current_dt: datetime) -> None:
        """Reset starting equity at 00:00 UTC each day."""
        today = current_dt.date()
        if self.current_date != today or self.daily_starting_equity is None:
            if self.daily_starting_equity is not None:
                # Log previous day's performance
                daily_change = current_equity - self.daily_starting_equity
                daily_pct = (daily_change / self.daily_starting_equity) * 100
                print(f"[RISK] Previous day closed: ${self.daily_starting_equity:,.2f} -> ${current_equity:,.2f} ({daily_pct:+.2f}%)")
            self.current_date = today
            self.daily_starting_equity = current_equity
            self.is_circuit_breaker_active = False
            self.consecutive_losses = 0
            self.daily_trade_count = 0
            self.daily_pnl = 0.0
            print(f"[RISK] Daily equity reset for {today}: ${current_equity:,.2f}")

    def check_daily_drawdown(self, current_equity: float) -> Tuple[bool, float]:
        """
        Check if account has hit the maximum daily loss limit (e.g. 3%).
        Returns: (is_halted, current_drawdown_pct)
        """
        if self.daily_starting_equity is None or self.daily_starting_equity <= 0:
            return False, 0.0

        drawdown_pct = (self.daily_starting_equity - current_equity) / self.daily_starting_equity
        max_allowed = self.config.risk.MAX_DAILY_DRAWDOWN_PCT

        if drawdown_pct >= max_allowed:
            self.is_circuit_breaker_active = True
            print(
                f"[ALERT] MAXIMUM DAILY DRAWDOWN HIT! "
                f"Current Loss: {drawdown_pct*100:.2f}% (Limit: {max_allowed*100:.1f}%). "
                f"Halting trading for today."
            )
            return True, drawdown_pct

        # Check consecutive losses
        if self.consecutive_losses >= self.config.risk.MAX_CONSECUTIVE_LOSSES:
            self.is_circuit_breaker_active = True
            print(
                f"[ALERT] MAXIMUM CONSECUTIVE LOSSES HIT! "
                f"{self.consecutive_losses} losses in a row. "
                f"Halting trading for today."
            )
            return True, drawdown_pct

        return False, drawdown_pct

    def record_trade_result(self, pnl: float) -> None:
        """Record the result of a closed trade for daily tracking."""
        self.daily_trade_count += 1
        self.daily_pnl += pnl
        
        if pnl < 0:
            self.consecutive_losses += 1
            print(f"[RISK] Trade closed: ${pnl:+.2f} | Consecutive losses: {self.consecutive_losses}/{self.config.risk.MAX_CONSECUTIVE_LOSSES}")
        else:
            self.consecutive_losses = 0
            print(f"[RISK] Trade closed: ${pnl:+.2f} | Daily PnL: ${self.daily_pnl:+.2f}")

    def calculate_lot_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        tick_value: float = 1.0,  # Dollar value of 1 point per 1.0 standard lot on Gold
    ) -> Tuple[float, str]:
        """
        Calculates ATR/Distance-based lot size risking strictly MAX_RISK_PER_TRADE_PCT (e.g. 1%).
        
        Lot Size = (Account Equity * Risk %) / (SL Distance in Points * Point Value)
        Clamped to Broker bounds [MIN_LOT, MAX_LOT] and rounded to LOT_STEP.
        
        Returns: (lot_size, warning_message) where warning may alert about over-risking.
        """
        risk_pct = self.config.risk.MAX_RISK_PER_TRADE_PCT
        risk_amount = account_equity * risk_pct

        sl_distance_price = abs(entry_price - stop_loss_price)
        # For standard Gold (100 oz contract), 1 point = $0.01 price change = $1.00 per standard lot
        # e.g., $5.00 SL distance = 500 points
        sl_points = sl_distance_price * 100.0

        if sl_points <= 0:
            return self.config.instrument.MIN_LOT, ""

        # Compute raw lots
        raw_lot = risk_amount / (sl_points * tick_value)

        # Round to nearest lot step (e.g. 0.01)
        lot_step = self.config.instrument.LOT_STEP
        clamped_lot = math.floor(raw_lot / lot_step) * lot_step

        # Clamp between min and max allowed
        final_lot = max(
            self.config.instrument.MIN_LOT,
            min(clamped_lot, self.config.instrument.MAX_LOT),
        )
        final_lot = round(final_lot, 2)

        # Calculate ACTUAL risk vs intended risk
        actual_risk = final_lot * sl_points * tick_value
        risk_ratio = actual_risk / risk_amount if risk_amount > 0 else 999

        warning = ""
        if final_lot == self.config.instrument.MIN_LOT and raw_lot < self.config.instrument.MIN_LOT:
            # Forced to use minimum lot because calculated lot is too small
            over_risk_pct = ((actual_risk - risk_amount) / risk_amount * 100) if risk_amount > 0 else 0
            warning = (f"[RISK WARNING] Minimum lot {self.config.instrument.MIN_LOT} forces "
                      f"{over_risk_pct:.0f}% over-risk! Actual: ${actual_risk:.2f} vs Budget: ${risk_amount:.2f}")
            print(warning)
        
        if account_equity < 300:
            warning_msg = (f"[RISK WARNING] Account ${account_equity:.0f} is below recommended $500 minimum. "
                          f"Spread costs may erode profitability.")
            if not warning:
                warning = warning_msg
            print(warning_msg)

        return final_lot, warning

    def validate_trade_entry(
        self,
        current_spread_points: float,
        current_equity: float,
        is_news_freeze: bool = False,
        model_confidence: float = 1.0,
        current_dt: Optional[datetime] = None,
        calendar_collector: Optional[object] = None,
    ) -> Tuple[bool, str]:
        """
        Runs comprehensive pre-flight trade checks:
        1. Spread anomaly check
        2. Daily drawdown check
        3. Dynamic event-specific macro freeze check (FOMC vs NFP vs CPI)
        4. Model confidence threshold check
        """
        # 1. Daily Drawdown Guard
        halted, dd_pct = self.check_daily_drawdown(current_equity)
        if halted or self.is_circuit_breaker_active:
            return False, f"Daily Drawdown Limit Exceeded ({dd_pct*100:.2f}%)"

        # 2. Spread Guard
        max_spread = self.config.risk.MAX_ALLOWED_SPREAD_POINTS
        if current_spread_points > max_spread:
            return False, f"Spread Too High ({current_spread_points:.1f} pts > {max_spread} pts limit)"

        # 3. Dynamic Macro News Circuit Breaker
        if calendar_collector is not None and current_dt is not None:
            is_frozen, reason, _ = calendar_collector.check_event_risk_window(current_dt)
            if is_frozen:
                return False, f"Macro Circuit Breaker: {reason}"
        elif is_news_freeze:
            return False, "High-Impact Macroeconomic News Freeze Active"

        # 4. Model Confidence Check
        min_conf = self.config.risk.MIN_CONFIDENCE_THRESHOLD
        if model_confidence < min_conf:
            return False, f"Confidence Below Threshold ({model_confidence:.2f} < {min_conf:.2f})"

        return True, "Approved"

    def compute_sl_tp_prices(
        self,
        action: str,
        entry_price: float,
        atr: float,
    ) -> Tuple[float, float]:
        """
        Calculates dynamic Stop-Loss and Take-Profit price levels based on current ATR.
        """
        tp_dist = atr * self.config.labeling.TP_ATR_MULTIPLIER
        sl_dist = atr * self.config.labeling.SL_ATR_MULTIPLIER

        if action == "BUY":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        elif action == "SELL":
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
        else:
            sl_price = entry_price
            tp_price = entry_price

        return round(sl_price, 2), round(tp_price, 2)
