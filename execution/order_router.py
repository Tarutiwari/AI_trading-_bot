"""
MetaTrader 5 Order Execution & Position Management Router.
Dispatches market orders with hard server-side SL/TP, manages trailing stops,
and handles breakeven modifications.
"""

from typing import Dict, List, Optional, Tuple
import time
import sys
from pathlib import Path

import MetaTrader5 as mt5

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig


class MT5OrderRouter:
    """
    Handles live order dispatch, modification, and position tracking via MT5 API.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.magic = config.risk.MAGIC_NUMBER
        self.comment = config.risk.ORDER_COMMENT
        self.risk_manager = None  # Set by LiveTradingEngine to track trade results

    def set_risk_manager(self, risk_manager):
        """Connect risk manager for trade result tracking."""
        self.risk_manager = risk_manager

    def get_account_info(self) -> Optional[Dict[str, float]]:
        """Fetch current account balance, equity, and margin."""
        acc = mt5.account_info()
        if acc is None:
            print(f"[ERROR] Failed to get MT5 account info: {mt5.last_error()}")
            return None
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin_free": acc.margin_free,
            "profit": acc.profit,
        }

    def get_open_positions(self, symbol: str) -> List[any]:
        """Fetch all active positions opened by this bot for the specified symbol."""
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        # Filter strictly by bot magic number
        return [p for p in positions if p.magic == self.magic]

    def open_market_order(
        self,
        symbol: str,
        action: str,  # 'BUY' or 'SELL'
        lot_size: float,
        sl_price: float,
        tp_price: float,
    ) -> Optional[int]:
        """
        Dispatches a market order with server-side SL and TP.
        Returns ticket number if successful, None otherwise.
        """
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[ERROR] Could not get tick for {symbol}")
            return None

        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": price,
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20,
            "magic": self.magic,
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Check order filling mode compatibility (some brokers use FOK or RETURN)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is not None:
            filling = symbol_info.filling_mode
            if filling == 1:  # FOK
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif filling == 2:  # IOC
                request["type_filling"] = mt5.ORDER_FILLING_IOC
            else:
                request["type_filling"] = mt5.ORDER_FILLING_RETURN

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[ERROR] Order Send Failed! Retcode: {result.retcode}, Comment: {result.comment}")
            return None

        print(
            f"[EXECUTED] {action} {lot_size} lots of {symbol} @ {price:.2f} | "
            f"SL: {sl_price:.2f}, TP: {tp_price:.2f} | Ticket: #{result.order}"
        )
        return result.order

    def modify_position_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modifies Stop-Loss and Take-Profit for an existing open position."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(new_sl),
            "tp": float(new_tp),
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"[WARN] Failed to modify #{ticket}: {result.comment}")
            return False

        print(f"[MODIFIED] Ticket #{ticket} SL updated to {new_sl:.2f}, TP: {new_tp:.2f}")
        return True

    def close_position(self, ticket: int) -> bool:
        """Closes an open position at current market price."""
        pos = mt5.positions_get(ticket=ticket)
        if not pos or len(pos) == 0:
            return False

        p = pos[0]
        close_action = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask

        # Get PnL before closing
        pnl = p.profit
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": close_action,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"{self.comment}_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,  # Required by most brokers
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Record trade result for risk management
            if self.risk_manager:
                self.risk_manager.record_trade_result(pnl)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def update_trailing_stops_and_breakeven(
        self, symbol: str, current_atr: float
    ) -> None:
        """
        Manages open positions:
        1. Moves SL to Breakeven once 1.0R profit is reached.
        2. Trails SL dynamically by trailing_atr_mult behind market price.
        """
        positions = self.get_open_positions(symbol)
        if not positions:
            return

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return

        for p in positions:
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            current_price = tick.bid if is_buy else tick.ask
            open_price = p.price_open
            current_sl = p.sl
            current_tp = p.tp

            # 1. Breakeven Check (1R profit)
            initial_risk = abs(open_price - current_sl) if current_sl > 0 else (current_atr * 1.5)
            if is_buy:
                profit_dist = current_price - open_price
                if profit_dist >= initial_risk and current_sl < open_price:
                    # Move SL to Entry + 10 points buffer
                    new_sl = round(open_price + 0.10, 2)
                    self.modify_position_sl_tp(p.ticket, new_sl, current_tp)

                # 2. Dynamic Trailing Stop
                if self.config.risk.ENABLE_TRAILING_STOP and current_sl >= open_price:
                    trailing_dist = current_atr * self.config.risk.TRAILING_ATR_MULTIPLE
                    proposed_sl = round(current_price - trailing_dist, 2)
                    if proposed_sl > current_sl:
                        self.modify_position_sl_tp(p.ticket, proposed_sl, current_tp)

            else:  # Sell Position
                profit_dist = open_price - current_price
                if profit_dist >= initial_risk and (current_sl > open_price or current_sl == 0):
                    new_sl = round(open_price - 0.10, 2)
                    self.modify_position_sl_tp(p.ticket, new_sl, current_tp)

                if self.config.risk.ENABLE_TRAILING_STOP and current_sl > 0 and current_sl <= open_price:
                    trailing_dist = current_atr * self.config.risk.TRAILING_ATR_MULTIPLE
                    proposed_sl = round(current_price + trailing_dist, 2)
                    if proposed_sl < current_sl:
                        self.modify_position_sl_tp(p.ticket, proposed_sl, current_tp)
