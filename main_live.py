"""
Live / Demo Automated Trading Engine for Gold (XAUUSD) with Mamba Decision Brain.
Subscribes to live MT5 market feeds, calculates real-time multi-timeframe features,
runs Mamba inference, and executes trades with institutional risk boundaries.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import time
import joblib
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import torch

from config import CONFIG, BotConfig
from collectors.mt5_collector import MT5DataCollector
from features.pipeline import FeaturePipeline
from models.mamba_model import MambaTradingModel
from execution.risk_manager import InstitutionalRiskManager
from execution.order_router import MT5OrderRouter
from execution.account_setup import AccountSetupWizard, quick_setup


class LiveTradingEngine:
    """
    Real-time autonomous trading loop for MetaTrader 5.
    """

    def __init__(self, config: BotConfig = CONFIG):
        self.config = config
        self.collector = MT5DataCollector(config)
        self.router = MT5OrderRouter(config)
        self.risk_manager = InstitutionalRiskManager(config)
        self.pipeline = FeaturePipeline(config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MambaTradingModel] = None
        self.scaler = None
        self.pruner = None
        self.symbol: Optional[str] = None

    def initialize_system(self, user_config: dict = None) -> bool:
        """Connects to MT5 and loads saved model checkpoints and scalers."""
        print("\n" + "=" * 60)
        print("    🤖 INITIALIZING MAMBA GOLD TRADING ENGINE")
        print("=" * 60)

        # 0. Connect risk manager to order router for trade tracking
        self.router.set_risk_manager(self.risk_manager)

        # 1. Connect to MT5
        if not self.collector.connect():
            return False

        self.symbol = self.collector.resolve_symbol()
        if not self.symbol:
            print("[ERROR] Could not resolve Gold symbol.")
            return False

        acc = self.router.get_account_info()
        if acc:
            print(f"[ACCOUNT] Balance: ${acc['balance']:,.2f} | Equity: ${acc['equity']:,.2f}")
            # Initialize daily tracking
            self.risk_manager.reset_daily_tracking_if_new_day(acc['equity'], datetime.now(timezone.utc))
            
            # Show risk config
            risk_pct = self.config.risk.MAX_RISK_PER_TRADE_PCT * 100
            dd_pct = self.config.risk.MAX_DAILY_DRAWDOWN_PCT * 100
            print(f"[RISK] Max risk/trade: {risk_pct:.1f}% (${acc['equity'] * self.config.risk.MAX_RISK_PER_TRADE_PCT:.2f})")
            print(f"[RISK] Max daily loss: {dd_pct:.1f}% (${acc['equity'] * self.config.risk.MAX_DAILY_DRAWDOWN_PCT:.2f})")
            print(f"[RISK] Max lot size: {self.config.instrument.MAX_LOT}")

        # 2. Load Model Checkpoint & Scaler
        ckpt_path = self.config.paths.CHECKPOINTS_DIR / "mamba_xauusd_best.pt"
        scaler_path = self.config.paths.CHECKPOINTS_DIR / "feature_scaler.joblib"
        pruner_path = self.config.paths.PROCESSED_DATA_DIR / "correlation_pruner.joblib"

        if not ckpt_path.exists() or not scaler_path.exists():
            print(f"[ERROR] Model checkpoint or scaler missing in {self.config.paths.CHECKPOINTS_DIR}.")
            print("Please run training/train.py first before starting live trading.")
            return False

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        in_features = checkpoint["in_features"]

        self.model = MambaTradingModel(
            in_features=in_features,
            d_model=self.config.mamba.D_MODEL,
            n_layers=self.config.mamba.N_LAYERS,
            d_state=self.config.mamba.D_STATE,
            d_conv=self.config.mamba.D_CONV,
            expand=self.config.mamba.EXPAND_FACTOR,
            dropout=0.0,  # Zero dropout during live inference
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "vol_scaler" in checkpoint:
            self.model.vol_scaler = checkpoint["vol_scaler"]
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        if pruner_path.exists():
            self.pruner = joblib.load(pruner_path)

        print(f"[SUCCESS] Mamba Model Loaded from Epoch {checkpoint['epoch']} (In Features: {in_features})")
        return True

    def run_trading_step(self) -> None:
        """
        Executes a single real-time market iteration:
        - Ingests recent multi-timeframe candles.
        - Computes real-time feature sequence.
        - Generates Mamba model predictions.
        - Manages active positions / executes new approved setups.
        """
        now = datetime.now(timezone.utc)
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return

        current_spread = (tick.ask - tick.bid) * 100.0  # Spread in points
        acc = self.router.get_account_info()
        current_equity = acc["equity"] if acc else 10_000.0

        # Reset daily equity tracking
        self.risk_manager.reset_daily_tracking_if_new_day(current_equity, now)

        # 1. Fetch only the most recent N+buffer bars for live feature computation (NOT full history)
        # Fetch 600 bars to provide enough window for the slowest rolling indicator (EMA-200)
        RECENT_BARS_LIMIT = 600
        raw_data = {}
        if not self.collector.connect():
            return
        for tf in [self.config.instrument.PRIMARY_TIMEFRAME] + self.config.instrument.CONTEXT_TIMEFRAMES:
            import MetaTrader5 as _mt5
            from datetime import timedelta
            from collectors.mt5_collector import MT5DataCollector
            tf_const = self.collector.TIMEFRAME_LOOKUP.get(tf)
            if tf_const is None:
                continue
            rates = _mt5.copy_rates_from_pos(self.symbol, tf_const, 0, RECENT_BARS_LIMIT)
            if rates is not None and len(rates) > 0:
                import pandas as pd
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
                cols_keep = [c for c in ["open","high","low","close","tick_volume","spread","real_volume"] if c in df.columns]
                raw_data[tf] = df[cols_keep]

        if not raw_data or self.config.instrument.PRIMARY_TIMEFRAME not in raw_data:
            return

        # 2. Build aligned features (transform only)
        X_df, _ = self.pipeline.build_dataset(raw_data, is_training=False, save_to_disk=False)
        if len(X_df) < self.config.features.LOOKBACK_SEQUENCE_LENGTH:
            return

        # 3. Extract last N bars & Scale
        seq_len = self.config.features.LOOKBACK_SEQUENCE_LENGTH
        window_raw = X_df.iloc[-seq_len:].values
        window_scaled = self.scaler.transform(window_raw)

        tensor_x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. Mamba Model Prediction
        prediction = self.model.predict_trade_action(
            tensor_x, confidence_threshold=self.config.risk.MIN_CONFIDENCE_THRESHOLD
        )
        action = prediction["action"]
        confidence = prediction["confidence"]
        regime = prediction["regime_id"]

        # Approximate current ATR: mean of (High - Low) over last 14 bars
        primary_df = raw_data[self.config.instrument.PRIMARY_TIMEFRAME]
        if len(primary_df) >= 14:
            current_atr = float(
                (primary_df["high"].iloc[-14:] - primary_df["low"].iloc[-14:]).mean()
            )
        else:
            current_atr = 1.50

        # 5. Position Management: Trailing Stops & Breakeven
        self.router.update_trailing_stops_and_breakeven(self.symbol, current_atr=current_atr)

        # 6. Check for New Trade Entry
        open_positions = self.router.get_open_positions(self.symbol)

        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp_str}] Gold: ${tick.bid:.2f}/{tick.ask:.2f} | "
            f"Spread: {current_spread:.1f} pts | "
            f"Signal: {action} ({confidence*100:.1f}%) | "
            f"Open: {len(open_positions)}"
        )

        if len(open_positions) == 0 and action in ["BUY", "SELL"]:
            # Pre-flight institutional risk checks
            valid, reason = self.risk_manager.validate_trade_entry(
                current_spread_points=current_spread,
                current_equity=current_equity,
                model_confidence=confidence,
                current_dt=now,
                calendar_collector=self.pipeline.session_extractor.calendar,
            )

            if valid:
                entry_price = tick.ask if action == "BUY" else tick.bid
                sl_price, tp_price = self.risk_manager.compute_sl_tp_prices(
                    action=action, entry_price=entry_price, atr=current_atr
                )
                lot, risk_warning = self.risk_manager.calculate_lot_size(
                    account_equity=current_equity,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                )

                if risk_warning and "over-risk" in risk_warning.lower():
                    if current_equity < 200:
                        print(f"[REJECTED] {risk_warning}")
                        print(f"[REJECTED] Trade skipped. Consider funding account to at least $500.")
                        return

                print(f"[TRADE] Executing {action} {lot} lots @ {entry_price:.2f} (SL: {sl_price}, TP: {tp_price})")
                self.router.open_market_order(
                    symbol=self.symbol,
                    action=action,
                    lot_size=lot,
                    sl_price=sl_price,
                    tp_price=tp_price,
                )
            else:
                print(f"[REJECTED] {reason}")

    def start_loop(self, poll_interval_seconds: int = 15) -> None:
        """Runs the continuous live trading event loop."""
        if not self.initialize_system():
            return

        print(f"\n[INFO] Live Event Loop Active. Polling every {poll_interval_seconds}s. Press Ctrl+C to stop.")
        try:
            while True:
                self.run_trading_step()
                time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            print("\n[INFO] Shutdown signal received. Disconnecting...")
        finally:
            self.collector.shutdown()


def setup_and_run():
    """Main entry point: setup wizard + live trading."""
    print("\n" + "=" * 60)
    print("  🤖 MAMBA GOLD (XAUUSD) TRADING BOT")
    print("=" * 60)

    # Check for saved config
    wizard = AccountSetupWizard()
    saved_config = wizard.load_saved_setup()

    if saved_config:
        print(f"\n  Found saved config: ${saved_config['equity']:,.0f} account")
        print(f"  Risk: {saved_config['risk_pct']:.1f}%/trade | Daily: {saved_config['daily_dd_pct']:.1f}%")
        choice = input("\n  Use saved config? (y/n/r for reconfigure): ").strip().lower()
        if choice == "r":
            saved_config = wizard.run_interactive_setup()
        elif choice != "y":
            saved_config = {}
    else:
        # First time setup
        saved_config = wizard.run_interactive_setup()

    # If no config (user skipped), use safe defaults
    if not saved_config:
        print("\n  [INFO] No config set. Using defaults: $500 / 500 = 0.01 lot")
        saved_config = quick_setup(500, divisor=500, daily_dd_pct=5.0)

    # Start trading
    engine = LiveTradingEngine()
    engine.start_loop(poll_interval_seconds=15)


if __name__ == "__main__":
    setup_and_run()
