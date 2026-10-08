#!/usr/bin/env python
"""
deploy_bot.py — Deployable autonomous Mamba trading bot for Gold (XAUUSD).

One file to run the trained model against your MetaTrader 5 account:

  python deploy_bot.py --check            # verify artifacts + model + MT5 (NO orders)
  python deploy_bot.py --dry-run --once   # one full live signal cycle, no orders
  python deploy_bot.py --signal-export    # export signals for MQL5 EA, no orders
  python deploy_bot.py                    # run the bot (places real orders)

MQL5 BRIDGE (native EA that runs INSIDE the terminal):
  Run `python deploy_bot.py --signal-export` on this PC AND attach the native
  Expert Advisor [MambaGoldEA.mq5] to the same broker's XAUUSD M5 chart.
  Python computes the Mamba signal and writes mamba_signal.csv into the MT5
  Common files folder; the EA reads it, then places and manages orders
  natively (server-side SL/TP, trailing, breakeven, max-hold, reverse,
  daily drawdown halt, news freeze). NEVER run Python live mode and the EA
  at the same time (that would double every order).

What it does autonomously (no human in the loop):
  * connects to your logged-in MetaTrader 5 terminal/account
  * pulls live multi-timeframe candles (M5 decision + M15/H1/H4 context)
  * rebuilds the exact 85-feature matrix the model was trained on
  * runs Mamba inference -> BUY / SELL / HOLD with confidence + regime
  * pre-flight risk checks (spread, daily drawdown, news freeze, confidence)
  * opens market orders WITH server-side SL and TP (ATR-based, 1.5x ATR each)
  * manages open trades entirely by itself:
      - breakeven at 1R profit, trailing stop at 1.5x ATR (server-side modify)
      - force-closes after max holding time (default 24 M5 bars = 2 hours,
        matching the label horizon the model was trained on)
      - closes and REVERSES when the model signal flips against the position
      - closes everything when the daily drawdown circuit breaker trips
  * loops every --interval seconds, logging to console + logs/deploy_bot.log

Flags:
  --interval SEC      poll interval seconds (default 15)
  --confidence P      min signal confidence, override config (default config 0.50)
  --max-hold-bars N   force-close after N M5 bars (default: config = 24)
  --no-reverse        do not auto-flip the position when the signal reverses
  --dry-run           never send/modify/close orders; log what WOULD happen
  --signal-export      implies --dry-run; writes mamba_signal.csv each cycle
                       for MambaGoldEA.mq5 to execute (Python never trades)
  --signal-dir DIR     folder for the signal file (default: MT5 Common files)
  --once              run exactly one iteration then exit (use with --dry-run)

Exit codes:
  0  ready / run completed
  1  model OK but environment not ready (MetaTrader5 package or terminal missing)
  2  broken artifacts (checkpoint / scalers / feature mismatch)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

try:
    import MetaTrader5 as mt5
    MT5_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - environment dependent
    mt5 = None
    MT5_IMPORT_ERROR = str(_e)

from config import CONFIG
from features.pipeline import FeaturePipeline
from models.mamba_model import MambaTradingModel
from execution.risk_manager import InstitutionalRiskManager

if MT5_IMPORT_ERROR is None:
    from collectors.mt5_collector import MT5DataCollector
    from execution.order_router import MT5OrderRouter
else:  # pragma: no cover - environment dependent
    MT5DataCollector = None
    MT5OrderRouter = None

LOGGER = logging.getLogger("deploy_bot")

RECENT_BARS_LIMIT = 600      # enough window for slowest indicator (EMA-200)
M5_BAR_SECONDS = 300         # M5 bar duration for max-hold accounting
CONSECUTIVE_FAILURE_LIMIT = 10
SIGNAL_FILENAME = "mamba_signal.csv"   # bridge file read by MambaGoldEA.mq5


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    log_dir = CONFIG.paths.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    file_h = RotatingFileHandler(log_dir / "deploy_bot.log", maxBytes=5_000_000,
                                 backupCount=3, encoding="utf-8")
    file_h.setFormatter(fmt)
    LOGGER.addHandler(file_h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deployable autonomous Mamba XAUUSD trading bot")
    p.add_argument("--check", action="store_true",
                   help="verify artifacts, model, feature parity and MT5 status; no orders")
    p.add_argument("--dry-run", action="store_true",
                   help="never send/modify/close orders; only log what would happen")
    p.add_argument("--once", action="store_true",
                   help="run exactly one trading iteration then exit")
    p.add_argument("--interval", type=int, default=15, help="poll interval seconds (default 15)")
    p.add_argument("--confidence", type=float, default=None,
                   help="min signal confidence override (default: config MIN_CONFIDENCE_THRESHOLD)")
    p.add_argument("--max-hold-bars", type=int, default=None,
                   help="force-close after N M5 bars (default: config MAX_HOLDING_BARS = 24)")
    p.add_argument("--no-reverse", action="store_true",
                   help="keep the position until SL/TP/max-hold even if the signal flips")
    p.add_argument("--signal-export", action="store_true",
                   help="export mamba_signal.csv for MQL5 EA execution "
                        "(implies --dry-run: Python never sends orders)")
    p.add_argument("--signal-dir", default=None,
                   help="folder for the signal file (default: MT5 Common files dir)")
    p.add_argument("--verbose", action="store_true", help="debug-level logging")
    return p.parse_args()


# --------------------------------------------------------------------------
# Main bot
# --------------------------------------------------------------------------
class DeployBot:
    """Self-contained deployable trading engine built on the repo's components."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = CONFIG
        if args.signal_export:
            args.dry_run = True  # safety: the MQL5 EA (or nobody) executes, never Python

        if args.confidence is not None:
            self.cfg.risk.MIN_CONFIDENCE_THRESHOLD = float(args.confidence)
        self.min_confidence = float(self.cfg.risk.MIN_CONFIDENCE_THRESHOLD)
        self.max_hold_bars = int(
            args.max_hold_bars if args.max_hold_bars is not None
            else self.cfg.labeling.MAX_HOLDING_BARS
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[MambaTradingModel] = None
        self.scaler = None
        self.pruner = None
        self.in_features: Optional[int] = None
        self.checkpoint_meta: Dict[str, object] = {}

        self.pipeline: Optional[FeaturePipeline] = None
        self.risk = InstitutionalRiskManager(self.cfg)
        self.collector = None
        self.router = None
        self.symbol: Optional[str] = None

        self._fail_streak = 0
        self.stats = {"iterations": 0, "signals": 0, "orders": 0, "closes": 0}

    # --------------------------- artifact loading -------------------------
    def load_artifacts(self) -> None:
        """Load checkpoint + scalers + pruner. Raises FileNotFoundError/RuntimeError."""
        ckpt_path = self.cfg.paths.CHECKPOINTS_DIR / "mamba_xauusd_best.pt"
        scaler_path = self.cfg.paths.CHECKPOINTS_DIR / "feature_scaler.joblib"
        pruner_path = self.cfg.paths.PROCESSED_DATA_DIR / "correlation_pruner.joblib"

        missing = [str(p) for p in (ckpt_path, scaler_path, pruner_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model artifacts:\n  " + "\n  ".join(missing) +
                "\nRun pipeline_runner.py steps 1-3 (collect -> features -> train) first."
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.in_features = int(checkpoint["in_features"])
        self.checkpoint_meta = {
            "epoch": checkpoint.get("epoch"),
            "val_acc": checkpoint.get("val_acc"),
            "val_f1": checkpoint.get("val_f1"),
            "path": ckpt_path.name,
        }

        m = self.cfg.mamba
        self.model = MambaTradingModel(
            in_features=self.in_features,
            d_model=m.D_MODEL, n_layers=m.N_LAYERS, d_state=m.D_STATE,
            d_conv=m.D_CONV, expand=m.EXPAND_FACTOR, dropout=0.0,  # zero dropout at inference
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "vol_scaler" in checkpoint:
            self.model.vol_scaler = checkpoint["vol_scaler"]
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.pruner = joblib.load(pruner_path)

        # Consistency checks across the artifact set
        scaler_n = getattr(self.scaler, "n_features_in_", None)
        if scaler_n is not None and int(scaler_n) != self.in_features:
            raise RuntimeError(
                f"Feature scaler expects {scaler_n} features but checkpoint expects "
                f"{self.in_features}. Scaler and checkpoint are from different training runs."
            )
        pruner_n = len(self.pruner.selected_features_ or [])
        if pruner_n and pruner_n != self.in_features:
            raise RuntimeError(
                f"Correlation pruner selects {pruner_n} features but checkpoint expects "
                f"{self.in_features}. Re-run features/pipeline.py + training together."
            )

        n_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(
            "[ARTIFACTS] Loaded %s (epoch=%s, val_acc=%.4f, val_f1=%.4f) | %d features | "
            "%.2fM params | device=%s",
            ckpt_path.name, self.checkpoint_meta["epoch"],
            float(self.checkpoint_meta["val_acc"] or 0),
            float(self.checkpoint_meta["val_f1"] or 0),
            self.in_features, n_params / 1e6, self.device,
        )

    def get_pipeline(self) -> FeaturePipeline:
        if self.pipeline is None:
            self.pipeline = FeaturePipeline(self.cfg)
        return self.pipeline

    # ------------------------------ MT5 layer -----------------------------
    def ensure_mt5(self) -> bool:
        """Connect/reconnect to MT5 and resolve the gold symbol. Returns success."""
        if MT5_IMPORT_ERROR is not None:
            LOGGER.error("[MT5] MetaTrader5 package not installed: %s "
                         "(pip install MetaTrader5)", MT5_IMPORT_ERROR)
            return False
        if self.collector is None:
            self.collector = MT5DataCollector(self.cfg)
            self.router = MT5OrderRouter(self.cfg)
            self.router.set_risk_manager(self.risk)

        if mt5.terminal_info() is None and not self.collector.connect():
            LOGGER.error("[MT5] Terminal unreachable. Start MetaTrader 5 and log in. (%s)",
                         mt5.last_error())
            return False

        if self.symbol is None:
            self.symbol = self.collector.resolve_symbol()
            if not self.symbol:
                LOGGER.error("[MT5] Could not resolve a Gold symbol on this broker.")
                return False
            LOGGER.info("[MT5] Trading symbol resolved: %s", self.symbol)
        return True

    def fetch_recent_bars(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch the latest RECENT_BARS_LIMIT bars for M5 + context timeframes."""
        raw_data: Dict[str, pd.DataFrame] = {}
        tfs = [self.cfg.instrument.PRIMARY_TIMEFRAME] + list(self.cfg.instrument.CONTEXT_TIMEFRAMES)
        for tf in tfs:
            tf_const = self.collector.TIMEFRAME_LOOKUP.get(tf)
            if tf_const is None:
                continue
            rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, RECENT_BARS_LIMIT)
            if rates is None or len(rates) == 0:
                LOGGER.warning("[DATA] No %s bars returned (%s)", tf, mt5.last_error())
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            keep = [c for c in ["open", "high", "low", "close",
                                "tick_volume", "spread", "real_volume"] if c in df.columns]
            raw_data[tf] = df[keep]
        if self.cfg.instrument.PRIMARY_TIMEFRAME not in raw_data:
            return None
        return raw_data

    # -------------------------- signal bridge -----------------------------
    def _signal_targets(self):
        """Folders that receive mamba_signal.csv: the MT5 files folder the EA
        reads (Common by default) plus logs/ as a debug copy."""
        targets = []
        if self.args.signal_dir:
            targets.append(Path(self.args.signal_dir) / SIGNAL_FILENAME)
        else:
            candidates = []
            env = os.environ.get("MT5_COMMON_FILES")
            if env:
                candidates.append(Path(env))
            appdata = os.environ.get("APPDATA")
            if appdata:
                candidates.append(Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files")
                candidates.append(Path(appdata) / "MetaTrader 5" / "Common" / "Files")
            for c in candidates:
                try:
                    if c.is_dir():
                        targets.append(c / SIGNAL_FILENAME)
                        break
                except OSError:
                    continue
        targets.append(self.cfg.paths.LOGS_DIR / SIGNAL_FILENAME)
        return targets

    def write_signal_file(self, pred: Dict[str, object], atr: float,
                          news_frozen: bool) -> list:
        """Atomically write the key=value bridge file for MambaGoldEA.mq5."""
        now = datetime.now(timezone.utc)
        lines = [
            "version=1",
            f"epoch={int(now.timestamp())}",
            f"time_utc={now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"symbol={self.symbol or ''}",
            f"action={pred['action']}",
            f"confidence={float(pred['confidence']):.4f}",
            f"p_buy={float(pred['p_buy']):.4f}",
            f"p_sell={float(pred['p_sell']):.4f}",
            f"p_neutral={float(pred['p_neutral']):.4f}",
            f"regime={int(pred['regime_id'])}",
            f"atr={float(atr):.5f}",
            f"news_frozen={1 if news_frozen else 0}",
            f"min_conf={self.min_confidence:.2f}",
            "source=deploy_bot.py",
            "",
        ]
        payload = "\n".join(lines)  # pure ASCII: EA reads with FILE_ANSI
        written = []
        for t in self._signal_targets():
            try:
                t.parent.mkdir(parents=True, exist_ok=True)
                tmp = t.with_name(t.name + ".tmp")
                tmp.write_text(payload, encoding="ascii")
                os.replace(tmp, t)  # atomic: EA never sees a half-written file
                written.append(str(t))
            except OSError as e:
                LOGGER.warning("[SIGNAL] could not write %s: %s", t, e)
        return written

    def _export_signal(self, pred: Dict[str, object], atr: float, now: datetime) -> None:
        try:
            cal = self.get_pipeline().session_extractor.calendar
            frozen, reason, _ = cal.check_event_risk_window(now)
            targets = self.write_signal_file(pred, atr, bool(frozen))
            if targets:
                LOGGER.info("[SIGNAL] exported %s (%.0f%%) -> %s%s",
                            pred["action"], float(pred["confidence"]) * 100,
                            targets[0], f" | NEWS FREEZE: {reason}" if frozen else "")
            else:
                LOGGER.warning("[SIGNAL] no writable target for %s", SIGNAL_FILENAME)
        except Exception as e:
            LOGGER.warning("[SIGNAL] export failed: %s", e)

    def _signal_export_check(self) -> Tuple[str, str]:
        """Write a probe signal (forced action=HOLD, can never trigger a trade)
        and verify the bridge file content. Returns (level, message)."""
        pred = getattr(self, "last_parity_pred", None)
        atr = getattr(self, "last_parity_atr", 2.0)
        if pred is None:
            return "fail", "no prediction available for the signal export test"
        probe = dict(pred)
        probe["action"] = "HOLD"
        targets = self.write_signal_file(probe, atr, False)
        if not targets:
            return "fail", f"no writable target for {SIGNAL_FILENAME}"
        try:
            txt = Path(targets[0]).read_text(encoding="ascii")
        except OSError as e:
            return "fail", f"cannot read back {targets[0]}: {e}"
        need = {"version", "epoch", "time_utc", "symbol", "action", "confidence",
                "atr", "news_frozen", "min_conf"}
        keys = {ln.split("=", 1)[0] for ln in txt.splitlines() if "=" in ln}
        missing = sorted(need - keys)
        if missing:
            return "fail", f"signal file missing keys: {missing}"
        in_mt5_dir = any("MetaQuotes" in t or "MetaTrader" in t for t in targets)
        if not in_mt5_dir and not self.args.signal_dir:
            return "warn", (f"wrote {targets[0]} but the MT5 Common files folder was "
                            "not found — the EA reads Common. Start the MT5 terminal "
                            "once (creates it) or pass --signal-dir pointing at an "
                            "MT5 Files folder")
        return "pass", f"{targets[0]} (probe action=HOLD, {len(keys)} keys)"

    # ---------------------------- model layer -----------------------------
    def build_features(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Rebuild the exact trained feature matrix from raw live bars."""
        X_df, _ = self.get_pipeline().build_dataset(
            raw_data, is_training=False, save_to_disk=False
        )
        if X_df.shape[1] != self.in_features:
            expected = set(self.pruner.selected_features_)
            got = set(X_df.columns)
            raise RuntimeError(
                f"Feature mismatch: built {X_df.shape[1]} columns, model expects "
                f"{self.in_features}. Missing={sorted(expected - got)} "
                f"Unexpected={sorted(got - expected)}"
            )
        return X_df

    @torch.no_grad()
    def predict(self, X_df: pd.DataFrame) -> Optional[Dict[str, object]]:
        seq_len = self.cfg.features.LOOKBACK_SEQUENCE_LENGTH
        if len(X_df) < seq_len:
            LOGGER.warning("[MODEL] Only %d feature bars (< %d); skipping this tick.",
                           len(X_df), seq_len)
            return None
        window = X_df.iloc[-seq_len:].values
        window_scaled = self.scaler.transform(window)
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.model.predict_trade_action(x, confidence_threshold=self.min_confidence)

    @staticmethod
    def estimate_atr(primary_df: pd.DataFrame) -> float:
        if len(primary_df) >= 14:
            return float((primary_df["high"].iloc[-14:] - primary_df["low"].iloc[-14:]).mean())
        return 1.50

    # -------------------------- trade management --------------------------
    def _close_position(self, ticket: int, reason: str) -> bool:
        if self.args.dry_run:
            LOGGER.info("[DRY-RUN] would close #%s — %s", ticket, reason)
            return False
        ok = self.router.close_position(ticket)
        if ok:
            self.stats["closes"] += 1
            LOGGER.info("[CLOSED] #%s — %s", ticket, reason)
        else:
            LOGGER.warning("[WARN] Failed to close #%s (%s)", ticket, reason)
        return ok

    def manage_positions(self, pred: Dict[str, object], atr: float,
                         positions: list, tick) -> int:
        """Breakeven/trailing, max-hold force-close, reverse-signal close.

        Returns number of positions closed this tick.
        """
        # 1. Breakeven + trailing stop (server-side SL modification)
        if not self.args.dry_run:
            self.router.update_trailing_stops_and_breakeven(self.symbol, current_atr=atr)

        closed = 0
        server_now = int(tick.time)  # same clock as position open time (server epoch)
        hold_limit_sec = self.max_hold_bars * M5_BAR_SECONDS
        action = pred["action"]
        conf = float(pred["confidence"])

        for p in positions:
            age_sec = max(0, server_now - int(p.time))

            # 2. Max-holding force close (matches the label horizon the model learned)
            if age_sec >= hold_limit_sec:
                if self._close_position(
                        p.ticket,
                        f"max-hold {age_sec // 60} min >= {hold_limit_sec // 60} min"):
                    closed += 1
                continue

            # 3. Reverse-signal close (model flipped against the open position)
            if self.args.no_reverse or action not in ("BUY", "SELL"):
                continue
            side = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            if action != side and conf >= self.min_confidence:
                if self._close_position(
                        p.ticket,
                        f"model reversed {side} -> {action} @ {conf * 100:.1f}%"):
                    closed += 1
        return closed

    def maybe_enter(self, pred: Dict[str, object], atr: float, spread_pts: float,
                    equity: float, now: datetime) -> None:
        action = pred["action"]
        if action not in ("BUY", "SELL"):
            return
        conf = float(pred["confidence"])

        valid, reason = self.risk.validate_trade_entry(
            current_spread_points=spread_pts,
            current_equity=equity,
            model_confidence=conf,
            current_dt=now,
            calendar_collector=self.get_pipeline().session_extractor.calendar,
        )
        if not valid:
            LOGGER.info("[REJECTED] %s %s @ %.1f%% — %s",
                        action, self.symbol, conf * 100, reason)
            return

        tick = mt5.symbol_info_tick(self.symbol)
        entry = tick.ask if action == "BUY" else tick.bid
        sl_price, tp_price = self.risk.compute_sl_tp_prices(action, entry, atr)
        lot, warning = self.risk.calculate_lot_size(
            account_equity=equity, entry_price=entry, stop_loss_price=sl_price
        )
        if warning and "over-risk" in warning.lower() and equity < 200:
            LOGGER.warning("[REJECTED] %s (equity $%.0f — fund to >= $500)",
                           warning, equity)
            return

        if self.args.dry_run:
            LOGGER.info("[DRY-RUN] would OPEN %s %.2f lots %s @ %.2f | SL %.2f | TP %.2f | "
                        "conf %.1f%% | ATR %.2f",
                        action, lot, self.symbol, entry, sl_price, tp_price, conf * 100, atr)
            return

        ticket = self.router.open_market_order(
            symbol=self.symbol, action=action, lot_size=lot,
            sl_price=sl_price, tp_price=tp_price,
        )
        if ticket is not None:
            self.stats["orders"] += 1
            LOGGER.info("[TRADE] OPENED #%s %s %.2f lots @ %.2f (SL %.2f / TP %.2f, "
                        "conf %.1f%%)", ticket, action, lot, entry, sl_price, tp_price,
                        conf * 100)

    # ------------------------------ main tick -----------------------------
    def run_iteration(self) -> bool:
        """One full market iteration. Returns False on a fatal/infrastructure failure."""
        self.stats["iterations"] += 1
        if not self.ensure_mt5():
            return False

        now = datetime.now(timezone.utc)
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            LOGGER.warning("[MT5] No live quotes for %s yet (market closed or feed warming); "
                           "retrying next tick", self.symbol)
            return True  # not fatal — weekend/off-hours/starting terminal

        acc = mt5.account_info()
        if acc is None:
            LOGGER.error("[MT5] account_info failed: %s", mt5.last_error())
            return False
        equity = float(acc.equity)
        self.risk.reset_daily_tracking_if_new_day(equity, now)
        spread_pts = (tick.ask - tick.bid) * 100.0

        raw_data = self.fetch_recent_bars()
        if raw_data is None:
            LOGGER.warning("[DATA] primary timeframe bars unavailable; retrying next tick")
            return True

        X_df = self.build_features(raw_data)
        pred = self.predict(X_df)
        if pred is None:
            return True

        atr = self.estimate_atr(raw_data[self.cfg.instrument.PRIMARY_TIMEFRAME])

        # --- signal bridge export (MQL5 EA mode / dry-run diagnostics) ---
        if self.args.signal_export or self.args.dry_run:
            self._export_signal(pred, atr, now)

        positions = self.router.get_open_positions(self.symbol)

        # --- position management ----------------------------------------
        if positions:
            halted, dd_pct = self.risk.check_daily_drawdown(equity)
            if halted or self.risk.is_circuit_breaker_active:
                LOGGER.warning("[CIRCUIT-BREAKER] daily drawdown %.2f%% — closing all "
                               "positions and halting new entries", dd_pct * 100)
                for p in list(positions):
                    self._close_position(p.ticket, "daily circuit breaker")
                positions = []
            else:
                n_closed = self.manage_positions(pred, atr, positions, tick)
                if n_closed:
                    positions = self.router.get_open_positions(self.symbol)

        # --- new entry ----------------------------------------------------
        if not positions:
            self.maybe_enter(pred, atr, spread_pts, equity, now)
        elif not self.args.dry_run:
            pass  # holding an open trade; SL/TP/trailing already manage it

        # --- status line ---------------------------------------------------
        self.stats["signals"] += 1
        LOGGER.info(
            "Gold %.2f/%.2f spread %.1f pts | signal %s %.1f%% | p(B/S/N) %.2f/%.2f/%.2f | "
            "regime %d | ATR %.2f | open %d | equity $%.2f%s",
            tick.bid, tick.ask, spread_pts, pred["action"], float(pred["confidence"]) * 100,
            float(pred["p_buy"]), float(pred["p_sell"]), float(pred["p_neutral"]),
            int(pred["regime_id"]), atr, len(positions), equity,
            " | DRY-RUN" if self.args.dry_run else "",
        )
        return True

    def run_loop(self) -> bool:
        """Run the trading loop. Returns True if the last iteration succeeded."""
        if self.args.signal_export:
            mode = "SIGNAL-EXPORT (Python writes mamba_signal.csv; MQL5 EA executes)"
        elif self.args.dry_run:
            mode = "DRY-RUN"
        else:
            mode = "LIVE (real orders)"
        LOGGER.info("=" * 62)
        LOGGER.info("  MAMBA GOLD BOT — DEPLOY MODE: %s", mode)
        LOGGER.info("  Confidence >= %.2f | Max hold %d bars (%.0f min) | Interval %ds",
                    self.min_confidence, self.max_hold_bars,
                    self.max_hold_bars * 5, self.args.interval)
        LOGGER.info("  Risk: %.1f%%/trade | daily stop %.1f%% | reverse %s",
                    self.cfg.risk.MAX_RISK_PER_TRADE_PCT * 100,
                    self.cfg.risk.MAX_DAILY_DRAWDOWN_PCT * 100,
                    "OFF" if self.args.no_reverse else "ON")
        LOGGER.info("=" * 62)

        last_ok = False
        try:
            while True:
                t0 = time.time()
                try:
                    ok = self.run_iteration()
                    self._fail_streak = 0 if ok else self._fail_streak + 1
                except Exception:
                    ok = False
                    self._fail_streak += 1
                    LOGGER.error("[ERROR] iteration failed (%d/%d):\n%s",
                                 self._fail_streak, CONSECUTIVE_FAILURE_LIMIT,
                                 traceback.format_exc())
                last_ok = ok
                if self._fail_streak >= CONSECUTIVE_FAILURE_LIMIT:
                    LOGGER.error("[FATAL] %d consecutive failures — shutting down.",
                                 self._fail_streak)
                    break
                if self.args.once:
                    break
                elapsed = time.time() - t0
                time.sleep(max(1.0, self.args.interval - elapsed))
        except KeyboardInterrupt:
            last_ok = True
            LOGGER.info("\n[SHUTDOWN] Ctrl+C received. Open positions keep their "
                        "server-side SL/TP. Disconnecting MT5.")
        finally:
            if mt5 is not None and mt5.terminal_info() is not None:
                mt5.shutdown()
        return last_ok

    # ------------------------------ self check ----------------------------
    def offline_parity_check(self) -> Tuple[bool, str]:
        """Build features from the most recent raw bars and run one inference.

        Proves the whole model chain (features -> pruner -> scaler -> Mamba)
        works end-to-end without sending any orders.
        """
        raw_dir = self.cfg.paths.RAW_DATA_DIR
        tfs = [self.cfg.instrument.PRIMARY_TIMEFRAME] + list(self.cfg.instrument.CONTEXT_TIMEFRAMES)
        raw_data: Dict[str, pd.DataFrame] = {}
        for tf in tfs:
            files = sorted(raw_dir.glob(f"*_{tf}_raw.parquet")) or \
                    sorted(raw_dir.glob(f"*_{tf}_raw.csv"))
            if not files:
                return False, f"no raw {tf} file in {raw_dir}"
            f = files[-1]
            df = pd.read_parquet(f) if f.suffix == ".parquet" else \
                pd.read_csv(f, index_col=0, parse_dates=True)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
            keep = [c for c in ["open", "high", "low", "close",
                                "tick_volume", "spread", "real_volume"] if c in df.columns]
            raw_data[tf] = df[keep].tail(RECENT_BARS_LIMIT)

        X_df = self.build_features(raw_data)
        pred = self.predict(X_df)
        if pred is None:
            return False, "not enough feature bars for a full lookback window"
        seq_len = self.cfg.features.LOOKBACK_SEQUENCE_LENGTH
        last_ts = X_df.index[-1]
        # Stash for the --check signal-bridge test
        self.last_parity_pred = pred
        self.last_parity_atr = self.estimate_atr(raw_data[self.cfg.instrument.PRIMARY_TIMEFRAME])
        return True, (
            f"features {X_df.shape[0]}x{X_df.shape[1]} (last bar {last_ts}), "
            f"window {seq_len} bars -> signal {pred['action']} "
            f"conf {float(pred['confidence']) * 100:.1f}% "
            f"p(B/S/N) {float(pred['p_buy']):.2f}/{float(pred['p_sell']):.2f}/"
            f"{float(pred['p_neutral']):.2f} regime {int(pred['regime_id'])}"
        )

    def check(self) -> int:
        """Full readiness check. Returns process exit code (0 ready / 1 env / 2 artifacts)."""
        LOGGER.info("=" * 62)
        LOGGER.info("  DEPLOY READINESS CHECK (no orders are ever sent by --check)")
        LOGGER.info("=" * 62)
        artifact_ok = True

        # 1. Artifacts + model forward pass
        try:
            self.load_artifacts()
        except Exception as e:
            artifact_ok = False
            LOGGER.error("[FAIL] artifacts: %s", e)

        # 2. Feature parity + inference on recent real data
        if artifact_ok:
            try:
                ok, msg = self.offline_parity_check()
                if ok:
                    LOGGER.info("[PASS] inference chain: %s", msg)
                else:
                    artifact_ok = False
                    LOGGER.error("[FAIL] inference chain: %s", msg)
            except Exception as e:
                artifact_ok = False
                LOGGER.error("[FAIL] inference chain: %s\n%s", e, traceback.format_exc())

            # 2b. MQL5 signal-bridge write test (writes an action=HOLD probe,
            #     which can never trigger a trade even with the EA attached)
            level, msg = self._signal_export_check()
            if level == "pass":
                LOGGER.info("[PASS] signal bridge: %s", msg)
            elif level == "warn":
                LOGGER.warning("[WARN] signal bridge: %s", msg)
            else:
                artifact_ok = False
                LOGGER.error("[FAIL] signal bridge: %s", msg)

        # 3. Economic calendar coverage (news freeze + news-distance features)
        cal_path = self.cfg.paths.DATA_DIR / "economic_calendar.csv"
        if cal_path.exists():
            try:
                cal = pd.read_csv(cal_path)
                cal["time"] = pd.to_datetime(cal["time"], utc=True, errors="coerce")
                now_utc = pd.Timestamp.now(tz="UTC")
                horizon = cal["time"].max()
                future_n = int((cal["time"] > now_utc).sum())
                if pd.isna(horizon) or horizon < now_utc:
                    LOGGER.warning("[WARN] economic calendar has no future events (last: %s) — "
                                   "news freeze windows cannot fire; refresh the calendar", horizon)
                else:
                    LOGGER.info("[PASS] economic calendar covers through %s (%d future events)",
                                horizon.date(), future_n)
            except Exception as e:
                LOGGER.warning("[WARN] could not read economic calendar: %s", e)
        else:
            LOGGER.warning("[WARN] no economic_calendar.csv — news filter disabled")

        # 4. MetaTrader5 package + terminal + account
        env_ok = True
        if MT5_IMPORT_ERROR is not None:
            env_ok = False
            LOGGER.error("[FAIL] MetaTrader5 package missing: %s (pip install MetaTrader5)",
                         MT5_IMPORT_ERROR)
        else:
            if not self.ensure_mt5():
                env_ok = False
            else:
                acc = mt5.account_info()
                tick = mt5.symbol_info_tick(self.symbol)
                if acc is None:
                    env_ok = False
                    LOGGER.error("[FAIL] MT5 terminal reachable but no account info — "
                                 "log in to your broker account")
                else:
                    mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(acc.trade_mode, "?")
                    LOGGER.info("[PASS] account #%s on %s | type=%s | balance $%.2f | "
                                "leverage 1:%d", acc.login, acc.server, mode,
                                acc.balance, acc.leverage)
                    if mode == "REAL":
                        LOGGER.warning("[WARN] this is a REAL money account and the model "
                                       "is underperforming OOS — consider starting on DEMO")
                    if tick is not None:
                        spread_pts = (tick.ask - tick.bid) * 100.0
                        if tick.bid > 0 and tick.ask > 0:
                            LOGGER.info("[PASS] %s tick %.2f/%.2f spread %.1f pts (limit %.0f)",
                                        self.symbol, tick.bid, tick.ask, spread_pts,
                                        self.cfg.risk.MAX_ALLOWED_SPREAD_POINTS)
                        else:
                            LOGGER.warning("[WARN] %s quotes are 0.00 — market closed or feed "
                                           "not streaming yet; the bot will idle until live "
                                           "ticks arrive (re-run --check in market hours)",
                                           self.symbol)
                    else:
                        env_ok = False
                        LOGGER.warning("[FAIL] no tick for %s — market closed or symbol "
                                       "not visible (add it to Market Watch)", self.symbol)
                mt5.shutdown()

        LOGGER.info("=" * 62)
        if not artifact_ok:
            LOGGER.error("VERDICT: NOT READY — fix artifact/model failures above (exit 2)")
            return 2
        if not env_ok:
            LOGGER.error("VERDICT: MODEL READY, ENVIRONMENT NOT READY (exit 1)")
            return 1
        LOGGER.info("VERDICT: READY — run `python deploy_bot.py --dry-run --once` first, "
                    "then `python deploy_bot.py` (exit 0)")
        return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    bot = DeployBot(args)

    if args.check:
        return bot.check()

    # Live run: artifacts are mandatory before anything touches the account
    try:
        bot.load_artifacts()
    except Exception as e:
        LOGGER.error("[FATAL] %s", e)
        return 2

    if MT5_IMPORT_ERROR is not None:
        LOGGER.error("[FATAL] MetaTrader5 package missing: %s\n"
                     "  pip install MetaTrader5   (Windows only, with MT5 terminal running)",
                     MT5_IMPORT_ERROR)
        return 1

    ok, msg = bot.offline_parity_check()
    if ok:
        LOGGER.info("[PRE-FLIGHT] inference chain OK — %s", msg)
    else:
        LOGGER.warning("[PRE-FLIGHT] offline inference check failed: %s "
                       "(continuing; live bars will be fetched from MT5)", msg)

    run_ok = bot.run_loop()
    LOGGER.info("[STATS] iterations=%d signals=%d orders=%d closes=%d",
                bot.stats["iterations"], bot.stats["signals"],
                bot.stats["orders"], bot.stats["closes"])
    return 0 if run_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
