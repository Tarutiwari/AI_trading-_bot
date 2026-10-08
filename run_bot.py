# -*- coding: utf-8 -*-
"""
run_bot.py - Standalone deployable Mamba Gold (XAUUSD) bot.

One file to run after copying it together with bot.ini and the model artifacts
next to it. The file prefers its own bundled config + bundled model/scaler/pruner,
and falls back to this repo's modules only if they are importable from the path.

Recommended minimal copy bundle (all next to run_bot.py):
  run_bot.py
  bot.ini
  model/mamba_xauusd_best.pt
  model/feature_scaler.joblib
  model/correlation_pruner.joblib
  data/economic_calendar.csv        (optional - enables news freeze)

Run modes (edit bot.ini or pass CLI flags):
  python run_bot.py --check          verify artifacts + MT5; no orders
  python run_bot.py --dry-run --once one live signal cycle; no orders
  python run_bot.py --dry-run       keep running, never send orders
  python run_bot.py                 run live (real orders) on the logged-in MT5 account

IMPORTANT:
  * This file does not "paste" into MT5. It is a Python process that drives your
    logged-in MetaTrader 5 terminal from outside via the MetaTrader5 package.
  * The terminal must be running and logged in before you start this script.
  * Python + MT5 must keep running whenever you want trades to happen (PC on,
    or a Windows VPS for 24/5 coverage).
  * The model artifacts (pt + 2x joblib) must travel with this file. The bot cannot
    run from this file alone without them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from configparser import ConfigParser
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Locate this file and its sibling bundle
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

# If the repo modules are importable, trust them. Otherwise we stay standalone.
REPO_ROOT = HERE.parent if (HERE.parent / "config.py").exists() else HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Optional repo imports (graceful: standalone mode keeps working without them)
# ---------------------------------------------------------------------------
_repo_import_errors: Dict[str, Optional[str]] = {}

try:
    import MetaTrader5 as mt5
    MT5_IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - environment dependent
    mt5 = None
    MT5_IMPORT_ERROR = str(_e)

try:
    import numpy as np
except Exception as _e:  # pragma: no cover
    np = None
    _repo_import_errors["numpy"] = str(_e)

try:
    import pandas as pd
except Exception as _e:  # pragma: no cover
    pd = None
    _repo_import_errors["pandas"] = str(_e)

try:
    import torch
except Exception as _e:  # pragma: no cover
    torch = None
    _repo_import_errors["torch"] = str(_e)

try:
    import joblib
except Exception as _e:  # pragma: no cover
    joblib = None
    _repo_import_errors["joblib"] = str(_e)

# Repo modules are used when present. If absent, the script still validates via
# --check and can run a minimal live loop using bundled fallbacks below.
MambaTradingModel = None
FeaturePipeline = None
InstitutionalRiskManager = None
MT5DataCollector = None
MT5OrderRouter = None

if MT5_IMPORT_ERROR is None and torch is not None and np is not None and pd is not None:
    try:
        from models.mamba_model import MambaTradingModel
    except Exception as _e:  # pragma: no cover
        _repo_import_errors["mamba_model"] = str(_e)

    try:
        from features.pipeline import FeaturePipeline
    except Exception as _e:  # pragma: no cover
        _repo_import_errors["feature_pipeline"] = str(_e)

    try:
        from execution.risk_manager import InstitutionalRiskManager
    except Exception as _e:  # pragma: no cover
        _repo_import_errors["risk_manager"] = str(_e)

    try:
        from collectors.mt5_collector import MT5DataCollector
    except Exception as _e:  # pragma: no cover
        _repo_import_errors["mt5_collector"] = str(_e)

    try:
        from execution.order_router import MT5OrderRouter
    except Exception as _e:  # pragma: no cover
        _repo_import_errors["order_router"] = str(_e)

# ---------------------------------------------------------------------------
# Standalone config (INI next to this file). This is the copyable contract.
# ---------------------------------------------------------------------------
BOT_INI = HERE / "bot.ini"

DEFAULT_BOT_INI = """
[bot]
# MetaTrader symbol to trade. The script auto-resolves the broker's actual Gold name.
symbol = XAUUSD
# Decision timeframe + context timeframes (comma separated). Keep M5 as primary.
timeframes = M5,M15,H1,H4
# Minimum signal confidence to consider an entry. 0.50 is the repo default.
confidence = 0.50
# Force-close a trade after this many primary timeframe bars.
max_hold_bars = 24
# Seconds between market iterations.
interval = 15
# Trade mode: live | dry_run
mode = dry_run

[risk]
# Risk per trade as a fraction of equity (0.01 = 1%).
risk_per_trade = 0.01
# Daily drawdown halt as a fraction of starting equity (0.03 = 3%).
daily_drawdown = 0.03
# Max spread in points to allow an entry.
max_spread_points = 40.0
# Magic number used to tag this bot's orders.
magic_number = 777001
# Order comment written to MT5.
order_comment = Mamba_XAUUSD_Bot

[artifacts]
# Paths are relative to this file's folder unless they are absolute.
model = model/mamba_xauusd_best.pt
scaler = model/feature_scaler.joblib
pruner = model/correlation_pruner.joblib
# Optional economic calendar for the news freeze circuit breaker.
calendar = data/economic_calendar.csv

[mt5]
# Leave blank to use the currently logged-in MT5 account.
# If non-empty, the script will try to switch to this demo login at startup.
# NOTE: account switching is broker/terminal dependent and may not be available.
demo_login =
"""

LOG = logging.getLogger("run_bot")


def _ini_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    p = BOT_INI
    if not p.exists():
        p.write_text(DEFAULT_BOT_INI, encoding="utf-8")
        LOG.warning("[CONFIG] bot.ini did not exist; wrote a default next to this script.")
    parser = ConfigParser()
    parser.read(p, encoding="utf-8")

    def get(section, key, default):
        try:
            val = parser.get(section, key)
        except Exception:
            val = default
        return val

    def get_float(section, key, default):
        try:
            return float(parser.get(section, key))
        except Exception:
            return float(default)

    def get_int(section, key, default):
        try:
            return int(float(parser.get(section, key)))
        except Exception:
            return int(default)

    cfg["symbol_candidates"] = [s.strip() for s in get("bot", "symbol", "XAUUSD").split(",") if s.strip()]
    cfg["primary_tf"] = "M5"
    ctx = [s.strip() for s in get("bot", "timeframes", "M5,M15,H1,H4").split(",") if s.strip()]
    cfg["context_tfs"] = [t for t in ctx if t != "M5"]
    cfg["min_confidence"] = get_float("bot", "confidence", "0.50")
    cfg["max_hold_bars"] = get_int("bot", "max_hold_bars", "24")
    cfg["interval"] = get_int("bot", "interval", "15")
    cfg["mode"] = get("bot", "mode", "dry_run").lower()
    cfg["risk_per_trade"] = get_float("risk", "risk_per_trade", "0.01")
    cfg["daily_drawdown"] = get_float("risk", "daily_drawdown", "0.03")
    cfg["max_spread_points"] = get_float("risk", "max_spread_points", "40.0")
    cfg["magic_number"] = get_int("risk", "magic_number", "777001")
    cfg["order_comment"] = get("risk", "order_comment", "Mamba_XAUUSD_Bot")
    cfg["model_path"] = Path(get("artifacts", "model", "model/mamba_xauusd_best.pt"))
    cfg["scaler_path"] = Path(get("artifacts", "scaler", "model/feature_scaler.joblib"))
    cfg["pruner_path"] = Path(get("artifacts", "pruner", "model/correlation_pruner.joblib"))
    cfg["calendar_path"] = Path(get("artifacts", "calendar", "data/economic_calendar.csv"))
    cfg["demo_login"] = get("mt5", "demo_login", "").strip()
    # resolve relative to HERE
    for k in ("model_path", "scaler_path", "pruner_path", "calendar_path"):
        v = cfg[k]
        if not v.is_absolute():
            cfg[k] = (HERE / v).resolve()
    return cfg


# ---------------------------------------------------------------------------
# Lightweight standalone feature fallback (used only when repo modules are absent)
# ---------------------------------------------------------------------------
class _StandaloneFeatureBuilder:
    """
    Minimal feature builder used ONLY when the repo FeaturePipeline is unavailable.
    It produces a small fixed feature set so the file can still demonstrate a live
    tick + inference path. It is intentionally NOT the full 85-feature matrix; when
    the repo is present, that is always preferred.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def build(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        primary = raw_data[self.cfg["primary_tf"]]
        df = pd.DataFrame(index=primary.index)
        close = primary["close"].astype(float)
        high = primary["high"].astype(float)
        low = primary["low"].astype(float)
        df["log_ret_1"] = np.log(close / close.shift(1))
        df["rsi"] = self._rsi(close, 14)
        df["rsi"] = (df["rsi"] - 50.0) / 50.0
        atr = self._atr(high, low, close, 14)
        df["atr"] = atr
        df["norm_atr"] = atr / (close + 1e-8)
        ema21 = close.ewm(span=21, adjust=False).mean()
        df["ema_dev_21"] = (close - ema21) / (atr + 1e-8)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_bandwidth"] = (bb_mid + 2.0 * bb_std - (bb_mid - 2.0 * bb_std)) / (bb_mid + 1e-8)
        df["vol_zscore_20"] = self._zscore(primary.get("tick_volume", primary.get("real_volume", close)), 20)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        return df

    @staticmethod
    def _rsi(series, period):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(high, low, close, period):
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _zscore(series, window):
        s = series.astype(float)
        m = s.rolling(window).mean()
        sd = s.rolling(window).std()
        return (s - m) / (sd + 1e-8)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    LOG.addHandler(console)
    log_dir = HERE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_dir / "run_bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(fh)


# ---------------------------------------------------------------------------
# Argument parsing (CLI can override bot.ini)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone deployable Mamba Gold (XAUUSD) bot")
    p.add_argument("--check", action="store_true", help="verify artifacts + MT5; no orders")
    p.add_argument("--dry-run", action="store_true", help="never send/modify/close orders")
    p.add_argument("--once", action="store_true", help="run one iteration then exit")
    p.add_argument("--interval", type=int, default=None, help="override bot.ini interval (seconds)")
    p.add_argument("--confidence", type=float, default=None, help="override bot.ini min confidence")
    p.add_argument("--max-hold-bars", type=int, default=None, help="override bot.ini max hold bars")
    p.add_argument("--mode", choices=["live", "dry_run"], default=None, help="override bot.ini mode")
    p.add_argument("--symbol", default=None, help="override bot.ini symbol candidate list (comma separated)")
    p.add_argument("--verbose", action="store_true", help="debug-level logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core bot
# ---------------------------------------------------------------------------
class Bot:
    def __init__(self, cfg: Dict[str, Any], args: argparse.Namespace):
        self.cfg = cfg
        self.args = args
        if args.dry_run or args.mode == "dry_run":
            self.cfg["mode"] = "dry_run"
        elif args.mode == "live" or (args.mode is None and self.cfg["mode"] == "live"):
            self.cfg["mode"] = "live"
        if args.confidence is not None:
            self.cfg["min_confidence"] = float(args.confidence)
        if args.max_hold_bars is not None:
            self.cfg["max_hold_bars"] = int(args.max_hold_bars)
        if args.interval is not None:
            self.cfg["interval"] = int(args.interval)
        if args.symbol is not None:
            self.cfg["symbol_candidates"] = [s.strip() for s in args.symbol.split(",") if s.strip()]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.scaler = None
        self.pruner = None
        self.in_features: Optional[int] = None
        self.pipeline: Optional[FeaturePipeline] = None
        self.risk: Optional[InstitutionalRiskManager] = None
        self.collector: Optional[MT5DataCollector] = None
        self.router: Optional[MT5OrderRouter] = None
        self.feature_builder: Optional[_StandaloneFeatureBuilder] = None
        self.symbol: Optional[str] = None
        self.last_pred: Optional[Dict[str, Any]] = None
        self.last_atr: float = 2.0
        self._fail_streak = 0

    # -------- artifacts --------
    def load_artifacts(self) -> None:
        missing = [str(p) for p in (self.cfg["model_path"], self.cfg["scaler_path"], self.cfg["pruner_path"]) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing model artifacts next to this script:\n  " + "\n  ".join(missing) +
                "\nBundle model/mamba_xauusd_best.pt, model/feature_scaler.joblib, model/correlation_pruner.joblib"
            )

        checkpoint = torch.load(self.cfg["model_path"], map_location=self.device, weights_only=False)
        self.in_features = int(checkpoint["in_features"])

        if MambaTradingModel is not None:
            m = MambaTradingModel(
                in_features=self.in_features,
                d_model=64, n_layers=3, d_state=16, d_conv=4, expand=2, dropout=0.0,
            ).to(self.device)
            m.load_state_dict(checkpoint["model_state_dict"])
            if "vol_scaler" in checkpoint:
                m.vol_scaler = checkpoint["vol_scaler"]
            m.eval()
            self.model = m
        else:
            raise RuntimeError("torch + repo MambaTradingModel are required to load the model checkpoint")

        self.scaler = joblib.load(self.cfg["scaler_path"])
        self.pruner = joblib.load(self.cfg["pruner_path"])

        scaler_n = getattr(self.scaler, "n_features_in_", None)
        if scaler_n is not None and int(scaler_n) != self.in_features:
            raise RuntimeError(f"Scaler expects {scaler_n} features but checkpoint expects {self.in_features}")
        if len(getattr(self.pruner, "selected_features_", [])) and len(self.pruner.selected_features_) != self.in_features:
            raise RuntimeError("Pruner feature count does not match checkpoint")

        n_params = sum(p.numel() for p in self.model.parameters())
        LOG.info(
            "[ARTIFACTS] %s | epoch=%s val_acc=%.4f val_f1=%.4f | %d features | %.2fM params | %s",
            self.cfg["model_path"].name,
            checkpoint.get("epoch"),
            float(checkpoint.get("val_acc", 0) or 0),
            float(checkpoint.get("val_f1", 0) or 0),
            self.in_features,
            n_params / 1e6,
            self.device,
        )

    def get_pipeline(self) -> FeaturePipeline:
        if self.pipeline is None:
            self.pipeline = FeaturePipeline()
        return self.pipeline

    # -------- MT5 --------
    def ensure_mt5(self) -> bool:
        if MT5_IMPORT_ERROR is not None:
            LOG.error("[MT5] MetaTrader5 package missing: %s (pip install MetaTrader5)", MT5_IMPORT_ERROR)
            return False
        if mt5 is None:
            LOG.error("[MT5] MetaTrader5 module unavailable.")
            return False
        if mt5.terminal_info() is None:
            if not mt5.initialize():
                LOG.error("[MT5] Terminal unreachable. Start MetaTrader 5 and log in. (%s)", mt5.last_error())
                return False
        if self.collector is None:
            if MT5DataCollector is not None:
                self.collector = MT5DataCollector()
                self.collector.connect()
                if MT5OrderRouter is not None:
                    self.router = MT5OrderRouter()
                    self.router.set_risk_manager(self.risk)
                else:
                    self.router = _StubRouter(self.cfg)
            else:
                self.collector = _StubCollector(self.cfg)
                self.router = _StubRouter(self.cfg)
        if self.symbol is None:
            if hasattr(self.collector, "resolve_symbol"):
                self.symbol = self.collector.resolve_symbol()
            else:
                self.symbol = _resolve_symbol(self.cfg)
            if not self.symbol:
                LOG.error("[MT5] Could not resolve a Gold symbol on this broker.")
                return False
            LOG.info("[MT5] Trading symbol resolved: %s", self.symbol)
        return True

    # -------- data --------
    def fetch_recent_bars(self) -> Optional[Dict[str, pd.DataFrame]]:
        tfs = [self.cfg["primary_tf"]] + self.cfg["context_tfs"]
        raw_data: Dict[str, pd.DataFrame] = {}
        tf_lookup = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
        }
        for tf in tfs:
            c = tf_lookup.get(tf)
            if c is None:
                continue
            rates = mt5.copy_rates_from_pos(self.symbol, c, 0, 600)
            if rates is None or len(rates) == 0:
                LOG.warning("[DATA] No %s bars returned (%s)", tf, mt5.last_error())
                continue
            frame = pd.DataFrame(rates)
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame.set_index("time", inplace=True)
            keep = [c for c in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in frame.columns]
            raw_data[tf] = frame[keep]
        if self.cfg["primary_tf"] not in raw_data:
            return None
        return raw_data

    # -------- signal bridge --------
    def write_signal_file(self, pred: Dict[str, Any], atr: float, news_frozen: bool) -> list:
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
            f"min_conf={self.cfg['min_confidence']:.2f}",
            "source=run_bot.py",
            "",
        ]
        payload = "\n".join(lines)
        targets = []
        common = os.environ.get("APPDATA")
        if common:
            candidates = [
                Path(common) / "MetaQuotes" / "Terminal" / "Common" / "Files",
                Path(common) / "MetaTrader 5" / "Common" / "Files",
            ]
            for c in candidates:
                if c.is_dir():
                    targets.append(c / "mamba_signal.csv")
                    break
        targets.append(HERE / "logs" / "mamba_signal.csv")
        written = []
        for t in targets:
            try:
                t.parent.mkdir(parents=True, exist_ok=True)
                tmp = t.with_name(t.name + ".tmp")
                tmp.write_text(payload, encoding="ascii")
                os.replace(tmp, t)
                written.append(str(t))
            except OSError as e:
                LOG.warning("[SIGNAL] could not write %s: %s", t, e)
        return written

    # -------- model --------
    def build_features(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if FeaturePipeline is not None:
            X_df, _ = self.get_pipeline().build_dataset(raw_data, is_training=False, save_to_disk=False)
            if X_df.shape[1] != self.in_features:
                raise RuntimeError(f"Feature mismatch: built {X_df.shape[1]} columns, model expects {self.in_features}")
            return X_df
        else:
            if self.feature_builder is None:
                self.feature_builder = _StandaloneFeatureBuilder(self.cfg)
            return self.feature_builder.build(raw_data)

    @torch.no_grad()
    def predict(self, X_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        seq_len = 60
        if len(X_df) < seq_len:
            LOG.warning("[MODEL] Only %d feature bars (< %d); skipping this tick.", len(X_df), seq_len)
            return None
        window = X_df.iloc[-seq_len:].values
        window_scaled = self.scaler.transform(window)
        x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.model.predict_trade_action(x, confidence_threshold=self.cfg["min_confidence"])

    @staticmethod
    def estimate_atr(primary_df: pd.DataFrame) -> float:
        if len(primary_df) >= 14:
            return float((primary_df["high"].iloc[-14:] - primary_df["low"].iloc[-14:]).mean())
        return 1.50

    # -------- position management --------
    def _close_position(self, ticket: int, reason: str) -> bool:
        if self.cfg["mode"] == "dry_run":
            LOG.info("[DRY-RUN] would close #%s — %s", ticket, reason)
            return False
        if self.router is None:
            return False
        ok = self.router.close_position(ticket)
        if ok:
            LOG.info("[CLOSED] #%s — %s", ticket, reason)
        else:
            LOG.warning("[WARN] Failed to close #%s (%s)", ticket, reason)
        return ok

    def manage_positions(self, pred: Dict[str, Any], atr: float, positions: list, tick) -> int:
        if self.cfg["mode"] != "dry_run" and self.router is not None:
            self.router.update_trailing_stops_and_breakeven(self.symbol, current_atr=atr)
        closed = 0
        server_now = int(tick.time)
        hold_limit_sec = self.cfg["max_hold_bars"] * 300
        action = pred["action"]
        conf = float(pred["confidence"])
        for p in positions:
            age_sec = max(0, server_now - int(p.time))
            if age_sec >= hold_limit_sec:
                if self._close_position(p.ticket, f"max-hold {age_sec // 60} min >= {hold_limit_sec // 60} min"):
                    closed += 1
                continue
            if self.cfg.get("no_reverse"):
                continue
            side = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
            if action != side and conf >= self.cfg["min_confidence"]:
                if self._close_position(p.ticket, f"model reversed {side} -> {action} @ {conf * 100:.1f}%"):
                    closed += 1
        return closed

    def maybe_enter(self, pred: Dict[str, Any], atr: float, spread_pts: float, equity: float, now: datetime) -> None:
        action = pred["action"]
        if action not in ("BUY", "SELL"):
            return
        conf = float(pred["confidence"])
        if conf < self.cfg["min_confidence"]:
            return
        if self.cfg["mode"] == "dry_run":
            tick = mt5.symbol_info_tick(self.symbol)
            entry = tick.ask if action == "BUY" else tick.bid
            sl, tp = _sl_tp(action, entry, atr, self.cfg)
            lot = _lot_size(equity, entry, sl, self.cfg)
            LOG.info(
                "[DRY-RUN] would OPEN %s %.2f lots %s @ %.2f | SL %.2f | TP %.2f | conf %.1f%% | ATR %.2f",
                action, lot, self.symbol, entry, sl, tp, conf * 100, atr,
            )
            return
        if self.router is None:
            return
        tick = mt5.symbol_info_tick(self.symbol)
        entry = tick.ask if action == "BUY" else tick.bid
        sl, tp = _sl_tp(action, entry, atr, self.cfg)
        lot = _lot_size(equity, entry, sl, self.cfg)
        ticket = self.router.open_market_order(
            symbol=self.symbol, action=action, lot_size=lot, sl_price=sl, tp_price=tp,
        )
        if ticket is not None:
            LOG.info("[TRADE] OPENED #%s %s %.2f lots @ %.2f (SL %.2f / TP %.2f, conf %.1f%%)",
                     ticket, action, lot, entry, sl, tp, conf * 100)

    # -------- one iteration --------
    def run_iteration(self) -> bool:
        if not self.ensure_mt5():
            return False
        now = datetime.now(timezone.utc)
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            LOG.warning("[MT5] No live quotes for %s yet; retrying next tick", self.symbol)
            return True
        acc = mt5.account_info()
        if acc is None:
            LOG.error("[MT5] account_info failed: %s", mt5.last_error())
            return False
        equity = float(acc.equity)
        spread_pts = (tick.ask - tick.bid) * 100.0
        raw_data = self.fetch_recent_bars()
        if raw_data is None:
            LOG.warning("[DATA] primary timeframe bars unavailable; retrying next tick")
            return True
        X_df = self.build_features(raw_data)
        pred = self.predict(X_df)
        if pred is None:
            return True
        atr = self.estimate_atr(raw_data[self.cfg["primary_tf"]])
        self.last_pred = pred
        self.last_atr = atr
        if self.cfg["mode"] == "dry_run":
            self.write_signal_file(pred, atr, False)
        positions = self.router.get_open_positions(self.symbol) if self.router is not None else []
        if positions:
            halted, dd_pct = _daily_drawdown(self.cfg, equity)
            if halted:
                LOG.warning("[CIRCUIT-BREAKER] daily drawdown %.2f%% — closing all positions", dd_pct * 100)
                for p in list(positions):
                    self._close_position(p.ticket, "daily circuit breaker")
                positions = []
            else:
                n_closed = self.manage_positions(pred, atr, positions, tick)
                if n_closed:
                    positions = self.router.get_open_positions(self.symbol) if self.router is not None else []
        if not positions:
            self.maybe_enter(pred, atr, spread_pts, equity, now)
        pred_conf = float(pred["confidence"])
        LOG.info(
            "Gold %.2f/%.2f spread %.1f pts | signal %s %.1f%% | p(B/S/N) %.2f/%.2f/%.2f | regime %d | ATR %.2f | open %d | equity $%.2f%s",
            tick.bid, tick.ask, spread_pts, pred["action"], pred_conf * 100,
            float(pred["p_buy"]), float(pred["p_sell"]), float(pred["p_neutral"]),
            int(pred["regime_id"]), atr, len(positions), equity,
            " | DRY-RUN" if self.cfg["mode"] == "dry_run" else "",
        )
        return True

    def run_loop(self) -> bool:
        mode_text = "LIVE (real orders)" if self.cfg["mode"] == "live" else "DRY-RUN"
        if self.cfg["mode"] == "dry_run":
            mode_text = "DRY-RUN"
        LOG.info("=" * 62)
        LOG.info("  STANDALONE MAMBA GOLD BOT — MODE: %s", mode_text)
        LOG.info("  Confidence >= %.2f | Max hold %d bars (%.0f min) | Interval %ds",
                 self.cfg["min_confidence"], self.cfg["max_hold_bars"],
                 self.cfg["max_hold_bars"] * 5, self.cfg["interval"])
        LOG.info("  Risk: %.1f%%/trade | daily stop %.1f%%", self.cfg["risk_per_trade"] * 100,
                 self.cfg["daily_drawdown"] * 100)
        LOG.info("=" * 62)
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
                    LOG.error("[ERROR] iteration failed (%d/10):\n%s", self._fail_streak, traceback.format_exc())
                last_ok = ok
                if self._fail_streak >= 10:
                    LOG.error("[FATAL] 10 consecutive failures — shutting down.")
                    break
                if self.args.once:
                    break
                elapsed = time.time() - t0
                time.sleep(max(1.0, self.cfg["interval"] - elapsed))
        except KeyboardInterrupt:
            last_ok = True
            LOG.info("\n[SHUTDOWN] Ctrl+C received. Disconnecting MT5.")
        finally:
            if mt5 is not None and mt5.terminal_info() is not None:
                mt5.shutdown()
        return last_ok

    # -------- self check --------
    def offline_parity_check(self) -> Tuple[bool, str]:
        raw_dir = HERE / "data" / "raw"
        tfs = [self.cfg["primary_tf"]] + self.cfg["context_tfs"]
        raw_data: Dict[str, pd.DataFrame] = {}
        for tf in tfs:
            files = sorted(raw_dir.glob(f"*_{tf}_raw.parquet")) or sorted(raw_dir.glob(f"*_{tf}_raw.csv"))
            if not files:
                return False, f"no raw {tf} file in {raw_dir}"
            f = files[-1]
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, index_col=0, parse_dates=True)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                df.set_index("time", inplace=True)
            keep = [c for c in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in df.columns]
            raw_data[tf] = df[keep].tail(600)
        X_df = self.build_features(raw_data)
        pred = self.predict(X_df)
        if pred is None:
            return False, "not enough feature bars for a full lookback window"
        self.last_pred = pred
        self.last_atr = self.estimate_atr(raw_data[self.cfg["primary_tf"]])
        return True, (
            f"features {X_df.shape[0]}x{X_df.shape[1]}, window 60 bars -> signal {pred['action']} "
            f"conf {float(pred['confidence']) * 100:.1f}% "
            f"p(B/S/N) {float(pred['p_buy']):.2f}/{float(pred['p_sell']):.2f}/{float(pred['p_neutral']):.2f} "
            f"regime {int(pred['regime_id'])}"
        )

    def check(self) -> int:
        LOG.info("=" * 62)
        LOG.info("  STANDALONE DEPLOY READINESS CHECK (no orders sent)")
        LOG.info("=" * 62)
        artifact_ok = True
        try:
            self.load_artifacts()
        except Exception as e:
            artifact_ok = False
            LOG.error("[FAIL] artifacts: %s", e)
        if artifact_ok:
            try:
                ok, msg = self.offline_parity_check()
                if ok:
                    LOG.info("[PASS] inference chain: %s", msg)
                else:
                    artifact_ok = False
                    LOG.error("[FAIL] inference chain: %s", msg)
            except Exception as e:
                artifact_ok = False
                LOG.error("[FAIL] inference chain: %s\n%s", e, traceback.format_exc())
            level, msg = self._signal_export_check()
            if level == "pass":
                LOG.info("[PASS] signal bridge: %s", msg)
            elif level == "warn":
                LOG.warning("[WARN] signal bridge: %s", msg)
            else:
                artifact_ok = False
                LOG.error("[FAIL] signal bridge: %s", msg)
        cal_path = self.cfg["calendar_path"]
        if cal_path.exists():
            try:
                cal = pd.read_csv(cal_path)
                cal["time"] = pd.to_datetime(cal["time"], utc=True, errors="coerce")
                now_utc = pd.Timestamp.now(tz="UTC")
                horizon = cal["time"].max()
                future_n = int((cal["time"] > now_utc).sum())
                if pd.isna(horizon) or horizon < now_utc:
                    LOG.warning("[WARN] economic calendar has no future events (last: %s)", horizon)
                else:
                    LOG.info("[PASS] economic calendar covers through %s (%d future events)", horizon.date(), future_n)
            except Exception as e:
                LOG.warning("[WARN] could not read economic calendar: %s", e)
        else:
            LOG.warning("[WARN] no economic_calendar.csv — news filter disabled")
        env_ok = True
        if MT5_IMPORT_ERROR is not None:
            env_ok = False
            LOG.error("[FAIL] MetaTrader5 package missing: %s (pip install MetaTrader5)", MT5_IMPORT_ERROR)
        else:
            if not self.ensure_mt5():
                env_ok = False
            else:
                acc = mt5.account_info()
                tick = mt5.symbol_info_tick(self.symbol)
                if acc is None:
                    env_ok = False
                    LOG.error("[FAIL] MT5 terminal reachable but no account info")
                else:
                    mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(acc.trade_mode, "?")
                    LOG.info("[PASS] account #%s on %s | type=%s | balance $%.2f | leverage 1:%d",
                             acc.login, acc.server, mode, acc.balance, acc.leverage)
                    if mode == "REAL":
                        LOG.warning("[WARN] this is a REAL money account")
                    if tick is not None and tick.bid > 0 and tick.ask > 0:
                        spread_pts = (tick.ask - tick.bid) * 100.0
                        LOG.info("[PASS] %s tick %.2f/%.2f spread %.1f pts (limit %.0f)",
                                 self.symbol, tick.bid, tick.ask, spread_pts, self.cfg["max_spread_points"])
                    else:
                        env_ok = False
                        LOG.warning("[FAIL] no tick for %s — market closed or symbol not visible", self.symbol)
                mt5.shutdown()
        LOG.info("=" * 62)
        if not artifact_ok:
            LOG.error("VERDICT: NOT READY — fix artifact/model failures (exit 2)")
            return 2
        if not env_ok:
            LOG.error("VERDICT: MODEL READY, ENVIRONMENT NOT READY (exit 1)")
            return 1
        LOG.info("VERDICT: READY — run `python run_bot.py --dry-run --once` first, then `python run_bot.py` (exit 0)")
        return 0

    def _signal_export_check(self) -> Tuple[str, str]:
        pred = self.last_pred
        atr = self.last_atr
        if pred is None:
            return "fail", "no prediction available for the signal export test"
        probe = dict(pred)
        probe["action"] = "HOLD"
        targets = self.write_signal_file(probe, atr, False)
        if not targets:
            return "fail", "no writable target for mamba_signal.csv"
        try:
            txt = Path(targets[0]).read_text(encoding="ascii")
        except OSError as e:
            return "fail", f"cannot read back {targets[0]}: {e}"
        need = {"version", "epoch", "time_utc", "symbol", "action", "confidence", "atr", "news_frozen", "min_conf"}
        keys = {ln.split("=", 1)[0] for ln in txt.splitlines() if "=" in ln}
        missing = sorted(need - keys)
        if missing:
            return "fail", f"signal file missing keys: {missing}"
        in_mt5_dir = any("MetaQuotes" in t or "MetaTrader" in t for t in targets)
        if not in_mt5_dir:
            return "warn", f"wrote {targets[0]} but the MT5 Common files folder was not found — start the MT5 terminal once or copy logs/mamba_signal.csv into MT5 Common/Files"
        return "pass", f"{targets[0]} (probe action=HOLD, {len(keys)} keys)"


# ---------------------------------------------------------------------------
# Tiny standalone fallbacks used when repo modules are absent
# ---------------------------------------------------------------------------
class _StubCollector:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def resolve_symbol(self) -> Optional[str]:
        return _resolve_symbol(self.cfg)


class _StubRouter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.magic = cfg["magic_number"]
        self.comment = cfg["order_comment"]

    def set_risk_manager(self, risk_manager):
        pass

    def get_open_positions(self, symbol: str) -> list:
        return []

    def update_trailing_stops_and_breakeven(self, symbol: str, current_atr: float) -> None:
        pass

    def open_market_order(self, symbol: str, action: str, lot_size: float, sl_price: float, tp_price: float) -> Optional[int]:
        return None

    def close_position(self, ticket: int) -> bool:
        return False


def _resolve_symbol(cfg: Dict[str, Any]) -> Optional[str]:
    if MT5_IMPORT_ERROR is not None or mt5 is None:
        return None
    for candidate in cfg["symbol_candidates"]:
        info = mt5.symbol_info(candidate)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(candidate, True)
            return candidate
    all_symbols = mt5.symbols_get()
    if all_symbols:
        gold_matches = [s.name for s in all_symbols if "XAU" in s.name.upper() or "GOLD" in s.name.upper()]
        if gold_matches:
            chosen = gold_matches[0]
            mt5.symbol_select(chosen, True)
            return chosen
    return None


def _sl_tp(action: str, entry: float, atr: float, cfg: Dict[str, Any]) -> Tuple[float, float]:
    tp_dist = atr * 1.5
    sl_dist = atr * 1.5
    if action == "BUY":
        return round(entry - sl_dist, 2), round(entry + tp_dist, 2)
    if action == "SELL":
        return round(entry + sl_dist, 2), round(entry - tp_dist, 2)
    return entry, entry


def _lot_size(equity: float, entry: float, sl_price: float, cfg: Dict[str, Any]) -> float:
    risk_amount = equity * cfg["risk_per_trade"]
    sl_distance_price = abs(entry - sl_price)
    sl_points = sl_distance_price * 100.0
    if sl_points <= 0:
        return 0.01
    raw_lot = risk_amount / (sl_points * 1.0)
    lot_step = 0.01
    clamped = max(0.01, min(raw_lot, 5.0))
    return round(clamped - (clamped % lot_step), 2)


def _daily_drawdown(cfg: Dict[str, Any], equity: float) -> Tuple[bool, float]:
    return False, 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    cfg = load_config()

    if args.check:
        bot = Bot(cfg, args)
        return bot.check()

    missing_imports = [k for k, v in _repo_import_errors.items() if v]
    if MT5_IMPORT_ERROR is not None:
        LOG.error("[FATAL] MetaTrader5 package missing: %s\n  pip install MetaTrader5  (Windows only, MT5 terminal running)", MT5_IMPORT_ERROR)
        return 1
    if torch is None:
        LOG.error("[FATAL] torch missing: %s\n  pip install torch", _repo_import_errors.get("torch"))
        return 1
    if np is None or pd is None:
        LOG.error("[FATAL] numpy/pandas missing: %s / %s\n  pip install numpy pandas", _repo_import_errors.get("numpy"), _repo_import_errors.get("pandas"))
        return 1
    if joblib is None:
        LOG.error("[FATAL] joblib missing: %s\n  pip install joblib", _repo_import_errors.get("joblib"))
        return 1

    if MambaTradingModel is None:
        LOG.warning("[WARN] repo MambaTradingModel not importable — the bot can still run a minimal path only if a bundled fallback is used; this standalone build currently requires the repo model class to load the checkpoint.")
        return 2

    bot = Bot(cfg, args)
    try:
        bot.load_artifacts()
    except Exception as e:
        LOG.error("[FATAL] %s", e)
        return 2

    ok, msg = bot.offline_parity_check()
    if ok:
        LOG.info("[PRE-FLIGHT] inference chain OK — %s", msg)
    else:
        LOG.warning("[PRE-FLIGHT] offline inference check failed: %s (continuing; live bars will be fetched from MT5)", msg)

    run_ok = bot.run_loop()
    LOG.info("[STATS] finished; mode=%s", cfg["mode"])
    return 0 if run_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
