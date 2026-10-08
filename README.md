# 🥇 Mamba Gold Trading Bot — XAUUSD AI Trading System

> **An institutional-grade AI trading bot for Gold (XAUUSD) on MetaTrader 5, powered by a custom Selective State Space Model (Mamba/S6) neural network with multi-task learning, multi-timeframe feature engineering, and rigorous risk management.**

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Trading Strategy](#-trading-strategy)
3. [System Architecture](#-system-architecture)
4. [Mamba Model Deep Dive](#-mamba-model-deep-dive)
5. [Feature Engineering](#-feature-engineering)
6. [Trading Parameters and Risk Configuration](#-trading-parameters-and-risk-configuration)
7. [Model Training and Optimization](#-model-training-and-optimization)
8. [How It Helps in the Market](#-how-it-helps-in-the-market)
9. [Quick Start and Entry Points](#-quick-start-and-entry-points)
10. [MT5 Setup](#-mt5-setup)
11. [Standalone Deployment Bundle](#-standalone-deployment-bundle)
12. [MQL5 EA Bridge](#-mql5-ea-bridge)
13. [Project File Structure](#-project-file-structure)
14. [Further Updates and Roadmap](#-further-updates-and-roadmap)
15. [Important Warnings](#-important-warnings)

---

## 🔭 Project Overview

The **Mamba Gold Trading Bot** is a Python-driven automated trading system that connects to MetaTrader 5 from outside the terminal. It implements a research-grade decision engine using the **Mamba Selective State Space Model (S6)** — a state-of-the-art sequence model that processes time series with **linear O(L) complexity**, unlike transformers which are O(L²).

### What makes this different from a standard EA

| Feature | Standard MT5 EA | This Bot |
|---|---|---|
| Decision Engine | Rule-based indicators | Mamba S6 Neural Network |
| Timeframe Analysis | Single TF | M5 + M15 + H1 + H4 (4 TFs) |
| Signal Type | Binary (Buy/Sell) | 3-class probability distribution |
| Risk Management | Fixed SL/TP | ATR-adaptive + trailing + breakeven |
| Market Regime Awareness | None | 4-class regime classifier (built-in) |
| News Filtering | None | Event-specific freeze windows (FOMC, NFP, CPI) |
| Volatility Estimation | None | Dynamic neural volatility head |

---

## 📈 Trading Strategy

### Core Philosophy

The bot uses a **multi-timeframe, regime-aware, confidence-gated** approach to Gold trading:

1. **M5 (5-Minute)** — Primary decision timeframe for entry/exit signals
2. **M15, H1, H4** — Higher timeframe context to align trade direction with macro trend

### Signal Generation Logic

```
Step 1: Fetch live multi-TF candles from MT5
Step 2: Rebuild feature matrix (60-bar sequence x N features)
Step 3: Run Mamba model → get [P_BUY, P_SELL, P_HOLD] + regime + volatility
Step 4: Apply confidence gate (default: 50% minimum)
Step 5: Apply risk checks (spread, news, drawdown)
Step 6: Execute BUY / SELL / HOLD
Step 7: Manage open trade (breakeven, trailing stop, time-based exit)
```

### Triple Barrier Labeling (Training Labels)

Training data is labeled using the **Marcos Lopez de Prado Triple Barrier Method**:

- **Upper Barrier (Take Profit):** `+1.5 x ATR` above entry
- **Lower Barrier (Stop Loss):** `-1.5 x ATR` below entry
- **Time Barrier:** 24 M5 bars = **2 hours maximum hold**
- **Neutral Zone:** Minimum return threshold of `0.05%` to avoid labeling noise as signals

This ensures the model learns economically meaningful, risk-adjusted moves — not just any price fluctuation.

### Market Session Awareness

The feature set encodes **trading session timing** (UTC):

| Session | Hours (UTC) | Gold Characteristic |
|---|---|---|
| Asian | 00:00 – 08:00 | Low volatility, range-bound |
| London | 07:00 – 16:00 | Trend initiation, institutional flow |
| New York | 12:00 – 21:00 | High volatility, news-driven moves |
| London-NY Overlap | 12:00 – 16:00 | Highest liquidity + trend continuation |

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MAMBA GOLD TRADING BOT                       │
├─────────────────┬───────────────────┬───────────────────────────┤
│  DATA LAYER     │  FEATURE LAYER    │  MODEL LAYER              │
│                 │                   │                           │
│  MetaTrader 5   │  Technical Feats  │  Input Projection         │
│  (Live Feed)    │  RSI/StochRSI     │  (Linear to d_model)      │
│                 │  ATR / BB / MACD  │                           │
│  M5  ─────────> │  Microstructure   │  Mamba Block x 3          │
│  M15 ─────────> │  Parkinson Vol    │  (Selective SSM)          │
│  H1  ─────────> │  Garman-Klass     │                           │
│  H4  ─────────> │  Multi-TF Context │  Direction Head           │
│                 │  Session Encoding │  Volatility Head          │
│  Economic       │  Volume Norms     │  Regime Head              │
│  Calendar       │  Corr Pruning     │                           │
│  (News Filter)  │                   │                           │
├─────────────────┴───────────────────┴───────────────────────────┤
│                    EXECUTION LAYER                               │
│  Risk Checks → Position Sizing → MT5 Order → Trade Management   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | File(s) | Role |
|---|---|---|
| **Configuration** | `config.py` | Centralized typed settings for all layers |
| **Data Collection** | `collectors/` | Historical OHLCV + calendar data fetcher |
| **Feature Engineering** | `features/` | Technical, volume, multi-TF, session features |
| **Labeling** | `features/labeler.py` | Triple Barrier label generator |
| **Model Definition** | `models/mamba_block.py`, `models/mamba_model.py` | S6 SSM core + multi-task network |
| **Training** | `training/train.py` | Full training pipeline with AMP + early stopping |
| **Hyperparameter Search** | `training/hyperparam_search.py` | Optuna-based HPO |
| **Walk-Forward Validation** | `training/walkforward.py` | Realistic OOS evaluation |
| **Live Bot** | `deploy_bot.py` | Primary deployable bot (repo-aware) |
| **Standalone Bot** | `run_bot.py` | Self-contained copyable bot |
| **MQL5 Bridge** | `MambaGoldEA.mq5` | Signal file bridge for MT5 EA execution |
| **Backtesting** | `backtest/` | Strategy backtesting tools |

---

## 🧠 Mamba Model Deep Dive

### Why Mamba (S6) Instead of LSTM or Transformer?

| Model | Complexity | Memory | Context | Financial Use |
|---|---|---|---|---|
| LSTM | O(L) | O(L) | Short-term | Limited context capture |
| Transformer | O(L²) | O(L²) | Long-term | Too slow for real-time |
| **Mamba S6** | **O(L)** | **O(L)** | **Long-term** | **Best of both worlds** |

Mamba uses **Selective State Space Models** that dynamically compress relevant information from the input sequence — ideal for financial time series where most data is noise and only a few key moments matter.

### Architecture Breakdown

Located in `models/mamba_model.py` and `models/mamba_block.py`.

#### 1. Input Projection and Normalization

```python
input_proj    → Linear(in_features → d_model=64)
input_norm    → LayerNorm(d_model)
input_dropout → Dropout(0.15)
```

#### 2. Selective State Space Block (The Mamba Core)

Each `MambaBlock` contains:

```
Input x (Batch, SeqLen, d_model)
    │
    ▼
in_proj → splits into [x_branch, z_branch]   (2 x d_inner)
    │
    ▼  x_branch
Causal Conv1D (kernel=4, depthwise)           Local temporal patterns
    │
    SiLU activation
    ▼
Selective SSM (S6):
  h_t = exp(Delta * A) * h_{t-1} + (Delta * B) * x_t    State update
  y_t = C * h_t + D * x_t                                Output
    │
    ▼
Gated Linear Unit: y_ssm * SiLU(z_branch)    Dynamic gating
    │
    ▼
out_proj → Linear(d_inner → d_model)
```

#### Key SSM Parameters

| Parameter | Value | Role |
|---|---|---|
| `d_model` | 64 | Hidden representation dimension |
| `d_state` | 16 | SSM memory / state space dimension |
| `d_conv` | 4 | Local convolution kernel width |
| `expand` | 2 | Inner expansion factor (d_inner = 128) |
| `n_layers` | 3 | Number of stacked Mamba blocks |
| `dropout` | 0.15 | Regularization rate |

#### 3. Multi-Task Decision Heads

The model simultaneously learns three related tasks from the last sequence step:

```
Last Sequence Step (Batch, d_model=64)
          │
    ┌─────┼─────────────┐
    ▼     ▼             ▼
Direction  Volatility   Regime
  Head      Head         Head
    │         │             │
[P_SELL,  Expected    [Bull Trend,
 P_HOLD,  ATR move     Bear Trend,
 P_BUY]   (float)      Chop,
                        News Spike]
```

**Direction Classes:**
- `0` → Bearish / Sell signal
- `1` → Neutral / Hold (choppy market)
- `2` → Bullish / Buy signal

**Regime Classes:**
- `0` → Bull Trend
- `1` → Bear Trend
- `2` → Choppy / Sideways Market
- `3` → News-Driven Volatility Spike

---

## ⚙️ Feature Engineering

### Feature Categories in `features/` module

#### A. Stationary Price Returns

| Feature | Description |
|---|---|
| `log_ret_1` | 1-bar log return |
| `log_ret_3` | 3-bar log return |
| `log_ret_6` | 6-bar log return |
| `log_ret_12` | 12-bar log return |

#### B. Volatility Features

| Feature | Description |
|---|---|
| `atr` | 14-period Average True Range |
| `norm_atr` | ATR / Close (relative volatility %) |
| `parkinson_vol` | High-Low based volatility estimator |
| `garman_klass_vol` | OHLC-based Garman-Klass volatility |
| `bb_bandwidth` | Bollinger Band squeeze / expansion |
| `bb_percent_b` | Price position within Bollinger Bands |

#### C. Momentum Oscillators

| Feature | Description |
|---|---|
| `rsi` | RSI(14) centered to [-1, +1] |
| `rsi_slope_3` | RSI rate-of-change over 3 bars |
| `rsi_slope_5` | RSI rate-of-change over 5 bars |
| `stoch_rsi_k` | Stochastic RSI %K |
| `stoch_rsi_d` | Stochastic RSI %D |
| `macd_hist_norm` | MACD histogram normalized by ATR |

#### D. Trend Alignment (EMA Deviations)

| Feature | Description |
|---|---|
| `ema_dev_9` | Price deviation from EMA(9) in ATR units |
| `ema_dev_21` | Price deviation from EMA(21) in ATR units |
| `ema_dev_50` | Price deviation from EMA(50) in ATR units |
| `ema_dev_200` | Price deviation from EMA(200) in ATR units |
| `ema_21_slope_5` | EMA(21) velocity over 5 bars |

#### E. Volume Microstructure

- Broker-agnostic volume normalization windows: **[20, 50, 200]**
- Relative volume ratios to detect unusual activity

#### F. Multi-Timeframe Context

- Aligns M15 / H1 / H4 features to M5 bars using forward-fill
- Provides macro trend context at each M5 decision point

#### G. Market Session Encoding

- Binary flags: `is_asian`, `is_london`, `is_ny`, `is_overlap`
- Time-of-day cyclical encoding (sin/cos)

#### H. Multicollinearity Reduction

- Drops features with Pearson |r| > **0.80**
- Reduces noise and prevents gradient instability during training

---

## 📊 Trading Parameters and Risk Configuration

All parameters are defined in `config.py`.

### Instrument Parameters

```python
SYMBOL_CANDIDATES  = ["XAUUSD.t", "XAUUSD", "GOLD", "XAUUSDm", ...]  # Auto-detected
PRIMARY_TIMEFRAME  = "M5"           # Decision bar
CONTEXT_TIMEFRAMES = ["M15", "H1", "H4"]
LOT_STEP           = 0.01
MIN_LOT            = 0.01
MAX_LOT            = 5.00
CONTRACT_SIZE      = 100.0          # 1 lot = 100 oz
```

### Feature and Model Parameters

```python
LOOKBACK_SEQUENCE_LENGTH = 60   # 60 x 5-min bars = 5 hours of context
RSI_PERIOD  = 14
ATR_PERIOD  = 14
EMA_FAST    = 9
EMA_MEDIUM  = 21
EMA_SLOW    = 50
EMA_TREND   = 200
CORRELATION_THRESHOLD = 0.80
```

### Triple Barrier Labeling

```python
TP_ATR_MULTIPLIER     = 1.5     # Take-profit at +1.5 ATR
SL_ATR_MULTIPLIER     = 1.5     # Stop-loss at -1.5 ATR
MAX_HOLDING_BARS      = 24      # Max 2 hours (24 x M5)
MIN_RETURN_THRESHOLD  = 0.0005  # 0.05% min move to count as signal
```

### Risk Management Rules

```python
MAX_RISK_PER_TRADE_PCT     = 0.01    # 1% account equity per trade
MAX_DAILY_DRAWDOWN_PCT     = 0.03    # 3% daily loss limit → pause trading
MAX_OPEN_TRADES            = 1       # Single position only
MAX_CONSECUTIVE_LOSSES     = 5       # Pause after 5 straight losers
MAX_ALLOWED_SPREAD_POINTS  = 40.0    # Reject entry if spread > 40 pts
MIN_CONFIDENCE_THRESHOLD   = 0.50    # Model must be >= 50% confident
BREAKEVEN_R_MULTIPLE       = 1.0     # Move SL to BE at 1R profit
ENABLE_TRAILING_STOP       = True
TRAILING_ATR_MULTIPLE      = 1.5     # Trail at 1.5 x ATR behind peak
```

### News Circuit Breakers

| Event | Freeze Before | Freeze After |
|---|---|---|
| FOMC Rate Decision | 45 minutes | 60 minutes |
| NFP / CPI | 15 minutes | 30 minutes |
| Other High-Impact | 15 minutes | 15 minutes |
| Surprise Decay Half-Life | — | 4 hours |

---

## 🔬 Model Training and Optimization

### Training Pipeline in `training/train.py`

#### Data Flow

```
Raw OHLCV Parquet → Feature Matrix (X) + Triple-Barrier Labels (y)
     → Chronological Split (Train / Val / Test, no shuffle)
     → RobustScaler normalization (fit on train only)
     → TimeSeriesSequenceDataset (sliding window, seq_len=60)
     → DataLoader (batch_size=256)
     → ModelTrainer.fit()
```

#### Optimizer and Scheduler

- **Optimizer:** AdamW (`lr=5e-4`, `weight_decay=1e-4`)
- **Scheduler:** Warmup Cosine Annealing
  - Linear warmup from 0 to base LR over first N steps
  - Cosine decay from base LR to `1e-6` over remaining epochs
- **AMP:** Mixed precision (FP16) on CUDA, FP32 on CPU

#### Loss Function — Multi-Task in `training/loss.py`

```
Total Loss = alpha * Direction_Loss + beta * Volatility_Loss + gamma * Regime_Loss

Direction_Loss  → Focal Loss       (handles class imbalance, alpha=1.0)
Volatility_Loss → Huber Loss       (robust to outliers, beta=0.5)
Regime_Loss     → Cross-Entropy    (gamma=0.3, detached from direction head)
```

#### Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | `5e-4` | Stable with warmup cosine |
| `weight_decay` | `1e-4` | L2 regularization |
| `batch_size` | `256` | Better generalization on noisy financial data |
| `max_epochs` | `50` | Early stopping terminates before this |
| `early_stopping_patience` | `10` | Patient stopping avoids premature exit |

### Walk-Forward Validation in `training/walkforward.py`

- Sliding window evaluation across time to simulate real deployment
- Prevents data leakage from future to past
- Reports out-of-sample (OOS) accuracy and F1-score per fold

### Hyperparameter Optimization in `training/hyperparam_search.py`

- **Optuna** framework integration
- Search space: `d_model`, `n_layers`, `d_state`, `dropout`, `lr`, `batch_size`
- Pruning of unpromising trials via Optuna's median pruner

### Model Artifacts Saved After Training

```
saved_models/
  mamba_xauusd_best.pt         Best validation checkpoint (state_dict)
  feature_scaler.joblib         RobustScaler fitted on training data
  correlation_pruner.joblib     Feature pruner (drop list)
```

---

## 💹 How It Helps in the Market

### 1. Captures Long-Range Gold Dependencies

Gold price is influenced by events hours or days ago (Fed speeches, geopolitical escalation, USD index shifts). The Mamba S6 model maintains a **compressed hidden state** that carries forward relevant context across 60 M5 bars (5 hours), something LSTMs struggle with and transformers are too slow to run in real-time.

### 2. Regime-Aware — Avoids the Biggest Trap in Algo Trading

Most bots blow up because they use a **trend-following strategy in choppy markets** or vice versa. This bot classifies the current market into one of 4 regimes (Bull, Bear, Chop, News Spike) and the confidence gate naturally filters low-conviction signals in unfavorable regimes.

### 3. Dynamic Volatility Sizing

Instead of fixed lot sizes, position sizing is scaled to **account equity x 1% risk** with ATR-based SL distance. In a volatile Gold market (e.g. post-NFP), ATR expands, lot size shrinks, and risk stays constant.

### 4. News-Event Protection

Gold is famously reactive to macroeconomic data releases. The bot reads an **economic calendar CSV** and automatically freezes trading around:

- FOMC interest rate decisions (±45/60 min)
- NFP and CPI releases (±15/30 min)
- Other high-impact events (±15 min)

### 5. Broker-Agnostic Symbol Detection

Different brokers list Gold differently (`XAUUSD`, `XAUUSD.t`, `GOLD`, `XAUUSDm`). The bot auto-detects the correct symbol from a priority candidate list, making deployment seamless across brokers.

### 6. Institutional-Grade Risk Controls

The bot enforces hard stops that institutional traders use:

- **Daily drawdown halt** (3% max daily loss)
- **Consecutive loss circuit breaker** (5 losses → pause)
- **Max spread filter** (no entry if market is illiquid)
- **Single position rule** (no compounding exposure)
- **Trailing stop + breakeven** (protect profits)

### 7. Why Mamba Training Works for XAUUSD

| XAUUSD Property | Mamba Advantage |
|---|---|
| Non-stationary price levels | Log-return + ATR-normalized inputs = stationary features |
| Long-range dependencies (macro) | S6 state compression = efficient long context |
| High noise-to-signal ratio | Focal loss + selective gating = focuses on hard cases |
| Volatility clustering | Dedicated volatility head + Huber loss |
| Regime changes | Separate regime classification head |
| Class imbalance (few real signals) | Triple Barrier labeling + Focal Loss |

---

## 🚀 Quick Start and Entry Points

### Installation

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Make sure MetaTrader 5 is installed, open, and logged in
# 3. Make sure XAUUSD is visible in MT5 Market Watch
```

### Using `deploy_bot.py` (Full Repo Bot)

```bash
python deploy_bot.py --check          # Readiness check, no orders placed
python deploy_bot.py --dry-run --once # One signal cycle, no orders
python deploy_bot.py --dry-run        # Continuous loop, no orders
python deploy_bot.py                  # Live trading mode (caution)
```

### Using `run_bot.py` (Standalone Copyable Bot)

```bash
python run_bot.py --check
python run_bot.py --dry-run --once
python run_bot.py --dry-run
python run_bot.py                     # Live trading mode (caution)
```

### Training or Retraining the Model

```bash
python pipeline_runner.py   # Full pipeline: collect → features → label → train
python -m training.train    # Training only (requires preprocessed data)
```

### Running Experiments

```bash
python training/run_experiments.py    # Run configured experiment set
python colab_experiments.py           # Google Colab-compatible training
```

---

## 🖥️ MT5 Setup

### Requirements

1. Install **MetaTrader 5** terminal
2. Log in to a **demo or live account**
3. Add `XAUUSD` (or broker equivalent) to **Market Watch** (`Ctrl+M`)
4. Keep the MT5 terminal **running** while the bot trades
5. Enable **Algorithmic Trading** in MT5 options

### Python Requirements

```
Python 3.9+
MetaTrader5          MT5 Python bridge
torch                PyTorch (CPU or CUDA)
pandas, numpy        Data handling
scikit-learn         Scaler + metrics
joblib               Artifact serialization
tqdm                 Progress bars
optuna               Hyperparameter search
```

Full list in `requirements.txt`.

---

## 📦 Standalone Deployment Bundle

To copy this bot to another machine or directory:

```
your_deploy_folder/
  run_bot.py                    Main bot file
  bot.ini                       Config (edit symbol, paths, risk settings)
  model/
    mamba_xauusd_best.pt
    feature_scaler.joblib
    correlation_pruner.joblib
  data/
    economic_calendar.csv       Optional: enables news freeze
```

Edit `bot.ini` to point artifact paths to your local `model/` folder.

---

## 🔌 MQL5 EA Bridge

An alternative execution path is available via the **MambaGoldEA** Expert Advisor:

- Python writes a **signal file** with direction + confidence
- The MQL5 EA reads the file and executes orders inside MT5 natively

When to use the EA bridge:

- When Python's `MetaTrader5` package has connectivity issues
- When you want MT5-side execution guarantees
- On VPS setups where MT5 runs as a service

**Never run Python live mode and the MQL5 EA simultaneously.** They will conflict on order management.

```bash
# Compile EA (Windows only, requires MT5 compiler)
compile_ea.bat

# EA files:
# MambaGoldEA.mq5   Source code
# MambaGoldEA.ex5   Compiled binary
```

---

## 🗂️ Project File Structure

```
AI_trading-_bot/
│
├── config.py                      All typed configuration settings
├── deploy_bot.py                  Main deployable bot (repo-linked)
├── run_bot.py                     Standalone copyable bot
├── bot.ini                        Standalone bot config file
├── pipeline_runner.py             Full end-to-end pipeline runner
├── main_live.py                   Alternate live runner
├── colab_experiments.py           Google Colab training script
├── weekly.py                      Weekly retraining helper
├── history.py                     Trade history utilities
├── requirements.txt               Python dependencies
├── compile_ea.bat                 MT5 EA compiler helper
│
├── models/                        Neural network definitions
│   ├── mamba_block.py             SelectiveSSM + MambaBlock layers
│   └── mamba_model.py             Full MambaTradingModel (multi-task)
│
├── features/                      Feature engineering pipeline
│   ├── technical.py               RSI, ATR, EMA, BB, MACD, Stochastic
│   ├── volume_microstructure.py   Volume normalization features
│   ├── multi_timeframe.py         M15/H1/H4 context alignment
│   ├── sessions.py                Trading session encoding
│   ├── labeler.py                 Triple Barrier label generator
│   ├── correlation_pruner.py      Multicollinearity feature pruner
│   └── pipeline.py                Orchestrates full feature pipeline
│
├── training/                      Training infrastructure
│   ├── train.py                   Full training loop with AMP
│   ├── dataset.py                 TimeSeriesSequenceDataset
│   ├── loss.py                    Multi-task Focal + Huber loss
│   ├── walkforward.py             Walk-forward OOS validation
│   ├── hyperparam_search.py       Optuna HPO search
│   ├── run_experiments.py         Experiment runner
│   ├── benchmark.py               Baseline model benchmarks
│   └── diagnose.py                Training diagnostics + plots
│
├── collectors/                    Data collection from MT5
├── execution/                     Order management utilities
├── backtest/                      Backtesting engine
├── utils/                         Shared utilities
│
├── saved_models/                  Trained model artifacts
│   ├── mamba_xauusd_best.pt       Best model checkpoint
│   ├── feature_scaler.joblib      RobustScaler
│   └── correlation_pruner.joblib  Pruner artifact
│
├── data/                          Data storage
│   ├── raw/                       Raw OHLCV from MT5
│   ├── processed/                 Feature matrices (X, y)
│   ├── merged/                    Multi-TF merged datasets
│   └── economic_calendar.csv      News event calendar
│
├── MambaGoldEA.mq5                MQL5 EA source code
├── MambaGoldEA.ex5                Compiled EA binary
└── MambaGoldEA.log                EA execution log
```

---

## 🛣️ Further Updates and Roadmap

### Short-Term (Immediate)

- Retrain with more data — extend historical data from 2020 to present
- Raise OOS confidence — current out-of-sample max softmax is ~0.43; target >= 0.55
- Walk-forward parameter optimization — automated weekly refit cycle
- Longer dry-run testing — 2 to 4 weeks of signal logging without orders

### Medium-Term

- Multi-symbol expansion — extend to EURUSD, US30, or crude oil (WTI)
- Online learning — fine-tune model weights incrementally as new data arrives
- Ensemble confidence voting — combine multiple model checkpoints
- Sharpe-weighted loss — replace direction focal loss with risk-adjusted return optimization
- Attention regime gate — inject regime embedding into the direction head

### Long-Term

- Reinforcement learning layer — PPO-based execution agent on top of the Mamba signal
- Full portable VPS deployment — Docker container with MT5 Wine + Python bot
- Web dashboard — Real-time signal monitoring, PnL tracking, drawdown display
- Order book microstructure features — Tick-level bid/ask imbalance as model input
- CUDA-optimized Mamba kernels — Replace vectorized scan with mamba-ssm CUDA kernels for 10x speed on GPU

---

## ⚠️ Important Warnings

**THIS IS EXPERIMENTAL SOFTWARE. TRADE AT YOUR OWN RISK.**

- The current model checkpoint is **weak out-of-sample** — live confidence frequently stays below the entry threshold (bot mostly emits HOLD).
- **Do NOT trade real money** until the model demonstrates consistent OOS performance above the confidence threshold.
- **Do NOT run Python live mode and the MQL5 EA simultaneously** — they will conflict.
- **Keep the MT5 terminal open** at all times while the bot is running.
- The bot enforces risk limits, but no risk system is foolproof on Black Swan events.
- Always test on a **demo account first**.

---

## 📋 Current Status

| Component | Status |
|---|---|
| Bot infrastructure | Complete |
| Feature pipeline | Complete |
| Mamba model architecture | Complete |
| Training pipeline | Complete |
| Risk management | Complete |
| MQL5 EA bridge | Complete |
| Model performance (OOS) | Needs improvement |
| Live trading readiness | Not yet recommended |

---

*Built with PyTorch · MetaTrader 5 · Mamba Selective State Space Models*
