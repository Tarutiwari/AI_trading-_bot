"""
Centralized Configuration for Gold (XAUUSD) Mamba Trading Bot.
Contains typed settings for data collection, feature engineering,
Mamba model architecture, risk management, and MT5 live execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class PathConfig:
    """Directory paths for the project."""
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MERGED_DATA_DIR: Path = DATA_DIR / "merged"
    CHECKPOINTS_DIR: Path = BASE_DIR / "saved_models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    def ensure_dirs(self) -> None:
        """Create directories if they do not exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MERGED_DATA_DIR,
            self.CHECKPOINTS_DIR,
            self.LOGS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class InstrumentConfig:
    """Settings for the Gold instrument and MT5 connection."""
    # Priority list of symbols to match across different MT5 brokers
    SYMBOL_CANDIDATES: List[str] = field(
        default_factory=lambda: [
            "XAUUSD.t",
            "XAUUSD",
            "GOLD",
            "XAUUSDm",
            "XAUUSD.raw",
            "XAUUSD.pro",
            "XAUUSD_i",
        ]
    )
    PRIMARY_TIMEFRAME: str = "M5"  # Main decision timeframe
    CONTEXT_TIMEFRAMES: List[str] = field(default_factory=lambda: ["M15", "H1", "H4"])
    
    # Point/pip conversion (Standard Gold: 1 lot = 100 oz, 1 point = $0.01)
    LOT_STEP: float = 0.01
    MIN_LOT: float = 0.01
    MAX_LOT: float = 5.00
    CONTRACT_SIZE: float = 100.0


@dataclass
class DataCollectionConfig:
    """Parameters for historical data collection from MT5 & Economic Calendar."""
    START_DATE: str = "2020-01-01"
    END_DATE: Optional[str] = None  # None = fetch up to current datetime
    MAX_BARS_PER_REQUEST: int = 100_000
    TIMEZONE: str = "UTC"


@dataclass
class FeatureConfig:
    """Parameters for stationary feature engineering & normalizations."""
    LOOKBACK_SEQUENCE_LENGTH: int = 60  # Number of past bars (N) fed to Mamba
    
    # Technical Indicators
    RSI_PERIOD: int = 14
    STOCH_RSI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    EMA_FAST: int = 9
    EMA_MEDIUM: int = 21
    EMA_SLOW: int = 50
    EMA_TREND: int = 200
    
    # Broker-Agnostic Volume Normalization Windows
    VOLUME_ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [20, 50, 200])
    
    # Multicollinearity Reduction
    CORRELATION_THRESHOLD: float = 0.80  # Drop features with |r| > 0.80
    
    # Session Timing (UTC hours)
    ASIAN_SESSION: tuple = (0, 8)      # 00:00 - 08:00 UTC
    LONDON_SESSION: tuple = (7, 16)    # 07:00 - 16:00 UTC
    NY_SESSION: tuple = (12, 21)       # 12:00 - 21:00 UTC
    OVERLAP_SESSION: tuple = (12, 16)  # 12:00 - 16:00 UTC (London + NY overlap)


@dataclass
class LabelingConfig:
    """Triple Barrier Labeling parameters (Marcos Lopez de Prado)."""
    TP_ATR_MULTIPLIER: float = 1.5   # Upper Take-Profit barrier = +1.5 * ATR
    SL_ATR_MULTIPLIER: float = 1.5   # Lower Stop-Loss barrier = -1.5 * ATR
    MAX_HOLDING_BARS: int = 24       # Time barrier: 24 bars on M5 = 2 hours max
    MIN_RETURN_THRESHOLD: float = 0.0005  # Min price return for active label


@dataclass
class MambaModelConfig:
    """Mamba (Selective State Space S6) Architecture Hyperparameters."""
    D_MODEL: int = 64          # Hidden dimension
    N_LAYERS: int = 3          # Number of stacked Mamba blocks
    D_STATE: int = 16          # State space SSM expansion dimension
    D_CONV: int = 4            # Local 1D Convolution kernel width
    EXPAND_FACTOR: int = 2     # Block expansion factor
    DROPOUT: float = 0.15      # Dropout rate
    
    # Training Parameters
    LEARNING_RATE: float = 5e-4       # Lower LR for more stable convergence with warmup
    WEIGHT_DECAY: float = 1e-4
    BATCH_SIZE: int = 256              # Reduced from 512 for better generalization on noisy financial data
    MAX_EPOCHS: int = 50               # More epochs; early stopping handles termination
    EARLY_STOPPING_PATIENCE: int = 10  # Patient early stopping to avoid premature exit
    
    # Multi-Task Loss Weights
    ALPHA_DIRECTION: float = 1.0   # Focal Loss weight for Direction
    BETA_VOLATILITY: float = 0.5    # Huber Loss weight for Expected Volatility
    GAMMA_REGIME: float = 0.3       # Cross-Entropy weight for Regime Classification


@dataclass
class RiskManagementConfig:
    """Institutional Risk Rules, News Filter & Execution Boundaries."""
    MAX_RISK_PER_TRADE_PCT: float = 0.01   # Risk at most 1.0% of account equity ($5 on $500)
    MAX_DAILY_DRAWDOWN_PCT: float = 0.03   # Stop trading for the day if loss exceeds 3.0% ($15 on $500)
    MAX_OPEN_TRADES: int = 1               # Strictly single-position exposure
    MAX_CONSECUTIVE_LOSSES: int = 5         # Pause trading after 5 consecutive losing trades in a day
    
    # Execution Guards
    MAX_ALLOWED_SPREAD_POINTS: float = 40.0  # Max spread in points allowed for entry
    MIN_CONFIDENCE_THRESHOLD: float = 0.65   # Model softmax probability required to trigger
    
    # Dynamic Order Management
    BREAKEVEN_R_MULTIPLE: float = 1.0   # Move SL to breakeven once price reaches 1R profit
    ENABLE_TRAILING_STOP: bool = True
    TRAILING_ATR_MULTIPLE: float = 1.5  # Trail stop at 1.5 * ATR behind peak price
    
    # Macroeconomic News Circuit Breakers (Dynamic Event-Specific Windows)
    FOMC_FREEZE_BEFORE_MINUTES: int = 45  # Freeze 45 mins before FOMC Rate Decision
    FOMC_FREEZE_AFTER_MINUTES: int = 60   # Wait 60 mins after FOMC (Powell Press Conference)
    NFP_CPI_FREEZE_BEFORE_MINUTES: int = 15  # Freeze 15 mins before NFP / CPI
    NFP_CPI_FREEZE_AFTER_MINUTES: int = 30   # Wait 30 mins after NFP / CPI
    DEFAULT_NEWS_FREEZE_BEFORE_MINUTES: int = 15  # General high-impact event freeze
    DEFAULT_NEWS_FREEZE_AFTER_MINUTES: int = 15
    SURPRISE_DECAY_HALF_LIFE_HOURS: float = 4.0   # Exponential decay half-life for surprise impact
    
    # Legacy aliases for backwards compatibility
    NEWS_FREEZE_MINUTES_BEFORE: int = 15
    NEWS_FREEZE_MINUTES_AFTER: int = 15
    
    # MT5 Identification
    MAGIC_NUMBER: int = 777001
    ORDER_COMMENT: str = "Mamba_XAUUSD_Bot"


@dataclass
class BotConfig:
    """Global composite configuration container."""
    paths: PathConfig = field(default_factory=PathConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    data: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    mamba: MambaModelConfig = field(default_factory=MambaModelConfig)
    risk: RiskManagementConfig = field(default_factory=RiskManagementConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()


# Global config instance for import across modules
CONFIG = BotConfig()
