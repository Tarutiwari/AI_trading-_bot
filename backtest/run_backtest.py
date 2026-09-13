"""
Standalone Backtest Runner for Gold (XAUUSD) Mamba Model.
Loads out-of-sample test data, executes transaction-cost-aware backtest,
and prints complete institutional performance tearsheet.
"""

from pathlib import Path
import joblib
import sys

import pandas as pd
import numpy as np
import torch

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from models.mamba_model import MambaTradingModel
from backtest.backtester import CostAwareBacktester


def run():
    print("\n=======================================================")
    print("       RUNNING OUT-OF-SAMPLE MAMBA BACKTEST            ")
    print("=======================================================")

    proc_dir = CONFIG.paths.PROCESSED_DATA_DIR
    raw_dir = CONFIG.paths.RAW_DATA_DIR
    ckpt_dir = CONFIG.paths.CHECKPOINTS_DIR

    x_path = proc_dir / "X_features.parquet"
    ckpt_path = ckpt_dir / "mamba_xauusd_best.pt"
    scaler_path = ckpt_dir / "feature_scaler.joblib"

    if not x_path.exists() or not ckpt_path.exists() or not scaler_path.exists():
        print("[ERROR] Required data or model checkpoint files not found.")
        print("Please ensure you have run features/pipeline.py and training/train.py first.")
        return

    # 1. Load Data
    X_df = pd.read_parquet(x_path)
    scaler = joblib.load(scaler_path)

    # 2. Load Raw Primary M5 Market Data for actual OHLC & Spread prices
    raw_files = list(raw_dir.glob("*_M5_raw.parquet"))
    if not raw_files:
        raw_files = list(raw_dir.glob("*_M5_raw.csv"))
        if not raw_files:
            print("[ERROR] Could not locate raw M5 price file.")
            return
        df_market = pd.read_csv(raw_files[0], index_col=0, parse_dates=True)
    else:
        df_market = pd.read_parquet(raw_files[0])

    # Align market DF to valid features index
    common_idx = X_df.index.intersection(df_market.index)
    X_df = X_df.loc[common_idx]
    df_market = df_market.loc[common_idx]

    # Use the 15% Out-of-Sample Test Split
    test_start = int(len(X_df) * 0.85)
    X_test_df = X_df.iloc[test_start:]
    df_test_market = df_market.iloc[test_start:]

    # Scale features
    X_test_scaled = scaler.transform(X_test_df.values)

    # 3. Load Trained Mamba Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    in_features = checkpoint["in_features"]

    model = MambaTradingModel(
        in_features=in_features,
        d_model=CONFIG.mamba.D_MODEL,
        n_layers=CONFIG.mamba.N_LAYERS,
        d_state=CONFIG.mamba.D_STATE,
        d_conv=CONFIG.mamba.D_CONV,
        expand=CONFIG.mamba.EXPAND_FACTOR,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if "vol_scaler" in checkpoint:
        model.vol_scaler = checkpoint["vol_scaler"]
    model.eval()

    # 4. Execute Backtest
    backtester = CostAwareBacktester(
        config=CONFIG,
        initial_capital=10_000.0,
        commission_per_lot=7.0,
        slippage_points=10.0,
    )

    equity_series, trades, metrics = backtester.run_backtest(
        df_market=df_test_market,
        X_features=X_test_scaled,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    run()
