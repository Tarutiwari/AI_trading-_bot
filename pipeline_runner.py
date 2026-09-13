"""
Interactive Master Orchestrator for Gold (XAUUSD) Mamba Trading Bot.
Provides a simple CLI menu to run data collection, feature engineering,
model training, benchmark comparison, backtesting, or live execution.
"""

from pathlib import Path
import sys

# Ensure root directory is on python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config import CONFIG
from collectors.mt5_collector import MT5DataCollector
from features.pipeline import FeaturePipeline
from training.train import ModelTrainer
from training.benchmark import BenchmarkRunner
from backtest.run_backtest import run as run_backtest_script
from main_live import LiveTradingEngine


def step_1_collect_data():
    print("\n[STEP 1] Starting MT5 Multi-Timeframe Data Collection...")
    collector = MT5DataCollector(CONFIG)
    try:
        if collector.connect():
            symbol = collector.resolve_symbol()
            if symbol:
                data = collector.collect_all_timeframes()
                print(f"\n[DONE] Collected {len(data)} timeframes successfully.")
    finally:
        collector.shutdown()


def step_2_build_features():
    print("\n[STEP 2] Running Feature Engineering & Triple Barrier Labeling...")
    collector = MT5DataCollector(CONFIG)
    try:
        if collector.connect():
            symbol = collector.resolve_symbol()
            if symbol:
                raw_data = collector.collect_all_timeframes(save_to_disk=False)
                pipeline = FeaturePipeline(CONFIG)
                X, y = pipeline.build_dataset(raw_data, is_training=True, save_to_disk=True)
                print(f"\n[DONE] Successfully processed {len(X):,} samples x {X.shape[1]} features.")
    finally:
        collector.shutdown()


def step_3_train_model():
    print("\n[STEP 3] Training Mamba Multi-Task Architecture...")
    trainer = ModelTrainer(CONFIG)
    X, y = trainer.load_processed_data()
    train_l, val_l, test_l, in_feats = trainer.prepare_dataloaders(X, y)
    model = trainer.train_model(train_l, val_l, in_feats)
    print("\n[DONE] Model training complete and best checkpoint saved.")


def step_4_benchmark():
    print("\n[STEP 4] Running Empirical Benchmark: Mamba vs. GRU...")
    trainer = ModelTrainer(CONFIG)
    runner = BenchmarkRunner(trainer)
    runner.run_benchmark(epochs=15)


def step_5_backtest():
    print("\n[STEP 5] Running Cost-Aware Event-Driven Backtest...")
    run_backtest_script()


def step_6_live():
    print("\n[STEP 6] Launching Real-Time MT5 Trading Engine...")
    engine = LiveTradingEngine(CONFIG)
    engine.start_loop(poll_interval_seconds=15)


def main():
    print("""
===============================================================
       MAMBA GOLD (XAUUSD) QUANTITATIVE TRADING BOT
===============================================================
  1. Collect Multi-Timeframe Data from MT5
  2. Extract Features & Generate Triple Barrier Labels
  3. Train Mamba Model (Multi-Task Learning)
  4. Benchmark Mamba vs. GRU Baseline (Empirical Proof)
  5. Run Cost-Aware Backtest (Out-of-Sample)
  6. Launch Live / Demo MT5 Trading Engine
  7. Run Full Pipeline (Steps 1 -> 5 End-to-End)
  0. Exit
===============================================================
    """)
    choice = input("Select an option [0-7]: ").strip()

    if choice == "1":
        step_1_collect_data()
    elif choice == "2":
        step_2_build_features()
    elif choice == "3":
        step_3_train_model()
    elif choice == "4":
        step_4_benchmark()
    elif choice == "5":
        step_5_backtest()
    elif choice == "6":
        step_6_live()
    elif choice == "7":
        step_1_collect_data()
        step_2_build_features()
        step_3_train_model()
        step_4_benchmark()
        step_5_backtest()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
