"""
Empirical Benchmark Harness.
Compares Mamba (Selective State Space S6) vs. Bidirectional GRU Baseline
under identical data splits, multi-task objectives, and out-of-sample metrics.
"""

from pathlib import Path
import time
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error
import pandas as pd

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from models.mamba_model import MambaTradingModel
from models.baselines import GRUTradingModel
from training.dataset import TimeSeriesSequenceDataset
from training.loss import MultiTaskTradingLoss
from training.train import ModelTrainer


class BenchmarkRunner:
    """
    Automated comparison suite to validate whether Mamba outperforms traditional baselines.
    """

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.device = trainer.device

    def run_benchmark(self, epochs: int = 15) -> pd.DataFrame:
        """
        Trains and benchmarks Mamba vs GRU on the same training split
        and evaluates on the out-of-sample test split.
        """
        X, y = self.trainer.load_processed_data()
        train_l, val_l, test_l, in_feats = self.trainer.prepare_dataloaders(X, y)

        models = {
            "Mamba (Selective SSM)": MambaTradingModel(
                in_features=in_feats,
                d_model=64,
                n_layers=3,
                d_state=16,
                d_conv=4,
            ).to(self.device),
            "Bidirectional GRU": GRUTradingModel(
                in_features=in_feats,
                d_model=64,
                n_layers=2,
            ).to(self.device),
        }

        results = []

        for name, model in models.items():
            print(f"\n[BENCHMARK] Training {name} for {epochs} epochs...")
            criterion = MultiTaskTradingLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

            # Fast training
            for epoch in range(1, epochs + 1):
                model.train()
                for x_b, y_dir_b, y_vol_b, y_reg_b in train_l:
                    x_b = x_b.to(self.device)
                    targets = {
                        "direction": y_dir_b.to(self.device),
                        "volatility": y_vol_b.to(self.device),
                        "regime": y_reg_b.to(self.device),
                    }
                    optimizer.zero_grad()
                    preds = model(x_b)
                    loss, _ = criterion(preds, targets)
                    loss.backward()
                    optimizer.step()

            # Measure Inference Latency on Test Set
            model.eval()
            start_time = time.perf_counter()
            test_loss, test_acc, test_f1, test_reg_acc, test_vol_mae, _ = self.trainer.evaluate(
                model, test_l, criterion
            )
            total_time = time.perf_counter() - start_time
            latency_ms = (total_time / (len(test_l.dataset) or 1)) * 1000.0

            results.append({
                "Model Architecture": name,
                "OOS Direction Acc": f"{test_acc*100:.2f}%",
                "OOS Macro F1": f"{test_f1:.4f}",
                "OOS Vol MAE": f"{test_vol_mae:.5f}",
                "Latency (ms/candle)": f"{latency_ms:.3f} ms",
            })

        df_results = pd.DataFrame(results)
        print("\n=======================================================")
        print("          EMPIRICAL BENCHMARK SUMMARY (OOS)")
        print("=======================================================")
        print(df_results.to_string(index=False))
        return df_results


if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        runner = BenchmarkRunner(trainer)
        runner.run_benchmark(epochs=10)
    except FileNotFoundError as e:
        print(f"[NOTE] {e}")
