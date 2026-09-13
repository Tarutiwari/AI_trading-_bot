"""
Production Training Pipeline for Mamba Trading Model.
Handles chronological splitting, robust scaling, Cosine Annealing with warmup,
early stopping, AMP mixed-precision, and best model checkpointing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import sys

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Enable maximum CPU parallelism on Windows
if hasattr(os, "cpu_count") and os.cpu_count():
    torch.set_num_threads(min(os.cpu_count(), 8))

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CONFIG, BotConfig
from models.mamba_model import MambaTradingModel
from training.dataset import TimeSeriesSequenceDataset
from training.loss import MultiTaskTradingLoss


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup from 0 to base LR over warmup_steps."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 1e-6):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            factor = step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return [max(self.eta_min, base_lr * factor) for base_lr in self.base_lrs]


class ModelTrainer:
    """
    Manages the lifecycle of model training, validation, checkpoint serialization,
    and automatic training resumption.
    """

    def __init__(self, config: BotConfig = CONFIG, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler = RobustScaler()
        self.vol_scaler = None
        self.use_amp = self.device.type == "cuda"
        self.amp_scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        print(f"[INFO] Initialized ModelTrainer on device: {self.device} "
              f"(CPU Threads: {torch.get_num_threads()}, AMP: {self.use_amp})")

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed features and labels from disk."""
        proc_dir = self.config.paths.PROCESSED_DATA_DIR
        x_path = proc_dir / "X_features.parquet"
        y_path = proc_dir / "y_labels.parquet"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Processed data files not found in {proc_dir}. "
                f"Please run features/pipeline.py first."
            )

        X = pd.read_parquet(x_path)
        y = pd.read_parquet(y_path)
        print(f"[INFO] Loaded processed dataset: {len(X):,} samples x {X.shape[1]} features.")
        return X, y

    def prepare_dataloaders(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
        """
        Chronological 70% Train / 15% Val / 15% Test split with zero-leakage scaling.
        """
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train_raw = X.iloc[:train_end].values
        y_train = y.iloc[:train_end]

        X_val_raw = X.iloc[train_end:val_end].values
        y_val = y.iloc[train_end:val_end]

        X_test_raw = X.iloc[val_end:].values
        y_test = y.iloc[val_end:]

        # Fit scaler ONLY on train split
        X_train = self.scaler.fit_transform(X_train_raw)
        X_val = self.scaler.transform(X_val_raw)
        X_test = self.scaler.transform(X_test_raw)

        # Standardize volatility target before training
        self.vol_scaler = StandardScaler()
        y_train_vol = y_train["target_volatility"].values.reshape(-1, 1)
        y_val_vol = y_val["target_volatility"].values.reshape(-1, 1)
        y_test_vol = y_test["target_volatility"].values.reshape(-1, 1)

        y_train_vol_scaled = self.vol_scaler.fit_transform(y_train_vol).flatten()
        y_val_vol_scaled = self.vol_scaler.transform(y_val_vol).flatten()
        y_test_vol_scaled = self.vol_scaler.transform(y_test_vol).flatten()

        # Scaler is saved after training completes (see train_model)
        self._scaler_path = self.config.paths.CHECKPOINTS_DIR / "feature_scaler.joblib"

        seq_len = self.config.features.LOOKBACK_SEQUENCE_LENGTH

        train_ds = TimeSeriesSequenceDataset(
            X_train,
            y_train["target_direction"].values,
            y_train_vol_scaled,
            y_train["target_regime"].values,
            seq_len=seq_len,
        )
        val_ds = TimeSeriesSequenceDataset(
            X_val,
            y_val["target_direction"].values,
            y_val_vol_scaled,
            y_val["target_regime"].values,
            seq_len=seq_len,
        )
        test_ds = TimeSeriesSequenceDataset(
            X_test,
            y_test["target_direction"].values,
            y_test_vol_scaled,
            y_test["target_regime"].values,
            seq_len=seq_len,
        )

        batch_size = self.config.mamba.BATCH_SIZE
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        print(f"[INFO] Data splits created: Train={len(train_ds):,}, Val={len(val_ds):,}, Test={len(test_ds):,}")
        return train_loader, val_loader, test_loader, X.shape[1]

    def _train_one_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: MultiTaskTradingLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Tuple[float, float, float, float]:
        """Run one training epoch. Steps scheduler per-batch. Returns avg losses."""
        model.train()
        train_loss = 0.0
        train_loss_dir = 0.0
        train_loss_vol = 0.0
        train_loss_reg = 0.0

        pbar = tqdm(train_loader, desc="  Train", leave=False, dynamic_ncols=True)

        for step, (x_b, y_dir_b, y_vol_b, y_reg_b) in enumerate(pbar, 1):
            x_b = x_b.to(self.device)
            targets = {
                "direction": y_dir_b.to(self.device),
                "volatility": y_vol_b.to(self.device),
                "regime": y_reg_b.to(self.device),
            }

            optimizer.zero_grad(set_to_none=True)

            # AMP forward pass
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(x_b)
                    loss, loss_dict = criterion(preds, targets)
                self.amp_scaler.scale(loss).backward()
                self.amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.amp_scaler.step(optimizer)
                self.amp_scaler.update()
            else:
                preds = model(x_b)
                loss, loss_dict = criterion(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Step scheduler per batch (not per epoch!)
            if scheduler is not None:
                scheduler.step()

            train_loss += loss.item()
            train_loss_dir += loss_dict["direction_loss"]
            train_loss_vol += loss_dict["volatility_loss"]
            train_loss_reg += loss_dict["regime_loss"]

            if step % 50 == 0 or step == len(train_loader):
                cur_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dir": f"{loss_dict['direction_loss']:.3f}",
                    "lr": f"{cur_lr:.2e}",
                })

        n = len(train_loader)
        return train_loss / n, train_loss_dir / n, train_loss_vol / n, train_loss_reg / n

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        in_features: int,
        alpha_direction: Optional[float] = None,
        beta_volatility: Optional[float] = None,
        gamma_regime: Optional[float] = None,
        experiment_name: Optional[str] = None,
        patience: Optional[int] = None,
        max_epochs: Optional[int] = None,
        save_best_by: str = "val_f1",
    ) -> MambaTradingModel:
        """
        Executes complete training loop with warmup + Cosine Annealing and early stopping.

        Args:
            save_best_by: Metric to select best model: "val_loss", "val_acc", or "val_f1".
                          Default "val_f1" is best for imbalanced classification.
        """
        m_cfg = self.config.mamba
        model = MambaTradingModel(
            in_features=in_features,
            d_model=m_cfg.D_MODEL,
            n_layers=m_cfg.N_LAYERS,
            d_state=m_cfg.D_STATE,
            d_conv=m_cfg.D_CONV,
            expand=m_cfg.EXPAND_FACTOR,
            dropout=m_cfg.DROPOUT,
        ).to(self.device)

        # Attach vol_scaler to model so predictions can be unscaled
        model.vol_scaler = self.vol_scaler

        # Compute inverse-frequency class weights dynamically
        train_targets = train_loader.dataset.target_directions
        class_counts = torch.bincount(train_targets, minlength=3).float()
        class_counts = torch.clamp(class_counts, min=1.0)
        weights = len(train_targets) / (3.0 * class_counts)
        direction_weights = weights.to(self.device)
        print(f"[INFO] Computed directional class weights: {direction_weights.cpu().tolist()}")

        alpha_dir = alpha_direction if alpha_direction is not None else m_cfg.ALPHA_DIRECTION
        beta_vol = beta_volatility if beta_volatility is not None else m_cfg.BETA_VOLATILITY
        gamma_reg = gamma_regime if gamma_regime is not None else m_cfg.GAMMA_REGIME

        criterion = MultiTaskTradingLoss(
            alpha_direction=alpha_dir,
            beta_volatility=beta_vol,
            gamma_regime=gamma_reg,
            direction_weights=direction_weights,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=m_cfg.LEARNING_RATE,
            weight_decay=m_cfg.WEIGHT_DECAY,
        )

        # Cosine schedule with linear warmup
        effective_epochs = max_epochs or m_cfg.MAX_EPOCHS
        warmup_steps = int(len(train_loader) * 0.05)  # ~5% of total steps (1 epoch warmup)
        total_steps = len(train_loader) * effective_epochs
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps, eta_min=1e-6
        )

        best_val_loss = float("inf")
        best_metric_value = -float("inf")
        patience_counter = 0
        effective_patience = patience if patience is not None else m_cfg.EARLY_STOPPING_PATIENCE
        
        best_ckpt_name = f"mamba_xauusd_best_{experiment_name}.pt" if experiment_name else "mamba_xauusd_best.pt"
        latest_ckpt_name = f"mamba_xauusd_latest_{experiment_name}.pt" if experiment_name else "mamba_xauusd_latest.pt"
        best_checkpoint_path = self.config.paths.CHECKPOINTS_DIR / best_ckpt_name
        latest_checkpoint_path = self.config.paths.CHECKPOINTS_DIR / latest_ckpt_name
        
        if experiment_name:
            self._scaler_path = self.config.paths.CHECKPOINTS_DIR / f"feature_scaler_{experiment_name}.joblib"

        print(f"\n=======================================================")
        print(f"  Starting Mamba Training ({effective_epochs} Epochs Max | Batch Size: {m_cfg.BATCH_SIZE})")
        print(f"  Best model selection: {save_best_by} | Early stopping patience: {effective_patience}")
        print(f"  Warmup steps: {warmup_steps:,} | Total steps: {total_steps:,}")
        print(f"  AMP enabled: {self.use_amp} | LR stepping: per-batch")
        print(f"=======================================================")

        def dict_to_cpu(d):
            if isinstance(d, dict):
                return {k: dict_to_cpu(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [dict_to_cpu(v) for v in d]
            elif torch.is_tensor(d):
                return d.cpu()
            return d

        # Ensure checkpoints directory exists
        self.config.paths.ensure_dirs()

        for epoch in range(1, effective_epochs + 1):
            # 1. Training Phase (scheduler stepped per-batch inside)
            train_loss, train_loss_dir, train_loss_vol, train_loss_reg = self._train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler
            )

            # 2. Validation Phase
            val_loss, val_acc, val_f1, val_reg_acc, val_vol_mae, val_loss_dict = self.evaluate(
                model, val_loader, criterion, epoch=epoch
            )

            # Determine which metric to track for best model
            if save_best_by == "val_f1":
                current_metric = val_f1
                metric_name = "F1"
            elif save_best_by == "val_acc":
                current_metric = val_acc
                metric_name = "Acc"
            else:
                current_metric = -val_loss  # Lower loss = higher negative
                metric_name = "Loss"

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:02d}/{effective_epochs:02d}] | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} (Dir: {train_loss_dir:.3f}, Vol: {train_loss_vol:.5f}) | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | "
                f"Reg Acc: {val_reg_acc*100:.2f}% | Vol MAE: {val_vol_mae:.5f}"
            )
            pc_f1 = val_loss_dict.get("per_class_f1", None)
            if pc_f1 is not None and len(pc_f1) == 3:
                print(f"  --> Per-class F1: Down={pc_f1[0]:.4f} | Flat={pc_f1[1]:.4f} | Up={pc_f1[2]:.4f}")

            # Prepare device-agnostic checkpoint
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": dict_to_cpu(model.state_dict()),
                "optimizer_state_dict": dict_to_cpu(optimizer.state_dict()),
                "in_features": in_features,
                "config": m_cfg,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_reg_acc": val_reg_acc,
                "vol_scaler": self.vol_scaler,
                "save_best_by": save_best_by,
            }

            # Always save latest checkpoint after each epoch
            torch.save(checkpoint_data, latest_checkpoint_path)

            # 3. Best Model Checkpointing & Early Stopping
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(checkpoint_data, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint ({metric_name}={current_metric:.4f})")
            else:
                patience_counter += 1
                print(f"  [INFO] No improvement in {metric_name} ({patience_counter}/{effective_patience} patience)")
                if patience_counter >= effective_patience:
                    print(f"\n  [EARLY STOP] No improvement for {effective_patience} epochs. Stopping.")
                    break

        # Load best checkpoint
        checkpoint = torch.load(
            best_checkpoint_path, 
            map_location=self.device,
            weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "vol_scaler" in checkpoint:
            self.vol_scaler = checkpoint["vol_scaler"]
            model.vol_scaler = checkpoint["vol_scaler"]
        print(f"[SUCCESS] Loaded best model checkpoint from Epoch {checkpoint['epoch']} "
              f"({checkpoint.get('save_best_by', 'val_f1')}={best_metric_value:.4f}).")

        # Save scaler only after training is confirmed complete
        joblib.dump(self.scaler, self._scaler_path)
        print(f"[SAVED] Feature scaler saved -> {self._scaler_path.name}")

        # Save volatility scaler
        vol_scaler_path = self.config.paths.CHECKPOINTS_DIR / (f"volatility_scaler_{experiment_name}.joblib" if experiment_name else "volatility_scaler.joblib")
        joblib.dump(self.vol_scaler, vol_scaler_path)
        print(f"[SAVED] Volatility scaler saved -> {vol_scaler_path.name}")
        return model

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: MultiTaskTradingLoss,
        epoch: Optional[int] = None,
    ) -> Tuple[float, float, float, float, float, Dict[str, float]]:
        """
        Evaluates model performance across all tasks.
        Returns: (avg_loss, accuracy, macro_f1, regime_accuracy, vol_mae, loss_dict)
        """
        model.eval()
        total_loss = 0.0
        total_loss_dir = 0.0
        total_loss_vol = 0.0
        total_loss_reg = 0.0
        all_preds_dir = []
        all_trues_dir = []
        all_preds_vol = []
        all_trues_vol = []
        all_preds_regime = []
        all_trues_regime = []

        desc = f"Epoch [{epoch:02d}] Val" if epoch is not None else "Evaluating"
        val_pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for x_b, y_dir_b, y_vol_b, y_reg_b in val_pbar:
                x_b = x_b.to(self.device)
                targets = {
                    "direction": y_dir_b.to(self.device),
                    "volatility": y_vol_b.to(self.device),
                    "regime": y_reg_b.to(self.device),
                }
                preds = model(x_b)
                loss, loss_dict = criterion(preds, targets)
                total_loss += loss.item()
                total_loss_dir += loss_dict["direction_loss"]
                total_loss_vol += loss_dict["volatility_loss"]
                total_loss_reg += loss_dict["regime_loss"]

                pred_dirs = preds["direction_logits"].argmax(dim=-1).cpu().numpy()
                all_preds_dir.extend(pred_dirs)
                all_trues_dir.extend(y_dir_b.numpy())

                all_preds_vol.extend(preds["volatility"].cpu().numpy().flatten())
                all_trues_vol.extend(y_vol_b.numpy().flatten())

                pred_regimes = preds["regime_logits"].argmax(dim=-1).cpu().numpy()
                all_preds_regime.extend(pred_regimes)
                all_trues_regime.extend(y_reg_b.numpy())

        n = len(loader)
        avg_loss = total_loss / n
        avg_loss_dir = total_loss_dir / n
        avg_loss_vol = total_loss_vol / n
        avg_loss_reg = total_loss_reg / n

        acc = accuracy_score(all_trues_dir, all_preds_dir)
        f1 = f1_score(all_trues_dir, all_preds_dir, average="macro", zero_division=0)
        per_class_f1 = f1_score(all_trues_dir, all_preds_dir, average=None, zero_division=0)
        reg_acc = accuracy_score(all_trues_regime, all_preds_regime)

        if getattr(self, "vol_scaler", None) is not None:
            all_preds_vol_np = np.array(all_preds_vol).reshape(-1, 1)
            all_trues_vol_np = np.array(all_trues_vol).reshape(-1, 1)
            all_preds_vol_unscaled = self.vol_scaler.inverse_transform(all_preds_vol_np).flatten()
            all_trues_vol_unscaled = self.vol_scaler.inverse_transform(all_trues_vol_np).flatten()
            vol_mae = mean_absolute_error(all_trues_vol_unscaled, all_preds_vol_unscaled)
        else:
            vol_mae = mean_absolute_error(all_trues_vol, all_preds_vol)

        return avg_loss, acc, f1, reg_acc, vol_mae, {
            "direction_loss": avg_loss_dir,
            "volatility_loss": avg_loss_vol,
            "regime_loss": avg_loss_reg,
            "per_class_f1": per_class_f1,
        }


if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        X, y = trainer.load_processed_data()
        train_l, val_l, test_l, num_feat = trainer.prepare_dataloaders(X, y)
        model = trainer.train_model(train_l, val_l, num_feat, save_best_by="val_f1")
    except FileNotFoundError as e:
        print(f"[NOTE] {e}")
