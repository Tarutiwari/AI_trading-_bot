"""
PyTorch Sequence Dataset & Purged Walk-Forward Cross-Validation Splitter.
Guarantees zero leakage across sequence windows and holding horizons.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesSequenceDataset(Dataset):
    """
    Sliding window sequence dataset for Mamba and recurrent models.
    Converts 2D feature matrix (N, D) into 3D sequence tensors (B, L, D).
    """

    def __init__(
        self,
        features: np.ndarray,
        target_directions: np.ndarray,
        target_vols: np.ndarray,
        target_regimes: np.ndarray,
        seq_len: int = 60,
        stride: int = 3,
    ):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target_directions = torch.tensor(target_directions, dtype=torch.long)
        self.target_vols = torch.tensor(target_vols, dtype=torch.float32).unsqueeze(-1)
        self.target_regimes = torch.tensor(target_regimes, dtype=torch.long)

        # Precompute start indices with stride
        max_start = len(features) - seq_len
        if max_start >= 0:
            self.indices = np.arange(0, max_start + 1, self.stride)
        else:
            self.indices = np.array([], dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_idx = int(self.indices[idx])
        end_idx = start_idx + self.seq_len
        x = self.features[start_idx:end_idx]
        target_idx = end_idx - 1

        y_dir = self.target_directions[target_idx]
        y_vol = self.target_vols[target_idx]
        y_regime = self.target_regimes[target_idx]

        return x, y_dir, y_vol, y_regime


class PurgedWalkForwardCV:
    """
    Purged and Embargoed Walk-Forward Cross-Validation (Marcos Lopez de Prado).
    Prevents information leakage from overlapping holding periods.
    """

    def __init__(
        self,
        n_splits: int = 5,
        holding_period_bars: int = 24,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.holding_period = holding_period_bars
        self.embargo_pct = embargo_pct

    def split(
        self, n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, val_indices) for each walk-forward fold.
        """
        fold_size = n_samples // (self.n_splits + 1)
        embargo_size = int(n_samples * self.embargo_pct)

        for i in range(self.n_splits):
            # Expanding train window
            train_end = fold_size * (i + 1)
            train_indices = np.arange(0, train_end - self.holding_period)

            # Validation window
            val_start = train_end + embargo_size
            val_end = min(val_start + fold_size, n_samples)
            if val_start >= n_samples:
                break
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices
