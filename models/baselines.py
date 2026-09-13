"""
Baseline Benchmark Models for Empirical Comparison against Mamba.
Includes Bidirectional GRU, LSTM, and LightGBM models with identical multi-task objectives.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUTradingModel(nn.Module):
    """
    Bidirectional GRU baseline model.
    Used to benchmark whether Mamba's Selective State Space outperforms standard RNNs.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.15,
        num_classes: int = 3,
        num_regimes: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Multi-task heads
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),
        )

        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_norm(self.input_proj(x))
        out, _ = self.gru(h)
        out = self.final_norm(out)
        last_step = out[:, -1, :]

        direction_logits = self.direction_head(last_step)
        volatility = self.volatility_head(last_step)
        regime_logits = self.regime_head(last_step.detach())

        return {
            "direction_logits": direction_logits,
            "direction_probs": F.softmax(direction_logits, dim=-1),
            "volatility": volatility,
            "regime_logits": regime_logits,
            "regime_probs": F.softmax(regime_logits, dim=-1),
        }
