"""
Multi-Task Mamba Neural Network Architecture for Gold (XAUUSD).
Outputs trade direction probability distribution, dynamic expected volatility,
and market regime classification.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_block import MambaBlock


class MambaTradingModel(nn.Module):
    """
    Complete Multi-Task Mamba Time Series Network.
    Processes historical multi-timeframe sequences (N x D) and generates
    trade signals, volatility magnitudes, and regime classifications.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.15,
        num_classes: int = 3,      # 0: Bearish / Sell, 1: Neutral / Chop, 2: Bullish / Buy
        num_regimes: int = 4,      # 0: Bull Trend, 1: Bear Trend, 2: Chop, 3: News Vol Spike
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.num_classes = num_classes

        # 1. Feature Projection & Input Normalization
        self.input_proj = nn.Linear(in_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # 2. Stacked Mamba Layers with Pre-LayerNorm & Residuals
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Final sequence LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # 3. Multi-Task Decision Heads
        # Head 1: Directional Signal Classifier (Bearish / Neutral / Bullish)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Head 2: Dynamic Expected Volatility / Excursion Regressor
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus(),  # Strictly positive volatility
        )

        # Head 3: Market Regime Classifier
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regimes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch_size, seq_len, in_features)
        Returns:
            - 'direction_logits': (batch_size, 3)
            - 'direction_probs': (batch_size, 3)
            - 'volatility': (batch_size, 1)
            - 'regime_logits': (batch_size, 4)
            - 'regime_probs': (batch_size, 4)
        """
        # Input projection & norm
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.input_dropout(h)

        # Pass through stacked Mamba blocks with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            residual = h
            h = residual + layer(norm(h))

        h = self.final_norm(h)

        # Extract last sequence bar representation (latest market state)
        last_step = h[:, -1, :]  # (batch_size, d_model)

        # Compute multi-task outputs
        direction_logits = self.direction_head(last_step)
        direction_probs = F.softmax(direction_logits, dim=-1)

        volatility = self.volatility_head(last_step)

        regime_logits = self.regime_head(last_step.detach())
        regime_probs = F.softmax(regime_logits, dim=-1)

        return {
            "direction_logits": direction_logits,
            "direction_probs": direction_probs,
            "volatility": volatility,
            "regime_logits": regime_logits,
            "regime_probs": regime_probs,
        }

    @torch.no_grad()
    def predict_trade_action(
        self, x: torch.Tensor, confidence_threshold: float = None
    ) -> Dict[str, any]:
        """
        High-level inference helper for live trading / backtesting.
        x: (1, seq_len, in_features)

        confidence_threshold: If None, falls back to CONFIG.risk.MIN_CONFIDENCE_THRESHOLD
                              so config.py changes automatically apply here too.
        """
        if confidence_threshold is None:
            try:
                from config import CONFIG
                confidence_threshold = CONFIG.risk.MIN_CONFIDENCE_THRESHOLD
            except Exception:
                confidence_threshold = 0.50  # safe fallback
        self.eval()
        out = self.forward(x)
        probs = out["direction_probs"][0].cpu().numpy()
        vol = out["volatility"][0, 0].item()
        if getattr(self, "vol_scaler", None) is not None:
            vol = float(self.vol_scaler.inverse_transform([[vol]])[0, 0])
        regime_id = int(out["regime_probs"][0].argmax().item())

        p_bearish = float(probs[0])
        p_neutral = float(probs[1])
        p_bullish = float(probs[2])

        action = "HOLD"
        confidence = p_neutral

        if p_bullish >= confidence_threshold and p_bullish > p_bearish:
            action = "BUY"
            confidence = p_bullish
        elif p_bearish >= confidence_threshold and p_bearish > p_bullish:
            action = "SELL"
            confidence = p_bearish

        return {
            "action": action,
            "confidence": confidence,
            "p_buy": p_bullish,
            "p_sell": p_bearish,
            "p_neutral": p_neutral,
            "expected_volatility": vol,
            "regime_id": regime_id,
        }
