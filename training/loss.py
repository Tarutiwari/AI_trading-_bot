"""
Multi-Task Loss Functions for Trading Models.
Includes Focal Loss for class-imbalanced directional prediction
and Huber / Smooth L1 Loss for dynamic volatility regression.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss to address class imbalance between choppy flat markets
    and strong directional expansion moves.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_classes)
        targets: (batch_size)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class MultiTaskTradingLoss(nn.Module):
    """
    Weighted composite loss for Direction, Volatility, and Regime prediction.
    """

    def __init__(
        self,
        alpha_direction: float = 1.0,
        beta_volatility: float = 0.5,
        gamma_regime: float = 0.3,
        focal_gamma: float = 2.0,
        direction_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha_direction
        self.beta = beta_volatility
        self.gamma = gamma_regime

        self.direction_loss_fn = FocalLoss(gamma=focal_gamma, alpha=direction_weights)
        self.volatility_loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.regime_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes total loss and individual loss components.
        """
        loss_dir = self.direction_loss_fn(
            predictions["direction_logits"], targets["direction"]
        )
        loss_vol = self.volatility_loss_fn(
            predictions["volatility"], targets["volatility"]
        )
        loss_reg = self.regime_loss_fn(
            predictions["regime_logits"], targets["regime"]
        )

        total_loss = (
            (self.alpha * loss_dir)
            + (self.beta * loss_vol)
            + (self.gamma * loss_reg)
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "direction_loss": loss_dir.item(),
            "volatility_loss": loss_vol.item(),
            "regime_loss": loss_reg.item(),
        }

        return total_loss, loss_dict
