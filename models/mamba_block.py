"""
Pure PyTorch Vectorized Selective State Space (S6) Mamba Block.
Designed for Windows CPU/CUDA environments with zero C++/CUDA compilation dependencies.
Linear-time O(L) complexity with dynamic input-dependent state compression.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Core Selective State Space (S6) mechanism.
    Discretizes continuous state equations using dynamic input projections:
    h_t = exp(Delta * A) * h_{t-1} + (Delta * B) * x_t
    y_t = C * h_t + D * x_t
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        # 1. State Matrix A (HiPPO Initialization: log of structured diagonal)
        # Parameterized as log(-A) to guarantee stability (A remains strictly negative)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # 2. Skip Connection D
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

        # 3. Dynamic Projections for Delta, B, C from input x
        self.x_proj = nn.Linear(d_inner, self.dt_rank + (2 * d_state), bias=False)

        # 4. Delta (dt) Projection
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialize dt_proj bias to span [dt_min, dt_max]
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Selective SSM.
        u: Input tensor of shape (batch, seq_len, d_inner)
        Returns: Output tensor of shape (batch, seq_len, d_inner)
        """
        batch_size, seq_len, _ = u.shape

        # A is strictly negative: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # Dynamic projections: (B, L, dt_rank + 2 * d_state)
        x_dbl = self.x_proj(u)
        delta_rank, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Discretization delta: (B, L, d_inner)
        delta = F.softplus(self.dt_proj(delta_rank))

        # Vectorized Selective Scan (Optimized for CPU/CUDA)
        # Discrete A_bar = exp(delta * A): (B, L, d_inner, d_state)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # Discrete B_bar * u: (B, L, d_inner, d_state)
        delta_B_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)

        # Vectorized step scan
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]  # (B, d_inner, d_state)
            # y_t = sum(h_t * C_t): (B, d_inner)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Add skip connection D * u
        y = y + (u * self.D)
        return y


class MambaBlock(nn.Module):
    """
    Standard Mamba Block with Gated Linear Unit and Causal 1D Convolution.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_conv = d_conv

        # Input projection into two parallel branches (2 * d_inner)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal Depthwise 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,  # Causal padding
        )

        # Selective State Space Model
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        _, seq_len, _ = x.shape

        # Linear projection into branch 1 (SSM) and branch 2 (Residual Gate)
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal Conv1D on branch 1
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim causal padding
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM
        y_ssm = self.ssm(x_conv)  # (B, L, d_inner)

        # Gated Linear Unit with Branch 2
        gated_out = y_ssm * F.silu(z_branch)

        # Output Projection & Dropout
        out = self.out_proj(gated_out)
        return self.dropout(out)
