#!/usr/bin/env python3
"""
src/models/pi_cca.py — Physics-Informed Convolutional Cellular Automaton
========================================================================
**Novel hybrid architecture** — main contribution of this research.

Combines:
  1. **Differentiable Physics Branch** — encodes Rothermel fire spread
     equations with learnable correction parameters (α_w, α_s, α_m)
  2. **Data-Driven CNN Branch** — residual convolutional blocks that
     capture complex non-linear patterns not modelled by physics
  3. **Cross-Attention Fusion** — spatially adaptive fusion of physics
     and data-driven predictions via multi-head cross-attention
  4. **MC-Dropout Uncertainty** — epistemic uncertainty estimation
     through Monte Carlo dropout at inference time

The key insight is that classical CA models are interpretable but rigid,
while pure DL models are flexible but opaque.  PI-CCA combines the best
of both worlds: physical grounding + data-driven correction.

References
----------
  Rothermel, R.C. (1972). A mathematical model for predicting fire spread.
  Alexandridis, A. et al. (2008). A cellular automata model for forest
      fire spread prediction. Applied Math. & Computation.
  Raissi, M. et al. (2019). Physics-informed neural networks: A deep
      learning framework for solving forward and inverse problems. JCP.
  Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.
  Karniadakis, G because et al. (2021). Physics-informed machine learning.
      Nature Reviews Physics, 3, 422–440.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════
# DIFFERENTIABLE PHYSICS MODULE
# ══════════════════════════════════════════════════════════════

class DifferentiableRothermel(nn.Module):
    """Differentiable Rothermel fire spread with learnable corrections.

    Adapted for real satellite data (Next Day Wildfire Spread):
      - elevation → slope via Sobel gradient
      - wind speed (th) & direction (vs) from GRIDMET
      - humidity (sph) as moisture proxy
      - NDVI as vegetation/fuel proxy
      - ERC as fire-potential proxy
      - PrevFireMask as fire state

    Learnable correction factors:
      - α_w : wind influence correction
      - α_s : slope influence correction
      - α_m : moisture (humidity) sensitivity correction
      - α_v : vegetation (NDVI) reactivity correction
    """

    def __init__(self, learnable: bool = True):
        super().__init__()
        if learnable:
            self.alpha_wind = nn.Parameter(torch.tensor(1.0))
            self.alpha_slope = nn.Parameter(torch.tensor(1.0))
            self.alpha_moisture = nn.Parameter(torch.tensor(1.0))
            self.alpha_veg = nn.Parameter(torch.tensor(1.0))
            self.p_base = nn.Parameter(torch.tensor(0.58))
        else:
            self.register_buffer("alpha_wind", torch.tensor(1.0))
            self.register_buffer("alpha_slope", torch.tensor(1.0))
            self.register_buffer("alpha_moisture", torch.tensor(1.0))
            self.register_buffer("alpha_veg", torch.tensor(1.0))
            self.register_buffer("p_base", torch.tensor(0.58))

        # Sobel kernels for slope estimation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute physics-based spread probability map.

        Parameters
        ----------
        x : (B, C, H, W) — input tensor with all features

        Returns
        -------
        (B, 1, H, W) — physics-based fire probability
        """
        from config import CH

        # Extract channels using real-data indices
        elevation   = x[:, CH["elevation"]:CH["elevation"]+1]     # (B,1,H,W)
        wind_speed  = x[:, CH["wind_speed"]:CH["wind_speed"]+1]
        wind_dir    = x[:, CH["wind_direction"]:CH["wind_direction"]+1]
        humidity    = x[:, CH["humidity"]:CH["humidity"]+1]
        ndvi        = x[:, CH["ndvi"]:CH["ndvi"]+1]
        erc         = x[:, CH["erc"]:CH["erc"]+1]
        fire_state  = x[:, CH["prev_fire_mask"]:CH["prev_fire_mask"]+1]

        # ── Slope from elevation (Sobel gradient) ──
        dx = F.conv2d(elevation, self.sobel_x, padding=1)
        dy = F.conv2d(elevation, self.sobel_y, padding=1)
        slope_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)

        # ── Wind effect ──
        wind_magnitude = torch.abs(wind_speed) + 1e-6
        wind_effect = 1.0 + self.alpha_wind * 0.045 * wind_magnitude

        # ── Slope effect ──
        slope_effect = 1.0 + self.alpha_slope * 0.078 * slope_mag

        # ── Moisture damping (higher humidity → slower spread) ──
        # sph is specific humidity (kg/kg), typically 0.001–0.02
        moisture_damp = torch.clamp(
            1.0 - self.alpha_moisture * 50.0 * humidity, 0.01, 1.0
        )

        # ── Vegetation / fuel factor (higher NDVI → more fuel) ──
        veg_factor = self.alpha_veg * (0.3 + 0.7 * torch.sigmoid(ndvi * 3.0))

        # ── Base spread probability ──
        p_spread = torch.sigmoid(self.p_base) * wind_effect * slope_effect * \
                   moisture_damp * veg_factor

        # ── Only spread from burning cells ──
        burning_neighbours = F.max_pool2d(
            fire_state, kernel_size=3, stride=1, padding=1
        )

        p_physics = p_spread * burning_neighbours * (1.0 - fire_state)
        p_physics = torch.clamp(p_physics, 0, 1)

        # Existing fires persist
        p_physics = torch.max(p_physics, fire_state)

        return p_physics


# ══════════════════════════════════════════════════════════════
# RESIDUAL CNN BLOCK
# ══════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual convolutional block with BatchNorm."""

    def __init__(self, channels: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ══════════════════════════════════════════════════════════════
# SPATIAL CROSS-ATTENTION FUSION
# ══════════════════════════════════════════════════════════════

class SpatialCrossAttention(nn.Module):
    """Multi-head cross-attention between physics and data-driven features.

    Queries come from the physics branch, keys/values from the CNN branch.
    This lets the model learn WHERE the physics prediction needs correction.
    """

    def __init__(self, physics_dim: int, cnn_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = cnn_dim // n_heads
        assert cnn_dim % n_heads == 0, "cnn_dim must be divisible by n_heads"

        self.q_proj = nn.Conv2d(physics_dim, cnn_dim, 1)
        self.k_proj = nn.Conv2d(cnn_dim, cnn_dim, 1)
        self.v_proj = nn.Conv2d(cnn_dim, cnn_dim, 1)
        self.out_proj = nn.Conv2d(cnn_dim, cnn_dim, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, physics_feat: torch.Tensor, cnn_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        physics_feat : (B, P, H, W)
        cnn_feat : (B, D, H, W)

        Returns
        -------
        (B, D, H, W) — attention-weighted features
        """
        B, _, H, W = cnn_feat.shape
        N = H * W

        Q = self.q_proj(physics_feat).view(B, self.n_heads, self.head_dim, N)
        K = self.k_proj(cnn_feat).view(B, self.n_heads, self.head_dim, N)
        V = self.v_proj(cnn_feat).view(B, self.n_heads, self.head_dim, N)

        attn = torch.einsum("bhdn,bhdm->bhnm", Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhdm->bhdn", attn, V)
        out = out.reshape(B, -1, H, W)
        out = self.out_proj(out)

        return out


# ══════════════════════════════════════════════════════════════
# CHANNEL ATTENTION (SE BLOCK)
# ══════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ══════════════════════════════════════════════════════════════
# PI-CCA — MAIN MODEL
# ══════════════════════════════════════════════════════════════

class PIConvCellularAutomaton(nn.Module):
    """Physics-Informed Convolutional Cellular Automaton (PI-CCA).

    Architecture
    ------------
    ┌─────────────┐      ┌───────────────────┐
    │   Input x    │─────→│  Physics Branch    │──→ physics_prob (B,1,H,W)
    │ (B,C,H,W)  │      │  (Rothermel +      │     │
    │              │      │   learnable α)     │     │
    └──────┬───────┘      └───────────────────┘     │
           │                                         ▼
           │         ┌───────────────────┐    ┌──────────────┐
           └────────→│  CNN Branch       │───→│ Cross-Attn   │
                     │  (ResBlocks)      │    │ Fusion       │
                     └───────────────────┘    └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  Channel Attn │
                                              │  + Head       │
                                              └──────┬───────┘
                                                     │
                                              ┌──────▼───────┐
                                              │  Output       │
                                              │ (B,1,H,W)    │
                                              └──────────────┘

    The physics probability acts as a strong prior, while the CNN branch
    learns corrections. Cross-attention enables spatially-adaptive fusion.
    MC-Dropout provides uncertainty estimates at inference time.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        from config import N_INPUT_CHANNELS
        cfg = config or {}
        in_ch = N_INPUT_CHANNELS
        phys_ch = cfg.get("physics_channels", 32)
        cnn_ch = cfg.get("cnn_channels", 64)
        n_heads = cfg.get("attention_heads", 4)
        n_res = cfg.get("n_res_blocks", 3)
        dropout = cfg.get("dropout", 0.15)
        learnable = cfg.get("learnable_physics", True)

        # ── Physics Branch ──
        self.rothermel = DifferentiableRothermel(learnable=learnable)
        self.physics_encoder = nn.Sequential(
            nn.Conv2d(1, phys_ch // 2, 3, padding=1),
            nn.BatchNorm2d(phys_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(phys_ch // 2, phys_ch, 3, padding=1),
            nn.BatchNorm2d(phys_ch),
            nn.ReLU(inplace=True),
        )

        # ── CNN Branch ──
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_ch, cnn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(cnn_ch, dropout) for _ in range(n_res)]
        )

        # ── Cross-Attention Fusion ──
        self.cross_attention = SpatialCrossAttention(phys_ch, cnn_ch, n_heads)

        # ── Channel Attention ──
        self.channel_attn = ChannelAttention(cnn_ch)

        # ── Fusion & Output Head ──
        self.fusion = nn.Sequential(
            nn.Conv2d(cnn_ch + 1, cnn_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(cnn_ch // 2, 1, 1),
            nn.Sigmoid(),
        )

        # Physics weight (learnable balance between physics and data)
        self.physics_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)
        return_components : if True, also return physics and cnn predictions

        Returns
        -------
        (B, 1, H, W) — fused fire probability
        """
        # Physics branch
        p_physics = self.rothermel(x)                    # (B, 1, H, W)
        phys_feat = self.physics_encoder(p_physics)    # (B, phys_ch, H, W)

        # CNN branch
        cnn_feat = self.cnn_encoder(x)                 # (B, cnn_ch, H, W)
        cnn_feat = self.res_blocks(cnn_feat)           # (B, cnn_ch, H, W)

        # Cross-attention fusion
        attended = self.cross_attention(phys_feat, cnn_feat)  # (B, cnn_ch, H, W)
        attended = self.channel_attn(attended)

        # Concatenate physics probability + attended features
        combined = torch.cat([attended, p_physics], dim=1)    # (B, cnn_ch+1, H, W)
        out = self.fusion(combined)                            # (B, 1, H, W)

        # Learnable gated combination
        gate = torch.sigmoid(self.physics_gate)
        final = gate * p_physics + (1 - gate) * out
        final = torch.clamp(final, 0, 1)

        if return_components:
            return final, p_physics, out

        return final

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC-Dropout uncertainty estimation.

        Run n_samples forward passes with dropout enabled and return
        the mean prediction and standard deviation (epistemic uncertainty).
        """
        self.train()  # Enable dropout
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        preds = torch.stack(preds)  # (n_samples, B, 1, H, W)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        self.eval()
        return mean, std

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
