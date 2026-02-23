#!/usr/bin/env python3
"""
src/models/unet.py — U-Net with Attention Gates for Fire Spread Prediction
===========================================================================
Treats fire propagation as a dense segmentation task: given the current
grid state (C channels), predict a fire probability mask at t+1.

Includes optional spatial attention gates (Oktay et al., 2018) that help
the model focus on the active fire front and its immediate surroundings.

References
----------
  Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional
      Networks for Biomedical Image Segmentation. MICCAI 2015.
  Oktay, O. et al. (2018). Attention U-Net: Learning Where to Look for
      the Pancreas. MIDL 2018.
  Huot, F. et al. (2022). Next Day Wildfire Spread: A Machine Learning
      Dataset for Predicting Wildfire Spreading from Remote-Sensing Data.
      IEEE TGRS, 60, 1–13.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections (Oktay et al., 2018)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNetFire(nn.Module):
    """U-Net with optional attention gates for fire spread prediction.

    Architecture
    ------------
    Encoder: 4 down-sampling blocks (ConvBlock + MaxPool)
    Bottleneck: ConvBlock
    Decoder: 4 up-sampling blocks (TransposeConv + AttentionGate + ConvBlock)
    Head: 1x1 Conv → Sigmoid
    """

    def __init__(self, config: dict = None):
        super().__init__()
        from config import N_INPUT_CHANNELS
        cfg = config or {}
        in_ch = N_INPUT_CHANNELS
        base = cfg.get("base_filters", 32)
        depth = cfg.get("depth", 4)
        dropout = cfg.get("dropout", 0.2)
        use_attn = cfg.get("use_attention", True)

        filters = [base * (2 ** i) for i in range(depth + 1)]
        # e.g. [32, 64, 128, 256, 512]

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_ch
        for i in range(depth):
            self.encoders.append(ConvBlock(ch, filters[i], dropout if i > 0 else 0))
            self.pools.append(nn.MaxPool2d(2))
            ch = filters[i]

        # Bottleneck
        self.bottleneck = ConvBlock(filters[depth - 1], filters[depth], dropout)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(filters[i + 1], filters[i], 2, stride=2))
            if use_attn:
                self.attention_gates.append(AttentionGate(filters[i], filters[i], filters[i] // 2))
            else:
                self.attention_gates.append(nn.Identity())
            self.decoders.append(ConvBlock(filters[i] * 2, filters[i], dropout))

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.use_attn = use_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, 1, H, W) — fire probability
        """
        # Pad to power of 2 if needed
        _, _, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        # Encoder
        skips = []
        h = x
        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for up, attn, dec, skip in zip(
            self.upconvs, self.attention_gates, self.decoders, reversed(skips)
        ):
            h = up(h)
            # Handle size mismatch
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            if self.use_attn and isinstance(attn, AttentionGate):
                skip = attn(g=h, x=skip)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.head(h)

        # Remove padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
