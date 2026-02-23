#!/usr/bin/env python3
"""
src/models/convlstm.py — Convolutional LSTM for Fire Spread Prediction
=======================================================================
Spatio-temporal deep learning baseline that treats fire propagation as
a sequence-to-one video prediction task.

The model processes T consecutive fire-state grids and predicts the
fire mask at the next timestep.

References
----------
  Shi, X. et al. (2015). Convolutional LSTM Network: A Machine Learning
      Approach for Precipitation Nowcasting. NeurIPS 2015.
  Radke, D. et al. (2019). FireCast: Leveraging Deep Learning to Predict
      Wildfire Spread. IJCAI 2019.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ConvLSTMCell(nn.Module):
    """Single Convolutional LSTM cell."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=pad, bias=True
        )

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C_in, H, W)
        state : (h, c) each (B, C_hid, H, W) or None

        Returns
        -------
        h, c : (B, C_hid, H, W)
        """
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMModel(nn.Module):
    """Multi-layer ConvLSTM for spatial fire spread prediction.

    Architecture
    ------------
    Input (B, C, H, W) → ConvLSTM stack → 1x1 Conv → Sigmoid → (B, 1, H, W)

    For single-step prediction, the "sequence" is length 1.
    The model can also be unrolled autoregressively for multi-step.
    """

    def __init__(self, config: dict = None):
        super().__init__()
        from config import N_INPUT_CHANNELS
        cfg = config or {}
        in_ch = N_INPUT_CHANNELS
        hidden_list = cfg.get("hidden_channels", [32, 64, 32])
        ks = cfg.get("kernel_size", 3)
        dropout = cfg.get("dropout", 0.2)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_list[0], 3, padding=1),
            nn.BatchNorm2d(hidden_list[0]),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM layers
        self.lstm_cells = nn.ModuleList()
        ch_in = hidden_list[0]
        for ch_out in hidden_list:
            self.lstm_cells.append(ConvLSTMCell(ch_in, ch_out, ks))
            ch_in = ch_out

        self.dropout = nn.Dropout2d(dropout)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_list[-1], 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W) — single timestep input

        Returns
        -------
        (B, 1, H, W) — fire probability at next timestep
        """
        h = self.encoder(x)

        states = [None] * len(self.lstm_cells)
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(h, states[i])
            states[i] = (h, c)
            h = self.dropout(h)

        out = self.decoder(h)
        return out

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
