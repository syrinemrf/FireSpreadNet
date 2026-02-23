#!/usr/bin/env python3
"""
src/models/cellular_automata.py — Stochastic Cellular Automata Baseline
========================================================================
Physics-based fire spread model using transition rules derived from
Rothermel equations and the Alexandridis et al. (2008) CA framework.

Adapted for real satellite data (Next Day Wildfire Spread, Huot et al. 2022):
  - Elevation (SRTM) → slope computed via Sobel gradient
  - Wind speed & direction (GRIDMET)
  - Humidity (GRIDMET sph) → moisture damping
  - NDVI (VIIRS) → fuel / vegetation proxy
  - PrevFireMask → fire state source

This model has NO learnable parameters — it serves as the physics baseline.

References
----------
  Alexandridis, A., Vakalis, D., Siettos, C.I. & Bafas, G.V. (2008).
      A cellular automata model for forest fire spread prediction.
      Applied Mathematics and Computation, 204(1), 191–201.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CH  # Channel index mapping


class CellularAutomataModel(nn.Module):
    """Deterministic / stochastic CA fire spread (no learnable params).

    Uses real satellite features:
      - elevation → slope via Sobel gradient
      - wind speed & direction
      - humidity/NDVI as fire-condition proxies
      - prev_fire_mask as fire source
    """

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        self.p_burn = cfg.get("p_burn_base", 0.58)
        self.w_wind = cfg.get("wind_weight", 0.045)
        self.w_slope = cfg.get("slope_weight", 0.078)
        # Dummy parameter so .parameters() is not empty (required by optimiser)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Sobel kernels for slope estimation from elevation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _compute_slope(self, elevation: torch.Tensor) -> torch.Tensor:
        """Compute slope (degrees) from elevation using Sobel gradients.

        Parameters
        ----------
        elevation : (B, 1, H, W)

        Returns
        -------
        slope : (B, 1, H, W) in degrees
        """
        dx = F.conv2d(elevation, self.sobel_x, padding=1)
        dy = F.conv2d(elevation, self.sobel_y, padding=1)
        gradient_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
        slope_deg = torch.atan(gradient_mag) * 180.0 / np.pi
        return slope_deg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next fire state from input tensor.

        Parameters
        ----------
        x : (B, C, H, W) — channels follow FEATURE_CHANNELS from config

        Returns
        -------
        (B, 1, H, W) — probability of fire at t+1
        """
        B, C, H, W = x.shape
        device = x.device

        # Extract channels using CH index mapping
        elevation  = x[:, CH["elevation"]:CH["elevation"]+1]     # (B,1,H,W)
        wind_speed = x[:, CH["wind_speed"]]                      # (B,H,W)
        wind_dir   = x[:, CH["wind_direction"]]                  # (B,H,W)
        humidity   = x[:, CH["humidity"]]                        # (B,H,W)
        ndvi       = x[:, CH["ndvi"]]                            # (B,H,W)
        fire_state = x[:, CH["prev_fire_mask"]]                  # (B,H,W)

        # Compute slope from elevation
        slope = self._compute_slope(elevation).squeeze(1)        # (B,H,W)

        output = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            fs = fire_state[b]       # (H, W)
            burning = (fs > 0.5)

            # Moore neighbourhood offsets
            offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)]
            dir_angles = [90, 45, 0, 315, 270, 225, 180, 135]

            prob = torch.zeros(H, W, device=device)

            for (dr, dc), d_angle in zip(offsets, dir_angles):
                # Shift burning mask
                shifted = torch.zeros_like(burning)
                sr = slice(max(0, -dr), H + min(0, -dr))
                dr2 = slice(max(0, dr), H + min(0, dr))
                sc = slice(max(0, -dc), W + min(0, -dc))
                dc2 = slice(max(0, dc), W + min(0, dc))
                shifted[sr, sc] = burning[dr2, dc2]

                if not shifted.any():
                    continue

                # Wind effect
                ws = wind_speed[b]
                wd = wind_dir[b]
                wind_math = (270.0 - wd) % 360.0
                angle_diff = (d_angle - wind_math) * np.pi / 180.0
                wind_factor = 1.0 + self.w_wind * torch.cos(angle_diff) * ws

                # Slope effect
                sl = slope[b]
                slope_effect = 1.0 + self.w_slope * torch.tan(
                    sl * np.pi / 180.0
                )

                # Humidity damping (drier → easier spread)
                hum = humidity[b]
                moisture_damp = torch.clamp(1.0 - 2.0 * hum * 200.0, 0.3, 1.0)

                # Vegetation factor (higher NDVI → more fuel)
                veg_factor = 0.5 + 0.5 * torch.clamp(ndvi[b], 0, 1)

                # Combine
                p_spread = self.p_burn * wind_factor * slope_effect * \
                    moisture_damp * veg_factor
                p_spread = torch.clamp(p_spread, 0, 1)

                # Accumulate (max over directions)
                contrib = shifted.float() * p_spread
                prob = torch.max(prob, contrib)

            # Only unburned cells can ignite
            unburned = (fs < 0.5)
            prob = prob * unburned.float()

            # Existing fire persists
            result = torch.max(fs.unsqueeze(0), prob.unsqueeze(0))
            output[b, 0] = result.squeeze(0)

        return output

    @property
    def n_parameters(self) -> int:
        return 0  # Pure physics model
