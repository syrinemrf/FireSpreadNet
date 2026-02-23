#!/usr/bin/env python3
"""
src/data/preprocessing.py — Data Preprocessing for Fire Propagation
====================================================================
Normalisation, augmentation, and feature engineering for the
Next Day Wildfire Spread dataset (Huot et al., 2022).

Real satellite features (12 channels):
  elevation, wind_speed, wind_direction, min_temp, max_temp,
  humidity, precipitation, drought_index, ndvi, erc,
  population, prev_fire_mask
"""

import numpy as np
import torch
from typing import Tuple, Dict

from config import FEATURE_CHANNELS, N_INPUT_CHANNELS


# ── Feature order (matches config.FEATURE_CHANNELS) ─────────
FEATURE_ORDER = FEATURE_CHANNELS

# ── Default normalisation statistics ─────────────────────────
# Approximate values from the Next Day Wildfire Spread dataset.
# These are overwritten by compute_statistics() on the actual training set.
DEFAULT_STATS = {
    "elevation":      {"mean": 1200.0, "std": 800.0},
    "wind_speed":     {"mean": 3.5,    "std": 1.5},
    "wind_direction": {"mean": 200.0,  "std": 80.0},
    "min_temp":       {"mean": 285.0,  "std": 8.0},
    "max_temp":       {"mean": 305.0,  "std": 8.0},
    "humidity":       {"mean": 0.005,  "std": 0.003},
    "precipitation":  {"mean": 1.0,    "std": 5.0},
    "drought_index":  {"mean": 0.0,    "std": 3.0},
    "ndvi":           {"mean": 0.3,    "std": 0.2},
    "erc":            {"mean": 40.0,   "std": 25.0},
    "population":     {"mean": 50.0,   "std": 200.0},
    "prev_fire_mask": {"mean": 0.0,    "std": 1.0},  # binary — no norm
}

# Channels that should NOT be normalised (binary masks)
_NO_NORM = {"prev_fire_mask"}


def compute_statistics(samples) -> Dict[str, Dict[str, float]]:
    """Compute per-channel mean / std from input tensors.

    Parameters
    ----------
    samples : list of np.ndarray (C, H, W) OR single (N, C, H, W) array.

    Returns
    -------
    dict  channel_name → {mean, std}
    """
    if isinstance(samples, np.ndarray) and samples.ndim == 4:
        stacked = samples
    else:
        stacked = np.stack(samples, axis=0)  # (N, C, H, W)
    stats = {}
    for i, name in enumerate(FEATURE_ORDER):
        ch = stacked[:, i]
        stats[name] = {
            "mean": float(np.nanmean(ch)),
            "std": float(np.nanstd(ch)) + 1e-8,
        }
    return stats


def normalise(tensor: np.ndarray, stats: dict = None) -> np.ndarray:
    """Z-score normalise a (C, H, W) array channel-wise.

    prev_fire_mask is NOT normalised (binary).
    """
    stats = stats or DEFAULT_STATS
    out = tensor.copy().astype(np.float32)
    for i, name in enumerate(FEATURE_ORDER):
        if name in _NO_NORM:
            continue
        if name in stats:
            out[i] = (out[i] - stats[name]["mean"]) / stats[name]["std"]
    return out


def denormalise(tensor: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of normalise()."""
    out = tensor.copy().astype(np.float32)
    for i, name in enumerate(FEATURE_ORDER):
        if name in _NO_NORM:
            continue
        if name in stats:
            out[i] = out[i] * stats[name]["std"] + stats[name]["mean"]
    return out


# ── Augmentation ───────────────────────────────────────────────
def augment_sample(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random spatial augmentations to an (input, target) pair.

    Both x (C, H, W) and y (1, H, W) are transformed identically.
    Augmentations: random 90° rotation, horizontal/vertical flip.
    """
    rng = rng or np.random.default_rng()

    # Random 90° rotations
    k = rng.integers(0, 4)
    if k > 0:
        x = np.rot90(x, k, axes=(1, 2)).copy()
        y = np.rot90(y, k, axes=(1, 2)).copy()

    # Horizontal flip
    if rng.random() > 0.5:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()

    # Vertical flip
    if rng.random() > 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()

    return x, y
