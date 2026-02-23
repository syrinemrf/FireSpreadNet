#!/usr/bin/env python3
"""
src/data/dataset.py — PyTorch Dataset & DataLoader for Fire Propagation
========================================================================
Loads the Next Day Wildfire Spread dataset (Huot et al., 2022) from
pre-processed .npz files and serves (input, target) pairs.

Supports two data layouts:
  1. Split files: train.npz, val.npz, test.npz  (from download_data.py)
  2. Single file: samples.npz + terrain_ids.npy  (legacy / synthetic)

Each sample:
  input  — (C, H, W)  float32 tensor  (12 channels of real satellite data)
  target — (1, H, W)  float32 tensor  (binary fire mask at t+1)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple

from src.data.preprocessing import (
    normalise, augment_sample, FEATURE_ORDER, DEFAULT_STATS
)


class FireSpreadDataset(Dataset):
    """Dataset of (satellite_features, next_day_fire_mask) pairs.

    Data layout on disk (preferred — real data)
    --------------------------------------------
    processed_dir/
      ├── train.npz     # {inputs: (N, C, H, W), targets: (N, 1, H, W)}
      ├── val.npz
      ├── test.npz
      └── metadata.json # {stats, feature_channels, …}

    Data layout on disk (legacy — synthetic)
    -----------------------------------------
    processed_dir/
      ├── samples.npz
      ├── terrain_ids.npy
      └── metadata.json
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        augment: bool = False,
        stats: dict = None,
        seed: int = 42,
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.augment = augment and (split == "train")
        self.rng = np.random.default_rng(seed)

        # Load metadata (stats)
        meta_path = self.processed_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.stats = self.metadata.get("stats", DEFAULT_STATS)
        else:
            self.metadata = {}
            self.stats = stats or DEFAULT_STATS

        # ── Strategy 1: Split files (train.npz, val.npz, test.npz) ──
        split_file = self.processed_dir / f"{split}.npz"
        if split_file.exists():
            data = np.load(split_file)
            self.inputs = data["inputs"]    # (N, C, H, W) float32
            self.targets = data["targets"]  # (N, 1, H, W) float32
            self.indices = np.arange(len(self.inputs))
            return

        # ── Strategy 2: Single file with terrain-based splitting ──
        samples_file = self.processed_dir / "samples.npz"
        if samples_file.exists():
            data = np.load(samples_file)
            self.inputs = data["inputs"]
            self.targets = data["targets"]

            terrain_ids_path = self.processed_dir / "terrain_ids.npy"
            if terrain_ids_path.exists():
                terrain_ids = np.load(terrain_ids_path)
                unique_terrains = np.unique(terrain_ids)
                n_terrains = len(unique_terrains)
                rng = np.random.default_rng(seed)
                perm = rng.permutation(unique_terrains)
                n_test = max(1, int(0.15 * n_terrains))
                n_val = max(1, int(0.15 * n_terrains))
                test_t = set(perm[:n_test])
                val_t = set(perm[n_test:n_test + n_val])
                if split == "test":
                    self.indices = np.where(np.isin(terrain_ids, list(test_t)))[0]
                elif split == "val":
                    self.indices = np.where(np.isin(terrain_ids, list(val_t)))[0]
                else:
                    train_t = set(perm[n_test + n_val:])
                    self.indices = np.where(np.isin(terrain_ids, list(train_t)))[0]
            else:
                n = len(self.inputs)
                idx = np.random.default_rng(seed).permutation(n)
                n_test = int(0.15 * n)
                n_val = int(0.15 * n)
                if split == "test":
                    self.indices = idx[:n_test]
                elif split == "val":
                    self.indices = idx[n_test:n_test + n_val]
                else:
                    self.indices = idx[n_test + n_val:]
            return

        raise FileNotFoundError(
            f"No data found in {self.processed_dir}. "
            f"Run 'python download_data.py' first."
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        x = self.inputs[real_idx].copy()   # (C, H, W)
        y = self.targets[real_idx].copy()  # (1, H, W)

        # Normalise
        x = normalise(x, self.stats)

        # Augment
        if self.augment:
            x, y = augment_sample(x, y, self.rng)

        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42,
) -> dict:
    """Create train / val / test DataLoaders.

    Returns
    -------
    dict with keys 'train', 'val', 'test' → DataLoader
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = FireSpreadDataset(
            processed_dir, split=split,
            augment=(augment_train and split == "train"),
            seed=seed,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
