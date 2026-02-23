#!/usr/bin/env python3
"""
download_data.py — Download & Convert the Next Day Wildfire Spread Dataset
============================================================================
Downloads the real satellite dataset from Kaggle (Huot et al., 2022)
and converts TFRecord files to NumPy .npz for PyTorch training.

Dataset: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread

The dataset contains:
  - 12 input features per 64×64 pixel patch (real satellite / reanalysis data)
  - 1 target: next-day fire mask (binary)

Two conversion backends are supported:
  1. `tensorflow` (recommended, most robust)
  2. `tfrecord`   (lightweight, no TF dependency)

Usage
-----
    # Option 1: Download from Kaggle + convert
    python download_data.py

    # Option 2: Convert already-downloaded TFRecords
    python download_data.py --skip-download --raw-dir data/raw

    # Option 3: Limit samples (for quick testing)
    python download_data.py --max-samples 5000
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, PROCESSED_DIR, SEED, GRID_SIZE,
    TFRECORD_INPUT_KEYS, TFRECORD_TARGET_KEY,
    FEATURE_CHANNELS, DATASET_CONFIG,
)


# ── TFRecord feature names ────────────────────────────────────
INPUT_FEATURES = TFRECORD_INPUT_KEYS
OUTPUT_FEATURES = [TFRECORD_TARGET_KEY]
ALL_FEATURES = INPUT_FEATURES + OUTPUT_FEATURES
N_PIXELS = GRID_SIZE * GRID_SIZE  # 64 * 64 = 4096


# ══════════════════════════════════════════════════════════════
# DOWNLOAD FROM KAGGLE
# ══════════════════════════════════════════════════════════════

def download_from_kaggle(output_dir: Path):
    """Download the dataset using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("  pip install kaggle")
        print("  Then set up your API key: https://www.kaggle.com/docs/api")
        print("\nAlternative: download manually from:")
        print("  https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread")
        print(f"  Extract to: {output_dir}")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    dataset_slug = DATASET_CONFIG["kaggle_dataset"]
    print(f"Downloading {dataset_slug} ...")
    api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
    print(f"Downloaded to {output_dir}")


# ══════════════════════════════════════════════════════════════
# PARSE TFRECORDS — TENSORFLOW BACKEND
# ══════════════════════════════════════════════════════════════

def parse_with_tensorflow(tfrecord_files: list, max_samples: int = None):
    """Parse TFRecord files using TensorFlow."""
    import tensorflow as tf

    feature_description = {
        feat: tf.io.FixedLenFeature([N_PIXELS], tf.float32)
        for feat in ALL_FEATURES
    }

    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    inputs_list, targets_list = [], []
    count = 0

    for fpath in tqdm(tfrecord_files, desc="Parsing TFRecords (tf)"):
        ds = tf.data.TFRecordDataset(str(fpath))
        for raw in ds:
            parsed = _parse(raw)
            # Stack input channels: (C, H, W)
            channels = []
            for feat in INPUT_FEATURES:
                arr = parsed[feat].numpy().reshape(GRID_SIZE, GRID_SIZE)
                channels.append(arr)
            x = np.stack(channels, axis=0).astype(np.float32)

            # Target: (1, H, W)
            y = parsed[TFRECORD_TARGET_KEY].numpy().reshape(
                1, GRID_SIZE, GRID_SIZE
            ).astype(np.float32)

            # Skip samples with no fire at all (no signal)
            if x[-1].max() == 0 and y.max() == 0:
                continue

            inputs_list.append(x)
            targets_list.append(y)
            count += 1

            if max_samples and count >= max_samples:
                return np.stack(inputs_list), np.stack(targets_list)

    if not inputs_list:
        return np.empty((0,)), np.empty((0,))
    return np.stack(inputs_list), np.stack(targets_list)


# ══════════════════════════════════════════════════════════════
# PARSE TFRECORDS — TFRECORD PACKAGE (lightweight, no TF)
# ══════════════════════════════════════════════════════════════

def parse_with_tfrecord(tfrecord_files: list, max_samples: int = None):
    """Parse TFRecord files using the lightweight `tfrecord` package."""
    import tfrecord

    description = {feat: "float" for feat in ALL_FEATURES}
    inputs_list, targets_list = [], []
    count = 0

    for fpath in tqdm(tfrecord_files, desc="Parsing TFRecords (tfrecord)"):
        loader = tfrecord.tfrecord_loader(
            str(fpath), index_path=None, description=description
        )
        for record in loader:
            channels = []
            for feat in INPUT_FEATURES:
                arr = np.array(record[feat], dtype=np.float32).reshape(
                    GRID_SIZE, GRID_SIZE
                )
                channels.append(arr)
            x = np.stack(channels, axis=0)

            y = np.array(record[TFRECORD_TARGET_KEY], dtype=np.float32).reshape(
                1, GRID_SIZE, GRID_SIZE
            )

            if x[-1].max() == 0 and y.max() == 0:
                continue

            inputs_list.append(x)
            targets_list.append(y)
            count += 1

            if max_samples and count >= max_samples:
                return np.stack(inputs_list), np.stack(targets_list)

    if not inputs_list:
        return np.empty((0,)), np.empty((0,))
    return np.stack(inputs_list), np.stack(targets_list)


# ══════════════════════════════════════════════════════════════
# FIND TFRECORD FILES
# ══════════════════════════════════════════════════════════════

def find_tfrecord_files(raw_dir: Path, split: str):
    """Find TFRecord shard files for a given split."""
    patterns = [
        f"next_day_wildfire_spread_{split}.tfrecord*",
        f"*_{split}.tfrecord*",
        f"{split}*.tfrecord*",
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(raw_dir.rglob(pattern)))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen and f.suffix != ".index":
            seen.add(f)
            unique.append(f)
    return unique


# ══════════════════════════════════════════════════════════════
# COMPUTE NORMALISATION STATISTICS
# ══════════════════════════════════════════════════════════════

def compute_statistics(inputs: np.ndarray) -> dict:
    """Compute per-channel mean / std from (N, C, H, W) array."""
    stats = {}
    for i, name in enumerate(FEATURE_CHANNELS):
        ch = inputs[:, i]
        stats[name] = {
            "mean": float(np.nanmean(ch)),
            "std": float(np.nanstd(ch)) + 1e-8,
        }
    return stats


# ══════════════════════════════════════════════════════════════
# MAIN CONVERSION PIPELINE
# ══════════════════════════════════════════════════════════════

def convert_dataset(raw_dir: Path, output_dir: Path, max_samples: int = None):
    """Convert all splits (train / val / test) to .npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose parsing backend
    parse_fn = None
    try:
        import tensorflow  # noqa: F401
        parse_fn = parse_with_tensorflow
        print("Using TensorFlow backend for TFRecord parsing.")
    except ImportError:
        try:
            import tfrecord  # noqa: F401
            parse_fn = parse_with_tfrecord
            print("Using tfrecord package for TFRecord parsing.")
        except ImportError:
            print("ERROR: Neither 'tensorflow' nor 'tfrecord' is installed.")
            print("Install one of them:")
            print("  pip install tensorflow     # full TF (recommended)")
            print("  pip install tfrecord        # lightweight alternative")
            sys.exit(1)

    split_map = {"train": "train", "val": "eval", "test": "test"}
    all_stats = None

    for our_split, file_split in split_map.items():
        print(f"\n{'='*60}")
        print(f"  Processing split: {our_split} (files: *_{file_split}*)")
        print(f"{'='*60}")

        tfrecord_files = find_tfrecord_files(raw_dir, file_split)
        if not tfrecord_files:
            print(f"  WARNING: No TFRecord files found for split '{file_split}'")
            print(f"  Searched in: {raw_dir}")
            continue

        print(f"  Found {len(tfrecord_files)} shard file(s)")

        max_per_split = max_samples if max_samples else None
        inputs, targets = parse_fn(tfrecord_files, max_per_split)

        if len(inputs) == 0:
            print(f"  WARNING: No valid samples found for {our_split}")
            continue

        # Replace NaN with 0 (some GRIDMET features have NaN at edges)
        inputs = np.nan_to_num(inputs, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)

        # Clamp target to binary {0, 1}
        targets = (targets > 0).astype(np.float32)

        print(f"  Samples: {len(inputs)}")
        print(f"  Input shape:  {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Fire prevalence: {targets.mean():.4f}")

        # Save .npz
        out_path = output_dir / f"{our_split}.npz"
        np.savez_compressed(out_path, inputs=inputs, targets=targets)
        print(f"  Saved: {out_path}")

        # Compute stats on training set only
        if our_split == "train":
            all_stats = compute_statistics(inputs)
            print(f"  Computed normalisation statistics from training set.")

    # Save metadata
    if all_stats:
        metadata = {
            "dataset": "Next Day Wildfire Spread (Huot et al., 2022)",
            "source": "MODIS/VIIRS, GRIDMET, SRTM, LandScan",
            "grid_size": GRID_SIZE,
            "n_input_channels": N_INPUT_CHANNELS,
            "feature_channels": FEATURE_CHANNELS,
            "tfrecord_keys": INPUT_FEATURES,
            "stats": all_stats,
        }
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved: {meta_path}")

    print("\n✅ Dataset conversion complete!")
    print(f"   Output directory: {output_dir}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Download & convert Next Day Wildfire Spread dataset"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Kaggle download (files already in --raw-dir)"
    )
    parser.add_argument(
        "--raw-dir", type=str, default=str(RAW_DIR),
        help="Directory containing raw TFRecord files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(PROCESSED_DIR),
        help="Output directory for processed .npz files"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum samples per split (for quick testing)"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not args.skip_download:
        download_from_kaggle(raw_dir)

    convert_dataset(raw_dir, output_dir, args.max_samples)


if __name__ == "__main__":
    main()
