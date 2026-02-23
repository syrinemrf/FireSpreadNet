#!/usr/bin/env python3
"""
train.py — Model Training for Fire Spread Prediction
=====================================================
Train and compare all four models on real satellite data
(Next Day Wildfire Spread — Huot et al., 2022).

Models:
  1. Cellular Automata (physics baseline, no training)
  2. ConvLSTM (DL baseline)
  3. U-Net (DL baseline)
  4. PI-CCA (hybrid, novel contribution)

Usage
-----
    python train.py                        # Train all models
    python train.py --model pi_cca         # Train specific model
    python train.py --evaluate             # Evaluate only (needs checkpoints)
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_CONFIG, TRAIN_CONFIG, PROCESSED_DIR, MODELS_DIR,
    RESULTS_DIR, FIGURES_DIR, SEED, N_INPUT_CHANNELS
)
from src.data.dataset import get_dataloaders
from src.models.cellular_automata import CellularAutomataModel
from src.models.convlstm import ConvLSTMModel
from src.models.unet import UNetFire
from src.models.pi_cca import PIConvCellularAutomaton
from src.training.trainer import Trainer
from src.visualization.fire_visualizer import plot_training_curves


# ── Reproducibility ───
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


MODEL_CLASSES = {
    "ca": (CellularAutomataModel, MODEL_CONFIG["ca"]),
    "convlstm": (ConvLSTMModel, MODEL_CONFIG["convlstm"]),
    "unet": (UNetFire, MODEL_CONFIG["unet"]),
    "pi_cca": (PIConvCellularAutomaton, MODEL_CONFIG["pi_cca"]),
}


def build_model(name: str):
    cls, cfg = MODEL_CLASSES[name]
    model = cls(config=cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {cfg.get('name', name)} — {n_params:,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train fire spread models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "ca", "convlstm", "unet", "pi_cca"])
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument("--data-dir", type=str, default=str(PROCESSED_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: Next Day Wildfire Spread (Huot et al., 2022)")
    print(f"Data directory: {args.data_dir}")

    # Data
    loaders = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        augment_train=True,
        seed=SEED,
    )
    print(f"Train: {len(loaders['train'].dataset)} | "
          f"Val: {len(loaders['val'].dataset)} | "
          f"Test: {len(loaders['test'].dataset)}")

    # Select models
    model_names = list(MODEL_CLASSES.keys()) if args.model == "all" else [args.model]

    all_histories = {}
    all_results = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  Model: {MODEL_CONFIG[name]['name']}")
        print(f"{'='*60}")

        model = build_model(name)
        cfg = TRAIN_CONFIG.copy()
        cfg["epochs"] = args.epochs

        trainer = Trainer(model, name, device=device, config=cfg)

        if not args.evaluate:
            # Train
            history = trainer.train(loaders["train"], loaders["val"])
            all_histories[name] = history
        else:
            # Load checkpoint
            ckpt = MODELS_DIR / name / "best_model.pt"
            if ckpt.exists():
                model.load_state_dict(torch.load(ckpt, map_location=device))
                print(f"  Loaded checkpoint: {ckpt}")
            else:
                print(f"  WARNING: No checkpoint found at {ckpt}")
                continue

        # Evaluate on test set
        test_metrics = trainer.evaluate(loaders["test"])
        all_results[name] = test_metrics

        print(f"\n  Test Results for {name}:")
        for k, v in test_metrics.items():
            print(f"    {k:>15s}: {v:.4f}")

        # Save individual results
        save_dir = MODELS_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "test_results.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    # Save comparison
    with open(RESULTS_DIR / "model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plot training curves
    if all_histories:
        plot_training_curves(all_histories, save_path=str(FIGURES_DIR / "training_curves.png"))

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':>20s} | {'IoU':>8s} | {'Dice':>8s} | {'F1':>8s} | {'Precision':>9s} | {'Recall':>8s}")
    print(f"{'-'*80}")
    for name, metrics in all_results.items():
        print(f"{MODEL_CONFIG[name]['name']:>20s} | "
              f"{metrics.get('iou', 0):.4f}   | "
              f"{metrics.get('dice', 0):.4f}   | "
              f"{metrics.get('f1', 0):.4f}   | "
              f"{metrics.get('precision', 0):.4f}    | "
              f"{metrics.get('recall', 0):.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
