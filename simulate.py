#!/usr/bin/env python3
"""
simulate.py — Visualise Fire Spread Predictions on Real Data
==============================================================
Load real test samples from Next Day Wildfire Spread (Huot et al., 2022)
and visualise model predictions vs. ground truth next-day fire masks.

Usage
-----
    python simulate.py --model pi_cca --n-samples 5
    python simulate.py --compare-all --n-samples 3 --gif
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_CONFIG, MODELS_DIR, FIGURES_DIR, PROCESSED_DIR,
    SEED, GRID_SIZE, FEATURE_CHANNELS, N_INPUT_CHANNELS, CH,
)
from src.data.dataset import FireSpreadDataset
from src.data.preprocessing import normalise, DEFAULT_STATS
from src.models.cellular_automata import CellularAutomataModel
from src.models.convlstm import ConvLSTMModel
from src.models.unet import UNetFire
from src.models.pi_cca import PIConvCellularAutomaton
from src.visualization.fire_visualizer import (
    plot_prediction_comparison, plot_uncertainty_map,
)

import matplotlib.pyplot as plt

MODEL_CLASSES = {
    "ca": CellularAutomataModel,
    "convlstm": ConvLSTMModel,
    "unet": UNetFire,
    "pi_cca": PIConvCellularAutomaton,
}


def load_model(name: str, device: torch.device):
    cls = MODEL_CLASSES[name]
    cfg = MODEL_CONFIG[name]
    model = cls(config=cfg)
    ckpt = MODELS_DIR / name / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {name} from {ckpt}")
    else:
        print(f"No checkpoint for {name}, using initialised weights")
    model.eval()
    model.to(device)
    return model


def visualise_predictions(model, model_name, samples, device, save_dir):
    """Visualise model predictions on real test samples."""
    for i, (x, y) in enumerate(samples):
        x_dev = x.unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad():
            pred = model(x_dev).squeeze().cpu().numpy()  # (H, W)

        # Ground truth
        gt = y.squeeze().numpy()

        # Input fire mask
        fire_in = x[CH["prev_fire_mask"]].numpy()

        # Elevation for background
        elev = x[CH["elevation"]].numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(elev, cmap="terrain")
        axes[0].set_title("Elevation (SRTM)", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(fire_in, cmap="hot", vmin=0, vmax=1)
        axes[1].set_title("Input: Day-t Fire Mask", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
        axes[2].contour(gt, levels=[0.5], colors="lime", linewidths=1.5)
        axes[2].set_title(f"Prediction ({model_name})", fontsize=11)
        axes[2].axis("off")

        axes[3].imshow(gt, cmap="hot", vmin=0, vmax=1)
        axes[3].set_title("Ground Truth: Day-t+1", fontsize=11)
        axes[3].axis("off")

        fig.suptitle(
            f"Sample {i+1} — {MODEL_CONFIG[model_name]['name']}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            save_dir / f"predict_{model_name}_sample{i+1}.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Saved sample {i+1}")


def main():
    parser = argparse.ArgumentParser(description="Visualise fire spread predictions")
    parser.add_argument("--model", type=str, default="pi_cca",
                        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--data-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load test samples
    test_ds = FireSpreadDataset(
        args.data_dir, split="test", augment=False, seed=args.seed
    )
    rng = np.random.default_rng(args.seed)
    sample_idxs = rng.choice(len(test_ds), size=min(args.n_samples, len(test_ds)),
                              replace=False)
    samples = [test_ds[i] for i in sample_idxs]
    print(f"Loaded {len(samples)} test samples from {args.data_dir}")

    if args.compare_all:
        # Compare all models on same samples
        all_preds = {}
        for name in MODEL_CLASSES:
            model = load_model(name, device)
            preds = []
            for x, y in samples:
                with torch.no_grad():
                    p = model(x.unsqueeze(0).to(device)).squeeze().cpu().numpy()
                preds.append(p)
            all_preds[name] = preds

        # Plot comparison grid
        for i, (x, y) in enumerate(samples):
            gt = y.squeeze().numpy()
            elev = x[CH["elevation"]].numpy()
            fire_in = x[CH["prev_fire_mask"]].numpy()

            n_models = len(MODEL_CLASSES)
            fig, axes = plt.subplots(1, n_models + 2, figsize=(5 * (n_models + 2), 5))

            axes[0].imshow(fire_in, cmap="hot", vmin=0, vmax=1)
            axes[0].set_title("Input Fire (Day t)", fontsize=10)
            axes[0].axis("off")

            for j, (name, preds) in enumerate(all_preds.items()):
                axes[j + 1].imshow(preds[i], cmap="hot", vmin=0, vmax=1)
                axes[j + 1].contour(gt, levels=[0.5], colors="lime", linewidths=1)
                axes[j + 1].set_title(MODEL_CONFIG[name]["name"], fontsize=10)
                axes[j + 1].axis("off")

            axes[-1].imshow(gt, cmap="hot", vmin=0, vmax=1)
            axes[-1].set_title("Ground Truth (Day t+1)", fontsize=10)
            axes[-1].axis("off")

            fig.suptitle(f"Model Comparison — Sample {i+1}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"compare_all_sample{i+1}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved comparison for sample {i+1}")

        # Uncertainty for PI-CCA
        pi_cca = load_model("pi_cca", device)
        if hasattr(pi_cca, "predict_with_uncertainty"):
            print("\nComputing PI-CCA uncertainty maps...")
            for i, (x, y) in enumerate(samples[:3]):
                mean_p, std_p = pi_cca.predict_with_uncertainty(
                    x.unsqueeze(0).to(device), n_samples=30
                )
                fig = plot_uncertainty_map(
                    mean_p.squeeze().cpu().numpy(),
                    std_p.squeeze().cpu().numpy(),
                    y.squeeze().numpy(),
                    save_path=str(FIGURES_DIR / f"uncertainty_sample{i+1}.png"),
                )
                plt.close(fig)
                print(f"  Saved uncertainty map for sample {i+1}")
    else:
        model = load_model(args.model, device)
        visualise_predictions(model, args.model, samples, device, FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
