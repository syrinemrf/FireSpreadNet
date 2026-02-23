#!/usr/bin/env python3
"""
explain.py — SHAP & Grad-CAM Explainability for Fire Spread Models
====================================================================
Post-hoc interpretability analysis for trained fire spread models.

Usage
-----
    python explain.py                         # All models
    python explain.py --model pi_cca          # Specific model
    python explain.py --model unet --gradcam  # Grad-CAM only
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MODEL_CONFIG, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
    FIGURES_DIR, SEED, FEATURE_CHANNELS, CH,
)
from src.data.dataset import get_dataloaders
from src.models.convlstm import ConvLSTMModel
from src.models.unet import UNetFire
from src.models.pi_cca import PIConvCellularAutomaton
from src.explainability.shap_analysis import (
    compute_channel_shap, compute_gradcam,
    plot_shap_importance, plot_shap_beeswarm,
    plot_gradcam_overlay,
)

# CA excluded — no learnable parameters
MODEL_CLASSES = {
    "convlstm": ConvLSTMModel,
    "unet": UNetFire,
    "pi_cca": PIConvCellularAutomaton,
}


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "convlstm", "unet", "pi_cca"])
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--n-explain", type=int, default=100)
    parser.add_argument("--gradcam", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(PROCESSED_DIR, batch_size=32, seed=SEED)
    model_names = list(MODEL_CLASSES.keys()) if args.model == "all" else [args.model]

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  Explainability: {MODEL_CONFIG[name]['name']}")
        print(f"{'='*60}")

        cls = MODEL_CLASSES[name]
        cfg = MODEL_CONFIG[name]
        model = cls(config=cfg)

        ckpt = MODELS_DIR / name / "best_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f"Loaded checkpoint: {ckpt}")
        else:
            print(f"No checkpoint found, using initialised model")

        model.eval()
        model.to(device)

        # SHAP
        print("Computing SHAP values...")
        shap_result = compute_channel_shap(
            model, loaders["test"], FEATURE_CHANNELS,
            n_background=args.n_background,
            n_explain=args.n_explain,
            device=str(device),
        )

        # Save
        shap_dir = RESULTS_DIR / "shap" / name
        shap_dir.mkdir(parents=True, exist_ok=True)

        plot_shap_importance(
            shap_result, MODEL_CONFIG[name]["name"],
            save_path=str(shap_dir / "feature_importance.png"),
        )
        plot_shap_beeswarm(
            shap_result, MODEL_CONFIG[name]["name"],
            save_path=str(shap_dir / "beeswarm.png"),
        )

        # Save SHAP values
        np.savez(
            shap_dir / "shap_values.npz",
            shap_values=shap_result["shap_values"],
            mean_abs_shap=shap_result["mean_abs_shap"],
        )

        importance_dict = {
            FEATURE_CHANNELS[i]: float(shap_result["mean_abs_shap"][i])
            for i in range(len(FEATURE_CHANNELS))
        }
        with open(shap_dir / "feature_importance.json", "w") as f:
            json.dump(importance_dict, f, indent=2)

        print(f"  Top features:")
        for idx in shap_result["importance_order"][:5]:
            print(f"    {FEATURE_CHANNELS[idx]:>20s}: {shap_result['mean_abs_shap'][idx]:.4f}")

        # Grad-CAM
        if args.gradcam:
            print("Computing Grad-CAM...")
            x_sample, _ = next(iter(loaders["test"]))
            x_single = x_sample[:1].to(device)

            cam = compute_gradcam(model, x_single, device=str(device))

            fire_mask = x_single[0, CH["prev_fire_mask"]].cpu().numpy()
            plot_gradcam_overlay(
                cam, fire_mask, MODEL_CONFIG[name]["name"],
                save_path=str(shap_dir / "gradcam.png"),
            )
            print(f"  Saved Grad-CAM to {shap_dir / 'gradcam.png'}")

    print("\nExplainability analysis complete!")


if __name__ == "__main__":
    main()
