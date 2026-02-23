#!/usr/bin/env python3
"""
src/visualization/fire_visualizer.py — Fire Propagation Visualization
======================================================================
Produces animated fire spread visualisations:
  • Static snapshots at selected hours
  • Animated GIF of full propagation sequence
  • Side-by-side comparison of ground truth vs model prediction
  • Uncertainty maps for PI-CCA
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import Optional, List
import imageio


# ── Custom colormaps ───────────────────────────────────────────
FIRE_CMAP = ListedColormap([
    "#2d5016",   # unburned — dark green
    "#FF4500",   # burning — orange-red
    "#1a1a1a",   # burned out — dark grey
])

PROB_CMAP = "YlOrRd"  # yellow → orange → red for probabilities


def plot_fire_state(
    state: np.ndarray,
    terrain: Optional[np.ndarray] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> plt.Figure:
    """Plot a single fire state grid.

    Parameters
    ----------
    state : (H, W) int array — 0=unburned, 1=burning, 2=burned
    terrain : optional (H, W) DEM for hillshade background
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Hillshade background from DEM
    if terrain is not None:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(terrain, vert_exag=2)
        ax.imshow(hillshade, cmap="gray", alpha=0.4)

    # Fire overlay
    im = ax.imshow(state, cmap=FIRE_CMAP, vmin=0, vmax=2, alpha=0.8, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column (cell)")
    ax.set_ylabel("Row (cell)")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
        cbar.ax.set_yticklabels(["Unburned", "Burning", "Burned"])

    return fig


def plot_propagation_snapshots(
    fire_states: np.ndarray,
    terrain: Optional[np.ndarray] = None,
    hours: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "",
) -> plt.Figure:
    """Plot fire state at selected timesteps in a grid layout.

    Parameters
    ----------
    fire_states : (T, H, W) int array
    hours : list of timestep indices to show (default: evenly spaced)
    """
    T = len(fire_states)
    if hours is None:
        hours = list(range(0, T, max(1, T // 6)))[:6]
    n = len(hours)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, h in enumerate(hours):
        if h < T:
            plot_fire_state(fire_states[h], terrain, f"{title_prefix}t={h}h", axes[i], show_colorbar=False)
        else:
            axes[i].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{title_prefix}Fire Propagation Snapshots", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_propagation_gif(
    fire_states: np.ndarray,
    save_path: str,
    terrain: Optional[np.ndarray] = None,
    fps: int = 2,
    title: str = "Fire Propagation",
) -> str:
    """Create an animated GIF of fire propagation.

    Parameters
    ----------
    fire_states : (T, H, W) int array
    save_path : path for output .gif
    fps : frames per second
    """
    frames = []
    T = len(fire_states)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(6, 6))
        if terrain is not None:
            from matplotlib.colors import LightSource
            ls = LightSource(azdeg=315, altdeg=45)
            ax.imshow(ls.hillshade(terrain, vert_exag=2), cmap="gray", alpha=0.4)

        ax.imshow(fire_states[t], cmap=FIRE_CMAP, vmin=0, vmax=2, alpha=0.8, interpolation="nearest")
        burned = np.sum((fire_states[t] == 1) | (fire_states[t] == 2))
        total = fire_states[t].size
        ax.set_title(f"{title} — t={t}h | Burned: {burned}/{total} ({100*burned/total:.1f}%)",
                      fontsize=11, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return save_path


def plot_prediction_comparison(
    ground_truth: np.ndarray,
    predictions: dict,
    terrain: Optional[np.ndarray] = None,
    timestep: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of ground truth vs model predictions.

    Parameters
    ----------
    ground_truth : (H, W) int or float array
    predictions : dict {model_name: (H, W) float array}
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))

    # Ground truth
    axes[0].imshow(ground_truth, cmap=FIRE_CMAP, vmin=0, vmax=2, interpolation="nearest")
    axes[0].set_title("Ground Truth", fontsize=12, fontweight="bold")

    for i, (name, pred) in enumerate(predictions.items()):
        im = axes[i + 1].imshow(pred, cmap=PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
        axes[i + 1].set_title(f"{name}", fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=axes[i + 1], shrink=0.8, label="P(fire)")

    fig.suptitle(f"Model Comparison — t={timestep}h", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_uncertainty_map(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot PI-CCA uncertainty map (MC-Dropout).

    Parameters
    ----------
    mean_pred : (H, W) — mean prediction
    std_pred  : (H, W) — standard deviation (uncertainty)
    """
    n_plots = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    im0 = axes[0].imshow(mean_pred, cmap=PROB_CMAP, vmin=0, vmax=1)
    axes[0].set_title("Mean Prediction", fontsize=12, fontweight="bold")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(std_pred, cmap="viridis", vmin=0)
    axes[1].set_title("Epistemic Uncertainty (σ)", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap=FIRE_CMAP, vmin=0, vmax=2)
        axes[2].set_title("Ground Truth", fontsize=12, fontweight="bold")

    fig.suptitle("PI-CCA — Prediction with Uncertainty", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    histories: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training & validation loss curves for all models.

    Parameters
    ----------
    histories : dict {model_name: {"train_loss": [...], "val_loss": [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, (name, hist) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], color=c, label=f"{name} (train)")
        axes[0].plot(epochs, hist["val_loss"], color=c, linestyle="--", label=f"{name} (val)")

        if hist.get("val_metrics"):
            iou_vals = [m["iou"] if isinstance(m, dict) else 0 for m in hist["val_metrics"]]
            axes[1].plot(epochs, iou_vals, color=c, label=name)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_burned_area_evolution(
    burned_areas: List[float],
    save_path: Optional[str] = None,
    title: str = "Burned Area Over Time",
) -> plt.Figure:
    """Plot burned area (km²) over hours."""
    fig, ax = plt.subplots(figsize=(10, 5))
    hours = range(1, len(burned_areas) + 1)
    ax.fill_between(hours, burned_areas, alpha=0.3, color="#e74c3c")
    ax.plot(hours, burned_areas, "o-", color="#e74c3c", linewidth=2)
    ax.set_xlabel("Hours since ignition", fontsize=12)
    ax.set_ylabel("Burned area (km²)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
