#!/usr/bin/env python3
"""
src/explainability/shap_analysis.py — SHAP Explainability for Fire Models
==========================================================================
Channel-wise SHAP analysis for spatial fire spread models.

Since inputs are (C, H, W) grids, we compute SHAP values per channel
(feature importance) by treating each channel as a "feature".

For spatial explanations, we also provide GradCAM-style attribution maps
that show WHERE the model focuses for its predictions.

References
----------
  Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting
      Model Predictions. NeurIPS 2017.
  Selvaraju, R.R. et al. (2017). Grad-CAM: Visual Explanations from Deep
      Networks via Gradient-based Localization. ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# CHANNEL-WISE SHAP (Kernel SHAP on channel means)
# ══════════════════════════════════════════════════════════════

def compute_channel_shap(
    model: torch.nn.Module,
    data_loader,
    feature_names: list,
    n_background: int = 50,
    n_explain: int = 100,
    device: str = "cpu",
) -> dict:
    """Compute SHAP values per input channel.

    Strategy: reduce each (C, H, W) input to (C,) by spatial averaging,
    then use Kernel SHAP to attribute the model's output to each channel.

    Parameters
    ----------
    model : trained model
    data_loader : DataLoader
    feature_names : list of channel names
    n_background : number of background samples
    n_explain : number of samples to explain

    Returns
    -------
    dict with keys: shap_values (n_explain, C), feature_names, mean_abs_shap
    """
    import shap

    model.eval()
    model.to(device)

    # Collect samples
    all_x = []
    for x, _ in data_loader:
        all_x.append(x)
        if sum(xi.shape[0] for xi in all_x) >= n_background + n_explain:
            break
    all_x = torch.cat(all_x)[:n_background + n_explain]

    background = all_x[:n_background].to(device)
    explain = all_x[n_background:n_background + n_explain].to(device)

    # Wrapper: model takes (B, C, H, W), returns scalar (mean fire prob)
    def model_fn(x_tensor):
        with torch.no_grad():
            out = model(x_tensor)
            return out.mean(dim=(1, 2, 3))  # (B,) — mean probability

    # Channel-wise summarisation for SHAP
    # Reduce to (N, C) by spatial averaging
    bg_summary = background.mean(dim=(2, 3)).cpu().numpy()   # (n_bg, C)
    exp_summary = explain.mean(dim=(2, 3)).cpu().numpy()     # (n_exp, C)

    # Wrapper for reduced input
    def reduced_model_fn(x_reduced):
        """Reconstruct spatial input from channel means and predict."""
        x_tensor = torch.tensor(x_reduced, dtype=torch.float32, device=device)
        B, C = x_tensor.shape
        H, W = background.shape[2], background.shape[3]
        # Broadcast channel means to spatial grid (rough approximation)
        x_spatial = x_tensor.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        with torch.no_grad():
            out = model(x_spatial)
        return out.mean(dim=(1, 2, 3)).cpu().numpy()

    explainer = shap.KernelExplainer(reduced_model_fn, bg_summary)
    shap_values = explainer.shap_values(exp_summary, nsamples=200)

    # Mean absolute SHAP
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_order = np.argsort(mean_abs)[::-1]

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs,
        "importance_order": importance_order,
        "base_values": explainer.expected_value,
    }


# ══════════════════════════════════════════════════════════════
# SPATIAL GRAD-CAM
# ══════════════════════════════════════════════════════════════

def compute_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_layer_name: str = None,
    device: str = "cpu",
) -> np.ndarray:
    """Compute Grad-CAM saliency map for a fire spread model.

    Parameters
    ----------
    model : trained model
    x : (1, C, H, W) single sample
    target_layer_name : name of the conv layer to hook

    Returns
    -------
    (H, W) saliency map
    """
    model.eval()
    model.to(device)
    x = x.to(device).requires_grad_(True)

    # Find target layer
    target_layer = None
    if target_layer_name:
        for name, module in model.named_modules():
            if name == target_layer_name:
                target_layer = module
                break

    if target_layer is None:
        # Auto-detect last conv layer
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")

    # Hook
    activations = []
    gradients = []

    def fwd_hook(mod, inp, out):
        activations.append(out)

    def bwd_hook(mod, grad_in, grad_out):
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # Forward + backward
    output = model(x)
    loss = output.mean()
    loss.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Grad-CAM
    act = activations[0].detach()    # (1, C_layer, H', W')
    grad = gradients[0].detach()     # (1, C_layer, H', W')
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C_layer, 1, 1)

    cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Normalise
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ══════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def plot_shap_importance(
    shap_result: dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    top_k: int = 12,
) -> plt.Figure:
    """Bar plot of mean absolute SHAP values per channel."""
    names = np.array(shap_result["feature_names"])
    importance = shap_result["mean_abs_shap"]
    order = shap_result["importance_order"][:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, importance[order], color="#e74c3c", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_beeswarm(
    shap_result: dict,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """SHAP beeswarm plot (uses shap library)."""
    import shap as shap_lib

    fig = plt.figure(figsize=(10, 8))
    shap_lib.summary_plot(
        shap_result["shap_values"],
        feature_names=shap_result["feature_names"],
        show=False,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gradcam_overlay(
    cam: np.ndarray,
    fire_state: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlay Grad-CAM saliency on fire state map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(fire_state, cmap="RdYlGn_r", interpolation="nearest")
    axes[0].set_title("Input Fire State", fontsize=12, fontweight="bold")

    axes[1].imshow(cam, cmap="jet", interpolation="nearest")
    axes[1].set_title("Grad-CAM Saliency", fontsize=12, fontweight="bold")

    axes[2].imshow(fire_state, cmap="gray", alpha=0.5, interpolation="nearest")
    axes[2].imshow(cam, cmap="jet", alpha=0.6, interpolation="nearest")
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")

    fig.suptitle(f"Grad-CAM Analysis — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
