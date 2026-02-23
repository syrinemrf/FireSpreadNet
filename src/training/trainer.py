#!/usr/bin/env python3
"""
src/training/trainer.py — Training Loop & Evaluation for Fire Spread Models
============================================================================
Unified trainer supporting all four model types. Includes:
  - Focal + Dice combined loss (handles class imbalance: fire cells ≪ total)
  - Cosine-annealing LR scheduler with warm-up
  - Early stopping with patience
  - Full evaluation suite: IoU, Dice, F1, Precision, Recall, AUC
  - Automatic checkpointing of best model
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from config import TRAIN_CONFIG, MODELS_DIR


# ══════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for imbalanced fire/no-fire cells."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss for spatial overlap."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Weighted sum of Focal + Dice losses."""

    def __init__(self, focal_w: float = 0.5, dice_w: float = 0.5,
                 alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.focal_w = focal_w
        self.dice_w = dice_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_w * self.focal(pred, target) + \
               self.dice_w * self.dice(pred, target)


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute fire spread evaluation metrics.

    Parameters
    ----------
    pred   : (B, 1, H, W) float — predicted probabilities
    target : (B, 1, H, W) float — binary ground truth

    Returns
    -------
    dict with iou, dice, f1, precision, recall, accuracy, auc_approx
    """
    with torch.no_grad():
        p_bin = (pred > threshold).float()
        t_bin = target.float()

        tp = (p_bin * t_bin).sum().item()
        fp = (p_bin * (1 - t_bin)).sum().item()
        fn = ((1 - p_bin) * t_bin).sum().item()
        tn = ((1 - p_bin) * (1 - t_bin)).sum().item()

        eps = 1e-8
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
        specificity = tn / (tn + fp + eps)

    return {
        "iou": iou,
        "dice": dice,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
    }


# ══════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════

class Trainer:
    """Training and evaluation for fire spread models."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device = None,
        config: dict = None,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = config or TRAIN_CONFIG
        self.model.to(self.device)

        # Loss
        self.criterion = CombinedLoss(
            focal_w=self.cfg["focal_weight"],
            dice_w=self.cfg["dice_weight"],
            alpha=self.cfg["focal_alpha"],
            gamma=self.cfg["focal_gamma"],
        )

        # Optimiser (skip if model has no learnable params)
        params = [p for p in model.parameters() if p.requires_grad]
        if params:
            self.optimizer = torch.optim.AdamW(
                params, lr=self.cfg["learning_rate"],
                weight_decay=self.cfg["weight_decay"],
            )
        else:
            self.optimizer = None

        self.scheduler = None
        self.history = {"train_loss": [], "val_loss": [],
                        "train_metrics": [], "val_metrics": []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop with early stopping."""
        epochs = self.cfg["epochs"]
        patience = self.cfg["early_stopping_patience"]
        best_val_loss = float("inf")
        no_improve = 0

        # Scheduler
        if self.optimizer:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )

        save_dir = MODELS_DIR / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training {self.model_name} on {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train epoch
            if self.optimizer:
                train_loss, train_metrics = self._train_epoch(train_loader)
            else:
                # Physics-only model — just evaluate
                train_loss, train_metrics = self._eval_epoch(train_loader)

            # Validation
            val_loss, val_metrics = self._eval_epoch(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # History
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_metrics"].append(train_metrics)
            self.history["val_metrics"].append(val_metrics)

            dt = time.time() - t0
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_metrics['iou']:.4f} | "
                  f"Val Dice: {val_metrics['dice']:.4f} | "
                  f"Time: {dt:.1f}s")

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), save_dir / "best_model.pt")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save history
        serialisable_history = {
            k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
            for k, vals in self.history.items()
        }
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(serialisable_history, f, indent=2, default=str)

        return self.history

    def _train_epoch(self, loader: DataLoader):
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        for x, y in tqdm(loader, desc="Train", leave=False):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()

            if self.cfg.get("gradient_clip"):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["gradient_clip"])

            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            all_preds.append(pred.detach())
            all_targets.append(y.detach())

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        return avg_loss, metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []

        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.criterion(pred, y)
            total_loss += loss.item() * x.size(0)
            all_preds.append(pred)
            all_targets.append(y)

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Final evaluation on test set."""
        self.model.eval()
        all_preds, all_targets = [], []
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            all_preds.append(pred)
            all_targets.append(y)

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics = compute_metrics(preds, targets)

        # Loss
        loss = self.criterion(preds, targets).item()
        metrics["test_loss"] = loss

        return metrics
