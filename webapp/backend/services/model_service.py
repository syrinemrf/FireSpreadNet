"""Model loading and inference service."""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    MODELS_DIR, MODEL_CONFIGS, MODEL_PRIORITY,
    N_INPUT_CHANNELS, GRID_SIZE, NORM_STATS, CH,
)


class ModelService:
    """Singleton: loads the best available trained model and runs inference."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── loading ──────────────────────────────────────────────
    def load_model(self):
        """Try to load models in priority order (PI-CCA > U-Net > ConvLSTM)."""
        from src.models.pi_cca import PIConvCellularAutomaton
        from src.models.unet import UNetFire
        from src.models.convlstm import ConvLSTMModel

        model_classes = {
            "pi_cca":   PIConvCellularAutomaton,
            "unet":     UNetFire,
            "convlstm": ConvLSTMModel,
        }

        for name in MODEL_PRIORITY:
            ckpt = MODELS_DIR / name / "best_model.pt"
            if ckpt.exists():
                try:
                    model = model_classes[name](config=MODEL_CONFIGS[name])
                    model.load_state_dict(
                        torch.load(ckpt, map_location=self.device, weights_only=True)
                    )
                    model.to(self.device).eval()
                    self.model = model
                    self.model_name = name
                    print(f"✓ Loaded model: {name} from {ckpt}")
                    return
                except Exception as e:
                    print(f"✗ Failed to load {name}: {e}")

        print("⚠ No trained model found — simulation will use physics fallback")

    # ── inference ────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a (1, 12, 64, 64) tensor. Returns (64, 64) probabilities."""
        if self.model is None:
            return self._physics_fallback(input_tensor)
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        return output.squeeze().cpu().numpy()

    # ── helpers ──────────────────────────────────────────────
    def build_input_tensor(self, weather: dict, elevation_grid: np.ndarray,
                           fire_mask: np.ndarray) -> torch.Tensor:
        """Build a normalised (1, 12, 64, 64) tensor from real-world data."""
        grid = np.zeros((N_INPUT_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        grid[CH["elevation"]]      = elevation_grid
        grid[CH["wind_speed"]]     = weather.get("wind_speed", 5.0)
        grid[CH["wind_direction"]] = weather.get("wind_direction", 180.0)
        grid[CH["min_temp"]]       = weather.get("min_temp", 280.0)
        grid[CH["max_temp"]]       = weather.get("max_temp", 300.0)
        grid[CH["humidity"]]       = weather.get("humidity", 0.005)
        grid[CH["precipitation"]]  = weather.get("precipitation", 0.0)
        grid[CH["drought_index"]]  = weather.get("drought_index", 0.0)
        grid[CH["ndvi"]]           = weather.get("ndvi", 5000.0)
        grid[CH["erc"]]            = weather.get("erc", 50.0)
        grid[CH["population"]]     = weather.get("population", 25.0)
        grid[CH["prev_fire_mask"]] = fire_mask

        # Normalise
        for ch_name, idx in CH.items():
            stats = NORM_STATS[ch_name]
            grid[idx] = (grid[idx] - stats["mean"]) / (stats["std"] + 1e-8)

        return torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    def get_feature_importance(self) -> dict:
        """Return approximate feature importance from training analysis."""
        return {
            "prev_fire_mask": 0.80,
            "wind_speed": 0.08,
            "humidity": 0.04,
            "max_temp": 0.02,
            "elevation": 0.02,
            "ndvi": 0.015,
            "erc": 0.01,
            "precipitation": 0.005,
            "drought_index": 0.004,
            "wind_direction": 0.003,
            "min_temp": 0.002,
            "population": 0.001,
        }

    @staticmethod
    def _physics_fallback(input_tensor: torch.Tensor) -> np.ndarray:
        """Simple distance-based spread when no ML model is available."""
        fire = input_tensor[0, CH["prev_fire_mask"]].numpy()
        from scipy.ndimage import binary_dilation
        spread = binary_dilation(fire > 0.5, iterations=2).astype(np.float32) * 0.6
        return np.clip(spread, 0, 1)


model_service = ModelService()
