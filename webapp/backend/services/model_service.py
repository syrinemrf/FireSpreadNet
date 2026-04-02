"""Model loading and inference service."""

import math
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.ndimage import binary_dilation, gaussian_filter

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

        # Elevation with spatial variation
        if elevation_grid.shape == (GRID_SIZE, GRID_SIZE):
            grid[CH["elevation"]] = elevation_grid
        else:
            grid[CH["elevation"]] = float(elevation_grid.flat[0]) if elevation_grid.size > 0 else 500.0

        # Weather channels — uniform across grid (real weather is regional)
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
        """Wind-aware, terrain-influenced fire spread when no ML model is available.
        
        Uses:
        - Wind direction and speed for asymmetric spread
        - Slope (from elevation gradient) for uphill/downhill effects
        - Humidity for moisture damping
        - NDVI for vegetation fuel load
        """
        fire = input_tensor[0, CH["prev_fire_mask"]].numpy()
        wind_speed_norm = input_tensor[0, CH["wind_speed"]].numpy()
        wind_dir_norm = input_tensor[0, CH["wind_direction"]].numpy()
        humidity_norm = input_tensor[0, CH["humidity"]].numpy()
        elevation_norm = input_tensor[0, CH["elevation"]].numpy()
        ndvi_norm = input_tensor[0, CH["ndvi"]].numpy()

        # Denormalize key values (use center pixel as representative)
        ws_mean, ws_std = NORM_STATS["wind_speed"]["mean"], NORM_STATS["wind_speed"]["std"]
        wd_mean, wd_std = NORM_STATS["wind_direction"]["mean"], NORM_STATS["wind_direction"]["std"]
        hum_mean, hum_std = NORM_STATS["humidity"]["mean"], NORM_STATS["humidity"]["std"]

        wind_speed = float(wind_speed_norm[GRID_SIZE//2, GRID_SIZE//2]) * ws_std + ws_mean
        wind_dir_deg = float(wind_dir_norm[GRID_SIZE//2, GRID_SIZE//2]) * wd_std + wd_mean
        humidity = float(humidity_norm[GRID_SIZE//2, GRID_SIZE//2]) * hum_std + hum_mean

        # Clamp to reasonable ranges
        wind_speed = max(0, min(wind_speed, 100))
        humidity = max(0.001, min(humidity, 0.02))
        
        # Convert normalized fire mask to binary
        fire_binary = (fire > 0).astype(np.float32)
        
        if fire_binary.sum() == 0:
            return np.zeros_like(fire)

        # ── Base spread (isotropic dilation) ──
        base_spread = binary_dilation(fire_binary > 0.5, iterations=1).astype(np.float32)
        
        # ── Wind-directed spread ──
        # Create directional kernel based on wind direction
        wind_dir_rad = math.radians(wind_dir_deg)
        # Wind blows FROM this direction, fire spreads in opposite direction
        spread_dir_y = -math.cos(wind_dir_rad)  # North component
        spread_dir_x = math.sin(wind_dir_rad)   # East component
        
        # Create a 5x5 directional kernel
        kernel_size = 5
        half = kernel_size // 2
        wind_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                if dr == 0 and dc == 0:
                    continue
                dist = math.sqrt(dr * dr + dc * dc)
                if dist > half:
                    continue
                # Direction alignment with wind (fire moves downwind)
                if dist > 0:
                    dot = (-dr * spread_dir_y + dc * spread_dir_x) / dist
                else:
                    dot = 0
                # Weight: stronger in wind direction, weaker against
                weight = 0.1 + 0.9 * max(0, (dot + 1) / 2)  # 0.1 to 1.0
                # Distance falloff
                weight *= 1.0 / (1.0 + dist * 0.3)
                wind_kernel[dr + half, dc + half] = weight
        
        # Normalize kernel
        if wind_kernel.max() > 0:
            wind_kernel /= wind_kernel.max()
        
        # Apply wind kernel as custom dilation
        from scipy.ndimage import convolve
        wind_spread = convolve(fire_binary, wind_kernel, mode='constant', cval=0)
        
        # Wind speed factor (stronger wind = faster spread)
        wind_factor = 0.3 + min(0.7, wind_speed / 30.0)  # 0.3 to 1.0
        
        # ── Slope effect ──
        # Compute slope from elevation (fire spreads faster uphill)
        dy = np.gradient(elevation_norm, axis=0)
        dx = np.gradient(elevation_norm, axis=1)
        slope_mag = np.sqrt(dx**2 + dy**2)
        slope_factor = 1.0 + np.clip(slope_mag * 2.0, 0, 0.5)  # 1.0 to 1.5
        
        # ── Humidity damping ──
        # Higher humidity = slower spread
        humidity_factor = max(0.3, 1.0 - humidity * 80)  # 0.3 to ~0.9
        
        # ── Vegetation fuel ──
        # Higher NDVI (normalized) = more fuel
        ndvi_center = float(ndvi_norm[GRID_SIZE//2, GRID_SIZE//2])
        veg_factor = 0.5 + 0.5 * min(1.0, max(0, (ndvi_center + 2) / 4))  # 0.5 to 1.0
        
        # ── Combine ──
        combined = (
            base_spread * 0.3 +          # Isotropic base
            wind_spread * wind_factor * 0.7  # Wind-directed
        )
        combined *= slope_factor
        combined *= humidity_factor
        combined *= veg_factor

        # Threshold to get new fire cells
        spread_prob = np.clip(combined, 0, 1)
        
        # Add some noise for natural-looking spread
        noise = np.random.uniform(0, 0.15, spread_prob.shape).astype(np.float32)
        spread_prob = np.clip(spread_prob - noise, 0, 1)
        
        # Smooth edges for realistic appearance
        spread_prob = gaussian_filter(spread_prob, sigma=0.8)
        
        # Ensure original fire cells remain
        spread_prob = np.maximum(spread_prob, fire_binary * 0.9)
        
        return np.clip(spread_prob, 0, 1)


model_service = ModelService()
