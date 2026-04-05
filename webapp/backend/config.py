"""Backend configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# MODELS_DIR can be overridden at runtime (e.g. inside Docker / Cloud Run)
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "saved_models")))
SETUP_CONFIG_PATH = Path(os.getenv("SETUP_CONFIG_PATH", str(PROJECT_ROOT / "notebooks" / "setup_config.json")))

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")
OPEN_METEO_BASE = "https://api.open-meteo.com/v1/forecast"
OPEN_ELEVATION_BASE = "https://api.open-elevation.com/api/v1/lookup"

GRID_SIZE = 64
CELL_SIZE_KM = 1.0
N_INPUT_CHANNELS = 12

FEATURE_CHANNELS = [
    "elevation", "wind_speed", "wind_direction", "min_temp", "max_temp",
    "humidity", "precipitation", "drought_index", "ndvi", "erc",
    "population", "prev_fire_mask",
]

CH = {name: i for i, name in enumerate(FEATURE_CHANNELS)}

NORM_STATS = {
    "elevation":      {"mean": 896.571, "std": 842.610},
    "wind_speed":     {"mean": 146.647, "std": 3435.084},
    "wind_direction": {"mean": 3.628,   "std": 1.309},
    "min_temp":       {"mean": 281.852, "std": 18.497},
    "max_temp":       {"mean": 297.717, "std": 19.458},
    "humidity":       {"mean": 0.00653, "std": 0.00374},
    "precipitation":  {"mean": 0.323,   "std": 1.534},
    "drought_index":  {"mean": -0.773,  "std": 2.441},
    "ndvi":           {"mean": 5350.681,"std": 2185.219},
    "erc":            {"mean": 53.469,  "std": 25.098},
    "population":     {"mean": 30.460,  "std": 214.200},
    "prev_fire_mask": {"mean": -0.003,  "std": 0.138},
}

MODEL_CONFIGS = {
    "ca":       {"name": "Cellular Automata (CA)", "type": "Physics-based baseline"},
    "convlstm": {"name": "ConvLSTM", "type": "Recurrent neural network", "hidden_channels": [32, 64, 32]},
    "unet":     {"name": "U-Net + Attention", "type": "Segmentation network"},
    "pi_cca":   {"name": "PI-CCA", "type": "Physics-informed hybrid"},
}

MODEL_PRIORITY = ["pi_cca", "unet", "convlstm"]

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
