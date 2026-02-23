#!/usr/bin/env python3
"""
config.py — Central Configuration for FireSpread
=================================================
Physics-Informed Deep Learning for Wildfire Propagation Simulation

Dataset: Next Day Wildfire Spread  (Huot et al., 2022 — IEEE TGRS)
  Real satellite observations: MODIS/VIIRS active fire, GRIDMET weather,
  SRTM topography, VIIRS vegetation.
  https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread

References:
  - Huot, F. et al. (2022). Next Day Wildfire Spread: A Machine Learning
    Dataset to Predict Wildfire Spreading from Remote-Sensing Data.
    IEEE Transactions on Geoscience and Remote Sensing.
  - Rothermel, R.C. (1972). A mathematical model for predicting fire spread.
  - Alexandridis, A. et al. (2008). A cellular automata model for forest fire
    spread prediction. Applied Mathematics and Computation.
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "saved_models"

for _d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# GRID CONFIGURATION (matches the real dataset)
# ══════════════════════════════════════════════════════════════
GRID_SIZE = 64              # NxN grid cells  (dataset native)
CELL_SIZE = 1000.0          # ~1 km per pixel  (MODIS resolution)
TIMESTEP = 86400.0          # 1 day (24 h) — dataset temporal resolution

# ══════════════════════════════════════════════════════════════
# DATASET — Next Day Wildfire Spread  (Huot et al., 2022)
# Real satellite / reanalysis data from:
#   - Terrain: SRTM (elevation)
#   - Weather: GRIDMET (wind, temp, humidity, precip, drought, ERC)
#   - Vegetation: VIIRS (NDVI)
#   - Population: LandScan
#   - Fire: FIRMS / VIIRS (active fire detections)
# ══════════════════════════════════════════════════════════════
FEATURE_CHANNELS = [
    "elevation",       # SRTM DEM (metres)
    "wind_speed",      # GRIDMET 'th' — wind speed (m/s)
    "wind_direction",  # GRIDMET 'vs' — wind direction (degrees from N)
    "min_temp",        # GRIDMET 'tmmn' — daily minimum temperature (K)
    "max_temp",        # GRIDMET 'tmmx' — daily maximum temperature (K)
    "humidity",        # GRIDMET 'sph' — specific humidity (kg/kg)
    "precipitation",   # GRIDMET 'pr' — precipitation (mm)
    "drought_index",   # GRIDMET 'PDSI' — Palmer Drought Severity Index
    "ndvi",            # VIIRS NDVI — vegetation greenness (–1 to 1)
    "erc",             # GRIDMET 'ERC' — Energy Release Component
    "population",      # LandScan population density (people/km²)
    "prev_fire_mask",  # FIRMS/VIIRS — active fire mask at day t (binary)
]

N_INPUT_CHANNELS = len(FEATURE_CHANNELS)   # 12

# Mapping: our readable names → TFRecord column names in the dataset
TFRECORD_INPUT_KEYS = [
    "elevation", "th", "vs", "tmmn", "tmmx",
    "sph", "pr", "PDSI", "NDVI", "ERC",
    "population", "PrevFireMask",
]
TFRECORD_TARGET_KEY = "FireMask"

# Channel indices for physics-based models
CH = {name: i for i, name in enumerate(FEATURE_CHANNELS)}
# CH['elevation']=0, CH['wind_speed']=1, ..., CH['prev_fire_mask']=11

# ══════════════════════════════════════════════════════════════
# DATASET DOWNLOAD CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATASET_CONFIG = {
    "name": "next_day_wildfire_spread",
    "source": "kaggle",
    "kaggle_dataset": "fantineh/next-day-wildfire-spread",
    "n_train_shards": 64,
    "n_val_shards": 16,
    "n_test_shards": 16,
    "grid_size": GRID_SIZE,
    "pixel_size_km": 1.0,
}

# ══════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    "ca": {
        "name": "Cellular Automata (Alexandridis)",
        "type": "physics",
        "neighborhood": "moore",
        "p_burn_base": 0.58,
        "wind_weight": 0.045,
        "slope_weight": 0.078,
    },
    "convlstm": {
        "name": "ConvLSTM",
        "type": "deep_learning",
        "hidden_channels": [32, 64, 32],
        "kernel_size": 3,
        "num_layers": 3,
        "dropout": 0.2,
    },
    "unet": {
        "name": "U-Net Fire",
        "type": "deep_learning",
        "base_filters": 32,
        "depth": 4,
        "dropout": 0.2,
        "use_attention": True,
    },
    "pi_cca": {
        "name": "PI-CCA",
        "type": "hybrid",
        "physics_channels": 32,
        "cnn_channels": 64,
        "attention_heads": 4,
        "n_res_blocks": 3,
        "dropout": 0.15,
        "mc_samples": 20,
        "learnable_physics": True,
    },
}

# ══════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════
TRAIN_CONFIG = {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "dice_weight": 0.5,
    "focal_weight": 0.5,
    "early_stopping_patience": 10,
    "gradient_clip": 1.0,
}

# ══════════════════════════════════════════════════════════════
# RANDOM SEED
# ══════════════════════════════════════════════════════════════
SEED = 42
