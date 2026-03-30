import json
from pathlib import Path

SEED = 42

_setup_cfg_path = Path(__file__).parent / 'setup_config.json'
if not _setup_cfg_path.exists():
    raise FileNotFoundError(f'setup_config.json not found at {_setup_cfg_path}')

_setup_cfg = json.load(open(_setup_cfg_path))

PROCESSED_DIR = Path(_setup_cfg['PROCESSED_DIR'])
FEATURE_CHANNELS = _setup_cfg['FEATURE_CHANNELS']
N_INPUT_CHANNELS = _setup_cfg['N_INPUT_CHANNELS']
CH = _setup_cfg['CH']
norm_stats = _setup_cfg['norm_stats']

MODELS_DIR = Path(__file__).parent / 'saved_models'
RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'results' / 'figures'

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    'ca': {'name': 'Cellular Automata (CA)', 'type': 'Physics-based baseline'},
    'convlstm': {'name': 'ConvLSTM', 'type': 'Recurrent neural network', 'hidden_channels': [32, 64, 32]},
    'unet': {'name': 'U-Net + Attention', 'type': 'Segmentation network'},
    'pi_cca': {'name': 'PI-CCA', 'type': 'Physics-informed hybrid'},
}

TRAIN_CONFIG = {'epochs': 40, 'learning_rate': 3e-4, 'weight_decay': 1e-4}
