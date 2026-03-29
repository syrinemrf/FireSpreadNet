import json
import torch
from pathlib import Path
from config import MODEL_CONFIG, TRAIN_CONFIG, SEED
from src.data.dataset import get_dataloaders
from src.models.convlstm import ConvLSTMModel
from src.models.unet import UNetFire
from src.models.pi_cca import PIConvCellularAutomaton
from src.training.trainer import Trainer

cfg = json.load(open(Path('notebooks') / 'setup_config.json'))
processed_dir = Path(cfg['PROCESSED_DIR'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)
if device.type != 'cuda':
    raise SystemExit('No CUDA')

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
bs = int(TRAIN_CONFIG['batch_size'])
if vram_gb <= 3.5:
    bs = min(bs, 4)
elif vram_gb <= 4.0:
    bs = min(bs, 8)
print('vram_gb', round(vram_gb, 2), 'batch_size', bs)

loaders = get_dataloaders(processed_dir, batch_size=bs, num_workers=0, augment_train=True, seed=SEED)

model_map = {
    'convlstm': ConvLSTMModel,
    'unet': UNetFire,
    'pi_cca': PIConvCellularAutomaton,
}

for name in ['convlstm', 'unet', 'pi_cca']:
    print('\n--- quick train', name, '---')
    model = model_map[name](config=MODEL_CONFIG[name]).to(device)
    cfg_train = TRAIN_CONFIG.copy()
    cfg_train.update({
        'epochs': 1,
        'learning_rate': 2e-4,
        'weight_decay': 5e-5,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'early_stopping_patience': 1,
    })
    trainer = Trainer(model=model, model_name=f'{name}_quickcheck', device=device, config=cfg_train)
    _ = trainer.train(loaders['train'], loaders['val'])
    print('OK', name)

print('\nquick train sanity completed')
