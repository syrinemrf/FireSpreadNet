# FireSpreadNet — Notebooks

> **Self-contained Jupyter notebooks for wildfire spread prediction research**

This directory contains a complete, publication-ready workflow for training and evaluating physics-informed deep learning models on real satellite data.

---

## Quick Start

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
pip install torch torchvision  # For training notebooks
pip install scikit-learn shap  # For evaluation & XAI
```

### Data Requirements
You need the **Next Day Wildfire Spread** dataset (Huot et al., 2022):
- Download from Kaggle: https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
- Place `.tfrecord` files in `../data/raw/` or any subdirectory
- The setup notebook will auto-detect them

---

## Workflow

### 1. Setup & Data Conversion: `00_Setup.ipynb`
**Run this first!** Converts TFRecord satellite data to numpy arrays.

**What it does:**
- Auto-detects TFRecord files on your system
- Converts to portable `.npz` format (train/val/test splits)
- Computes normalization statistics from training data
- Generates `setup_config.json` with relative paths (portable!)

**Output:**
- `../data/processed/{train,val,test}.npz` — preprocessed numpy arrays
- `setup_config.json` — configuration loaded by all other notebooks

**Runtime:** ~5-15 minutes (depends on dataset size)

---

### 2. Exploratory Data Analysis: `01_EDA.ipynb`
Publication-quality visualizations and statistical analysis.

**Key Figures Generated:**
- `fig01_feature_distributions.png` — Histograms of all 12 input channels
- `fig02_correlation_matrix.png` — Feature correlations (pixel-level)
- `fig03_fire_analysis.png` — Fire coverage, growth patterns, spread categories
- `fig04_fire_vs_nofire.png` — Feature distributions at fire vs non-fire pixels (with statistical tests)
- `fig05_sample_*.png` — Representative samples (low/medium/high fire coverage)
- `fig06_spatial_patterns.png` — Mean maps across training set
- `fig07_split_summary.png` — Train/val/test split statistics

**Statistical Tests:**
- Mann-Whitney U test for fire-discriminative features
- Pearson correlation for feature dependencies
- Class imbalance analysis (motivates Focal Loss)

**References:**
- Huot et al. (2022) — Dataset paper
- Lin et al. (2017) — Focal Loss motivation
- Tucker (1979) — NDVI interpretation

**Runtime:** ~3-5 minutes

---

### 3. Preprocessing Pipeline: `02_Preprocessing.ipynb`
Documents data normalization and augmentation strategies.

**What it covers:**
- **Z-score normalization** — Channel-wise standardization (except binary masks)
- **Spatial augmentation** — 90° rotations + flips (increases dataset by 8×)
- **Data quality checks** — NaN/Inf detection, range validation

**Key Figures:**
- `fig_preprocessing_01_normalization.png` — Raw vs normalized feature comparison
- `fig_preprocessing_02_augmentations.png` — Same sample with different augmentations

**Why it matters:**
- Prevents features with large magnitudes from dominating gradients
- Augmentation critical for preventing overfitting on limited satellite data
- Geographic split prevents spatial leakage (Huot et al., 2022)

**References:**
- Ioffe & Szegedy (2015) — Batch normalization theory
- Shorten & Khoshgoftaar (2019) — Augmentation survey
- Roberts et al. (2017) — Spatial cross-validation

**Runtime:** ~1-2 minutes

---

### 4. Model Training: `03_Model_Training.ipynb`
Trains four fire spread models (or visualizes pre-trained results).

**Models Compared:**
| Model | Type | Parameters | Key Feature |
|-------|------|-----------|-------------|
| CA | Physics | 0 | Rothermel equations (baseline) |
| ConvLSTM | Deep Learning | ~350K | Temporal recurrence |
| U-Net | Deep Learning | ~2.1M | Multi-scale attention |
| PI-CCA | Hybrid (Novel) | ~1.5M | Physics + CNN fusion |

**Training Configuration:**
- Loss: Focal (α=0.75, γ=2) + Dice (50:50 blend)
- Optimizer: AdamW (lr=1e-3, weight decay=1e-4)
- Regularization: Early stopping (patience=15), dropout, augmentation

**Key Figure:**
- `fig_training_01_curves.png` — Training/validation loss and metrics over epochs

**Note:** Requires `src/` directory with model implementations. If unavailable, notebook can still visualize pre-trained results from saved checkpoints.

**References:**
- Lin et al. (2017) — Focal Loss
- Milletari et al. (2016) — Dice Loss
- Shi et al. (2015) — ConvLSTM
- Oktay et al. (2018) — Attention U-Net
- Raissi et al. (2019) — Physics-informed neural networks

**Runtime:** 30-60 min (GPU), 2-4 hours (CPU), or instant (load checkpoints)

---

### 5. Results & Evaluation: `04_Results.ipynb`
Comprehensive test-set evaluation with publication-quality figures.

**Metrics Computed:**
- IoU (Intersection over Union / Jaccard index)
- Dice / F1 score
- Precision & Recall
- AUC-ROC (receiver operating characteristic)
- Average Precision (area under PR curve)

**Key Figures:**
- `fig_results_01_roc_curves.png` — ROC curves with AUC values
- `fig_results_02_pr_curves.png` — Precision-Recall curves
- `fig_results_03_visual_comparison.png` — Side-by-side predictions vs ground truth
- `fig_results_04_confusion_matrices.png` — Normalized confusion matrices
- `fig_results_05_pi_cca_uncertainty.png` — MC-Dropout uncertainty maps

**Expected Performance** (typical ranges):
- Pure Physics (CA): IoU ~0.20
- ConvLSTM: IoU ~0.40
- U-Net: IoU ~0.50
- **PI-CCA: IoU ~0.55** (best, physically consistent)

**References:**
- Dice (1945), Jaccard (1912) — Classic similarity metrics
- Fawcett (2006) — ROC analysis
- Gal & Ghahramani (2016) — MC-Dropout uncertainty

**Runtime:** ~5-10 minutes

---

### 6. Explainability (SHAP & Grad-CAM): `05_XAI_SHAP.ipynb`
Interprets model predictions using state-of-the-art XAI techniques.

**Techniques Applied:**
1. **SHAP (SHapley Additive exPlanations)** — Feature importance
   - DeepSHAP for neural networks (Lundberg & Lee, 2017)
   - Identifies which of the 12 input channels drive predictions

2. **Grad-CAM** — Spatial attention
   - Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
   - Shows which spatial regions models focus on

3. **Physics Gate Analysis** (PI-CCA only)
   - Model-intrinsic interpretability via learnable λ gate
   - Reveals when model trusts physics vs CNN

**Key Figures:**
- `fig_xai_01_shap_importance.png` — Feature importance rankings
- `fig_xai_02_gradcam_comparison.png` — Spatial attention heatmaps
- `fig_xai_03_physics_gate.png` — PI-CCA adaptive blending analysis

**Key Finding:**
- **prev_fire_mask** is the #1 predictor (fire spreads from existing perimeter)
- **ERC** and **drought_index** rank 2nd and 3rd (fuel availability)
- **PI-CCA** physics gate adapts: λ≈1 (simple terrain), λ≈0 (complex terrain)

**References:**
- Lundberg & Lee (2017) — SHAP theory
- Selvaraju et al. (2017) — Grad-CAM
- Molnar (2020) — Interpretable ML guide
- Samek et al. (2021) — XAI survey

**Runtime:** ~10-20 minutes (SHAP is computationally expensive)

---

## Portability & Standalone Usage

### These notebooks are designed to be self-contained:

✅ **What you need:**
- The notebooks themselves (`notebooks/*.ipynb`)
- The dataset (`.tfrecord` files or `.npz` files)
- Python packages listed above

✅ **What you DON'T need for EDA/Preprocessing:**
- No `src/` directory required for notebooks 00, 01, 02
- All helper functions are embedded in the notebooks
- Paths are auto-detected and relative (portable across systems)

⚠️ **For Training/Results/XAI (notebooks 03-05):**
- Models require `src/models/` implementations (~2000 lines of PyTorch code)
- If `src/` is unavailable, notebooks can still:
  - Load and visualize pre-trained model results
  - Analyze training curves from saved history files
  - Display evaluation metrics and XAI results

---

## Figure Naming Convention

All figures follow a consistent naming pattern for easy reference in papers:
```
fig01_*.png     — EDA figures (Feature distributions, correlations, fire analysis)
fig02_*.png     — EDA continued
...
fig_preprocessing_*.png  — Preprocessing visualizations
fig_training_*.png       — Training curves
fig_results_*.png        — Test set evaluation
fig_xai_*.png           — Explainability analysis
```

All figures are saved at **300 DPI** (publication quality).

---

## Data Format

### Input Data Structure
After running `00_Setup.ipynb`, data is stored as:
```
data/
├── raw/               # Original TFRecord files (not modified)
└── processed/
    ├── train.npz      # {'X': (14979, 12, 64, 64), 'Y': (14979, 1, 64, 64)}
    ├── val.npz        # {'X': (1877, 12, 64, 64), 'Y': (1877, 1, 64, 64)}
    └── test.npz       # {'X': (1689, 12, 64, 64), 'Y': (1689, 1, 64, 64)}
```

### Configuration File
`setup_config.json` contains:
```json
{
  "PROCESSED_DIR_REL": "../data/processed",
  "FIGURES_DIR_REL": "../results/figures",
  "FEATURE_CHANNELS": ["elevation", "wind_speed", ...],
  "CH": {"elevation": 0, "wind_speed": 1, ...},
  "norm_stats": {"elevation": {"mean": 896.57, "std": 842.61}, ...},
  "split_samples": {"train": 14979, "val": 1877, "test": 1689}
}
```

---

## Troubleshooting

### "FileNotFoundError: setup_config.json not found"
→ Run `00_Setup.ipynb` first to convert data and generate config

### "No data found in ../data/processed/"
→ Ensure TFRecord files are accessible, then run `00_Setup.ipynb`

### "ImportError: No module named 'src'"
→ For training notebooks (03-05), the complete repo with `src/` is needed
→ Or use pre-trained model checkpoints if available

### Figures not saving
→ Check that `../results/figures/` directory is writable
→ The notebooks create this directory automatically

### Slow execution
→ Reduce sample sizes in EDA (change `n_sub`, `n_samples_corr` variables)
→ Use CPU if GPU is unavailable (slower but functional)

---

## Citation

If you use these notebooks or figures in your research, please cite:

**Dataset:**
```bibtex
@article{huot2022next,
  title={Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data},
  author={Huot, Fantine and Hu, R. Lynn and Goyal, Nita and Sanoja, Thaïs and Ihme, Matthias and Chen, Yi-Fan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--13},
  year={2022},
  publisher={IEEE},
  doi={10.1109/TGRS.2022.3192974}
}
```

**This Work (PI-CCA):**
```bibtex
@misc{firespreadnet2024,
  title={Physics-Informed Convolutional Cellular Automaton for Next-Day Wildfire Spread Prediction},
  author={FireSpreadNet Research},
  year={2024},
  note={Research notebooks available at https://github.com/syrinemrf/FireSpreadNet}
}
```

---

## License

MIT License — Free for academic and commercial use.

---

## Contact & Contributions

Found an issue or have suggestions? Open an issue on GitHub!

**Reproducibility Checklist:**
- [x] All data sources documented with references
- [x] Preprocessing steps fully described
- [x] Model architectures explained with citations
- [x] Evaluation metrics defined formally
- [x] Figures publication-ready (300 DPI, clear labels)
- [x] Complete reference list provided
- [x] Notebooks self-contained and portable
