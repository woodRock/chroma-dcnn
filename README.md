# MSFormer

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A transformer foundation model pretrained on GC-MS reference spectra via **Masked Spectra Modelling (MSM)** and evaluated on two food authentication tasks: olive oil geographic origin (GC-IMS) and Thai/foreign hemp seed discrimination (GC-MS).

---

## Overview

MSFormer applies BERT-style self-supervised pretraining to mass spectra. The encoder is pretrained on ~30k EI-MS reference spectra from [MoNA](https://mona.fiehnlab.ucdavis.edu) and [MassBank EU](https://massbank.eu), then evaluated under four experimental conditions:

| Condition | Description |
|---|---|
| `from_scratch` | Random init, trained on downstream data only |
| `msm_finetune` | Pretrained weights, all layers fine-tuned |
| `linear_probe_msm` | Pretrained encoder frozen, only classification head trained |

Results are compared against PLS-DA, Random Forest, and SVM baselines using the same cross-validation protocol.

---

## Repository Structure

```
msformer/
├── configs/                   # YAML configs for pretraining and fine-tuning
│   ├── pretrain_msm.yaml
│   ├── finetune_oliveoil.yaml
│   └── finetune_hemp.yaml
├── data/                      # Datasets (see Data Setup below)
│   ├── pretraining/raw/       # MoNA + MassBank reference spectra
│   ├── olive_oil/             # Pre-processed GC-IMS numpy arrays
│   └── hemp_seed/             # GC-MS peak table (Excel)
├── scripts/                   # Run in order: 01 → 02 → 03 → 04 → 05 → 06
├── src/msformer/
│   ├── data/                  # Download, preprocess, dataset classes
│   ├── models/                # SpectrumEncoder, MSMModel, SpectrumClassifier
│   ├── training/              # PretrainTrainer, Finetuner
│   ├── evaluation/            # Baselines, stats, attention visualisation
│   └── downstream/            # Olive oil and hemp seed data loaders
├── notebooks/                 # Results analysis
├── checkpoints/               # Saved model weights (git-ignored)
└── results/                   # CSV and JSON outputs (git-ignored)
```

---

## Prerequisites

- Python 3.9 or later
- CUDA-capable GPU recommended for pretraining (scripts fall back to MPS or CPU)
- ~5 GB free disk space for pretraining data and checkpoints

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/woodRock/msformer.git
cd msformer
conda env create -f environment.yml
conda activate msformer
```

### Option B — pip

```bash
git clone https://github.com/woodRock/msformer.git
cd msformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For the olive oil GC-IMS baseline (optional):

```bash
pip install -e ".[gcims]"
```

---

## Data Setup

### Pretraining data (auto-downloaded)

Script `01` handles MassBank automatically. MoNA source files (`mona_gcms_part1/2/3.json`) are committed to this repo and reassembled on first run.

### Olive oil — GC-IMS geographic origin

**Download required — file is 3 GB and cannot be committed to git.**

1. Download from Mendeley Data:
   **[DOI 10.17632/fr9t5fkkvz.3](https://data.mendeley.com/datasets/fr9t5fkkvz/3)**
   *(Christmann & Weller, 2022 — EVOO samples, Spain / Italy / Greece)*

2. Place the zip in `data/olive_oil/`:
   ```
   data/olive_oil/Olive oil geography by GC-IMS analysis.zip
   ```

3. Run the preprocessing script to extract numpy arrays (~30 seconds):
   ```bash
   python scripts/preprocess_olive_oil.py
   ```
   This streams `.mea` files from the zip without extracting the full 3 GB archive.

Pre-processed arrays (`X_sum.npy`, `X_apex.npy`, `y.npy`, `sample_ids.txt`) are already committed — you can skip steps 1–3 if you only want to run fine-tuning.

### Hemp seed — Thai vs foreign GC-MS

**Download required.**

1. Download from Zenodo:
   **[DOI 10.5281/zenodo.17505720](https://zenodo.org/records/17505720)**
   *(Sangkanu et al., 2026 — Thai vs foreign Cannabis sativa seed extracts)*

2. Extract and place the contents under `data/hemp_seed/`:
   ```
   data/hemp_seed/GC_MS_data/hemp-gcms-original.xlsx
   ```

---

## Usage

Run scripts in order. Each step depends on the previous one.

### Step 1 — Download pretraining spectra

Downloads MassBank EU (~140k records) and reassembles MoNA from committed parts.

```bash
python scripts/01_download_pretrain_data.py
```

Outputs: `data/pretraining/raw/combined_ei_spectra.json`

---

### Step 2 — Build HDF5 dataset

Bins spectra to 1000 m/z dims, applies √+L2 normalisation, writes train/val split.

```bash
python scripts/02_preprocess_pretrain_data.py \
    --raw-json data/pretraining/raw/combined_ei_spectra.json \
    --output-h5 data/pretraining/spectra.h5
```

---

### Step 3 — Pretrain the encoder (MSM)

Runs Masked Spectra Modelling on the HDF5 dataset. GPU recommended.

```bash
python scripts/03_pretrain.py --config configs/pretrain_msm.yaml
```

Checkpoint saved to `checkpoints/msm/best.pt`.

To override epochs or batch size:

```bash
python scripts/03_pretrain.py --config configs/pretrain_msm.yaml \
    --epochs 200 --batch-size 256
```

---

### Step 4 — Fine-tune and evaluate

Runs all three conditions (`from_scratch`, `msm_finetune`, `linear_probe_msm`) with 5-fold × 10-seed CV.

**Olive oil (both 1D reduction strategies):**

```bash
python scripts/04_finetune_evaluate.py \
    --task olive_oil \
    --config configs/finetune_oliveoil.yaml
```

**Hemp seed (group LOSO-CV):**

```bash
python scripts/04_finetune_evaluate.py \
    --task hemp_seed \
    --config configs/finetune_hemp.yaml
```

Results written to `results/{task}/transformer_results.json`.

---

### Step 5 — Run classical baselines

Evaluates PLS-DA, Random Forest, and SVM under the same CV protocol.

```bash
python scripts/05_run_baselines.py \
    --task olive_oil --config configs/finetune_oliveoil.yaml

python scripts/05_run_baselines.py \
    --task hemp_seed --config configs/finetune_hemp.yaml
```

---

### Step 6 — Attention visualisation

Generates attention rollout maps for representative samples.

```bash
python scripts/06_visualize_attention.py \
    --task hemp_seed \
    --checkpoint checkpoints/msm/best.pt \
    --config configs/finetune_hemp.yaml \
    --output-dir results/hemp_seed/attention_maps
```

---

## Key Results

### Olive oil — geographic origin (Spain / Italy / Greece), n = 157

| Method | Balanced Accuracy |
|---|---|
| SVM (APEX reduction) | **0.963 ± 0.051** |
| Random Forest (APEX) | 0.893 ± 0.067 |
| PLS-DA (SUM) | 0.787 ± 0.069 |
| Transformer (all conditions) | ~0.333 (chance) |

Domain mismatch between EI-MS pretraining and GC-IMS retention-time profiles prevents any transfer. Classical methods dominate.

### Hemp seed — Thai vs foreign origin, n = 12 (preliminary)

| Method | Balanced Accuracy |
|---|---|
| Transformer from scratch | **1.000 ± 0.000** |
| MSM linear probe | 0.967 ± 0.100 |
| MSM fine-tune | 0.792 ± 0.295 |
| Random Forest | 0.417 ± 0.144 |
| PLS-DA | 0.333 ± 0.333 |

*n = 12 — treat as a pilot result. The transformer benefits from the MSM linear probe regime; full fine-tuning on 9 training samples causes catastrophic forgetting.*

---

## Citation

If you use this code, please cite:

```bibtex
@misc{wood2026msformer,
  author = {Wood, Jesse},
  title  = {MSFormer: Self-Supervised Pretraining on GC-MS Reference Spectra for Food Authentication},
  year   = {2026},
  url    = {https://github.com/woodRock/msformer}
}
```

**Datasets:**

- Christmann & Weller (2022). *GC-IMS data on the discrimination between geographic origins of olive oils.* [doi:10.17632/fr9t5fkkvz.3](https://data.mendeley.com/datasets/fr9t5fkkvz/3)
- Sangkanu et al. (2026). *A GC/MS dataset for classification of Thai and foreign hemp seed extracts.* [doi:10.5281/zenodo.17505720](https://zenodo.org/records/17505720)
- MoNA — MassBank of North America. [mona.fiehnlab.ucdavis.edu](https://mona.fiehnlab.ucdavis.edu)
- MassBank EU. [massbank.eu](https://massbank.eu)
