# MSFormer — NZ Fish Species Identification by GC-MS

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![When the peaks already know the fish](meme.png)

Species identification of four commercially important New Zealand fish (Snapper, Gurnard, Tarakihi, Blue Cod) from GC-MS lipid profiles using a 1D dilated CNN with self-supervised pretraining.

---

## Dataset

103 GC-MS samples from six biological individuals across four species and six body parts (frame, gonad, head, liver, rest-of-guts, skin).

| Label | Species | n |
|---|---|---|
| 0 | SNA — Snapper (*Pagrus auratus*) | 15 |
| 1 | GUR — Gurnard (*Chelidonichthys kumu*) | 14 |
| 2 | TAR — Tarakihi (*Nemadactylus macropterus*) | 17 |
| 3 | BCO — Blue Cod (*Parapercis colias*) | 57 |

Raw data: 2D GC-MS chromatograms (4800 RT scans × 344 m/z channels, 50–543 Da).  
Preprocessing: uniform 200-bin RT resampling, sqrt + L2 normalisation per bin.

Preprocessed arrays and chromatograms are committed to this repo — no re-processing needed to run training.

---

## Model

**ChromatogramCNN** — a 1D dilated CNN operating on the RT axis of the 2D chromatogram.

```
[B, 200, 1000]                    per-sample 2D chromatogram
    ↓  per-bin spectral embedding
[B, 200, 128]                     RT sequence of spectral features
    ↓  three dilated ResBlocks (kernel=7, dilation=1/2/4)
[B, 128, 200]                     temporal feature map
    ↓  global max-pool ∥ soft attention-pool
[B, 256]                          dual-pooled representation
    ↓  linear head
[B, 4]                            class logits
```

Two conditions are evaluated:

| Condition | Per-bin encoder | Pretrained? |
|---|---|---|
| `from_scratch` | `Linear(1000 → 128)` | No |
| `chroma_pretrain` | `Linear(1000 → 128)` | Yes — next-frame prediction on synthetic GC-MS (MoNA/MassBank) |

---

## Repository Structure

```
msformer/
├── configs/
│   ├── pretrain.yaml                  # ChromaCNN next-frame pretraining
│   ├── finetune.yaml                  # ChromatogramCNN fine-tuning
│   └── finetune_fish_oil.yaml         # Baselines config
├── data/
│   └── fish_oil/
│       ├── X.npy                      # [103, 1000] sum spectra
│       ├── y.npy                      # [103] labels
│       ├── groups.txt                 # biological group per sample
│       ├── sample_ids.txt             # sample name per row
│       ├── chroma/                    # [103 × 200 × 1000] RT-binned chromatograms
│       └── scans/                     # [103 × K × 1000] per-scan spectra
├── scripts/
│   ├── 01_download_pretrain_data.py   # download MoNA + MassBank
│   ├── 02_preprocess_pretrain_data.py # build data/pretraining/spectra.h5
│   ├── 04_finetune_evaluate.py        # full CV evaluation (CNN conditions)
│   ├── 05_run_baselines.py            # PLS-DA, RF, SVM, MLP baselines
│   ├── 07_smoke_test.py               # single seed × fold sanity check
│   ├── 08_pretrain_chroma.py          # pretrain ChromaCNN on synthetic GC-MS
│   ├── preprocess_fish_oil.py         # raw CSV → X.npy, y.npy
│   ├── preprocess_fish_oil_chroma.py  # raw CSV → chroma/*.npz
│   └── preprocess_fish_oil_scans.py   # raw CSV → scans/*.npz
└── src/msformer/
    ├── data/                          # datasets, download, preprocessing
    ├── models/                        # ChromatogramCNN, ChromaNextFramePredictor
    ├── training/                      # pretraining and fine-tuning loops
    ├── evaluation/                    # baselines, stats
    └── downstream/                    # fish_oil data loaders
```

---

## Installation

```bash
git clone https://github.com/woodRock/msformer.git
cd msformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Running the Experiments

The preprocessed fish oil data is committed — clone the repo and start from step 3 or 4.

### Step 1 — Download pretraining spectra (if needed)

```bash
python scripts/01_download_pretrain_data.py
```

Outputs `data/pretraining/raw/combined_ei_spectra.json` from MoNA + MassBank EU.

### Step 2 — Build HDF5 dataset (if needed)

```bash
python scripts/02_preprocess_pretrain_data.py
```

Outputs `data/pretraining/spectra.h5` (~9,500 EI-MS spectra, sqrt + L2 normalised).

### Step 3 — Pretrain ChromaCNN (next-frame prediction)

Uses synthetic GC-MS chromatograms generated on-the-fly from `spectra.h5` — no fish oil data used, so there is no information leakage into fine-tuning CV folds.

```bash
python scripts/08_pretrain_chroma.py
```

Checkpoint saved to `checkpoints/chroma_pretrain/best.pt`.

### Step 4 — Full CV evaluation (ChromatogramCNN)

10 seeds × 5-fold stratified CV (~10 min on GPU).

```bash
python scripts/04_finetune_evaluate.py
```

Results saved to `results/fish_oil/chroma_results.json`.

### Step 5 — Classical baselines

```bash
python scripts/05_run_baselines.py
```

Evaluates PLS-DA, RF, SVM, MLP on three feature representations:
- `sum` — per-m/z sum spectrum
- `max_proj` — per-m/z maximum across RT (peak heights)
- `chroma_pca` — full 2D chromatogram, 50 PCs within each CV fold

Results saved to `results/fish_oil/baseline_gcms_{representation}_results.json`.

### Step 6 — Smoke test (optional)

Quick single-seed × single-fold check before committing to the full run:

```bash
python scripts/07_smoke_test.py --seed 0 --fold 0 --epochs 100
```

---

## Results

10 seeds × 5-fold stratified CV (n = 50 per method). Statistical comparisons use Mann-Whitney U with Bonferroni correction (α = 0.05); effect size is rank-biserial correlation r vs `chroma_pretrain`.

| Rank | Method | Representation | Balanced Accuracy | \|r\| vs best | p (Bonferroni) | p < 0.05 |
|:----:|--------|----------------|:-----------------:|:-------------:|:--------------:|:--------:|
| 1 | **ChromatogramCNN** (`chroma_pretrain`) | 2D chromatogram | **0.951 ± 0.065** | — | — | — |
| 2 | ChromatogramCNN (`from_scratch`) | 2D chromatogram | 0.759 ± 0.138 | 0.796 | < 0.001 | Yes |
| 3 | PLS-DA | chroma_pca | 0.703 ± 0.091 | 0.957 | < 0.001 | Yes |
| 4 | RF | sum | 0.697 ± 0.113 | 0.945 | < 0.001 | Yes |
| 5 | SVM | chroma_pca | 0.694 ± 0.108 | 0.956 | < 0.001 | Yes |
| 6 | SVM | max_proj | 0.532 ± 0.120 | 0.998 | < 0.001 | Yes |
| 7 | RF | chroma_pca | 0.529 ± 0.116 | 1.000 | < 0.001 | Yes |
| 8 | SVM | sum | 0.493 ± 0.083 | 1.000 | < 0.001 | Yes |
| 9 | PLS-DA | max_proj | 0.469 ± 0.107 | 1.000 | < 0.001 | Yes |
| 10 | PLS-DA | sum | 0.429 ± 0.114 | 1.000 | < 0.001 | Yes |
| 11 | MLP | max_proj | 0.395 ± 0.123 | 1.000 | < 0.001 | Yes |
| 12 | RF | max_proj | 0.382 ± 0.094 | 1.000 | < 0.001 | Yes |
| 13 | MLP | chroma_pca | 0.377 ± 0.159 | 1.000 | < 0.001 | Yes |
| 14 | MLP | sum | 0.369 ± 0.125 | 1.000 | < 0.001 | Yes |

`chroma_pretrain` significantly outperforms every other method (all p < 0.001 after Bonferroni correction). Among the classical baselines, the `chroma_pca` representation (full 2D chromatogram → 50 within-fold PCs) consistently yields the strongest PLS-DA and SVM results.

---

## Citation

```bibtex
@misc{wood2026msformer,
  author = {Wood, Jesse},
  title  = {MSFormer: GC-MS Fish Species Identification with Self-Supervised Pretraining},
  year   = {2026},
  url    = {https://github.com/woodRock/msformer}
}
```

**Pretraining data:**
- MoNA — MassBank of North America: [mona.fiehnlab.ucdavis.edu](https://mona.fiehnlab.ucdavis.edu)
- MassBank EU: [massbank.eu](https://massbank.eu)
