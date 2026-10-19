"""
Gradient saliency for pretrained ChromaDCNN on the fish oil dataset.

Protocol (matches the main experiment, scripts/04_finetune_evaluate.py):
  seed 0, 5-fold stratified CV.  Each fold trains a chroma_pretrain model
  through ChromaFinetuner's training loop (fit / val split, early stopping on
  val, class-weighted loss).  Saliency |∂logit_c / ∂X| is computed for the
  fold's held-out test samples only, with c the true class, so every sample
  contributes exactly once and never from a model that trained on it.

Usage:
  python scripts/11_saliency.py
  python scripts/11_saliency.py --device cuda:0

Outputs → figs/
  saliency-heatmaps.png   per-species RT × m/z heatmaps
  saliency-1d.png         RT and m/z marginal profiles
  saliency-3d.png         saliency overlaid on a real chromatogram surface
and results/fish_oil/saliency_mean.npz (per-species mean saliency, [4, 200, 1000]) and
results/fish_oil/saliency_summary.json (what the paper reports: fold accuracy, each RT band's share
of every species' RT saliency, and how similar the species' RT and m/z profiles are).

  python scripts/11_saliency.py --summary-only   # recompute the summary from saliency_mean.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_paths
from chroma_dcnn.training.finetune_chroma import (
    ChromaFinetuner,
    _build_model,
    _inner_val_split,
    _train_fold,
)

CONFIG_PATH = Path('configs/finetune.yaml')
FIGS_DIR    = Path('figs')
RESULTS_DIR = Path('results/fish_oil')

SPECIES = {0: 'SNA (Snapper)', 1: 'GUR (Gurnard)', 2: 'TAR (Tarakihi)', 3: 'BCO (Blue Cod)'}
SPECIES_LABEL = {0: 'Snapper (SNA)', 1: 'Gurnard (GUR)',
                 2: 'Tarakihi (TAR)', 3: 'Blue Cod (BCO)'}
COLOURS = {0: '#2196f3', 1: '#ff9800', 2: '#4caf50', 3: '#e91e63'}


# ── Saliency ──────────────────────────────────────────────────────────────────

def compute_saliency(
    seed: int, device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    paths, y = load_fish_oil_chroma_paths(cfg['data']['data_dir'])
    finetuner = ChromaFinetuner(cfg, paths, y, device=device)
    dev = finetuner.device
    chromas = np.stack([np.load(p)['chroma'] for p in paths]).astype(np.float32)

    n_folds = cfg['task'].get('cv_folds', 5)
    val_fraction = cfg['finetuning'].get('val_fraction', 0.2)
    lr = cfg['finetuning'].get('lr', 1e-3)

    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_sample = np.zeros_like(chromas)
    fold_ba = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, y)):
        fit_rel, val_rel = _inner_val_split(y[train_idx], val_fraction, seed=seed * 1000 + fold)
        fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]

        model = _build_model(finetuner.model_config, 'chroma_pretrain',
                             finetuner.chroma_pretrain_ckpt).to(dev)
        metrics = _train_fold(
            model,
            [paths[i] for i in fit_idx], [paths[i] for i in val_idx],
            [paths[i] for i in test_idx],
            y[fit_idx], y[val_idx], y[test_idx],
            cfg, dev, lr=lr,
        )
        fold_ba.append(metrics['balanced_accuracy'])
        print(f'  fold {fold}: test BA = {metrics["balanced_accuracy"]:.3f}')

        model.eval()
        for i in test_idx:
            x = torch.tensor(chromas[i:i + 1], device=dev, requires_grad=True)
            model(x)[0, int(y[i])].backward()
            per_sample[i] = x.grad.detach().abs().squeeze(0).cpu().numpy()

    sal_mean = np.stack([per_sample[y == c].mean(axis=0) for c in range(4)])
    return sal_mean, chromas, y, fold_ba


# ── Figures (styling as in notebooks/04_saliency.ipynb) ───────────────────────

def plot_heatmaps(sal_mean: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for c, ax in enumerate(axes.flatten()):
        sal  = sal_mean[c][:, :600]
        vmax = np.percentile(sal, 99)
        im = ax.imshow(sal, aspect='auto', origin='lower', cmap='hot',
                       vmin=0, vmax=vmax, extent=[0, 600, 5, 45])
        ax.set_title(SPECIES[c], color=COLOURS[c], fontweight='bold')
        ax.set_xlabel('m/z bin (0–600)')
        ax.set_ylabel('RT (min)')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='|∂score/∂input|')
    plt.suptitle('Saliency maps: which RT windows and m/z channels drive each species prediction?',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_marginals(sal_mean: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    rt_vals = np.linspace(5, 45, 200)
    for c in range(4):
        rt_sal = sal_mean[c].mean(axis=1)
        axes[0].plot(rt_vals, rt_sal / rt_sal.max(),
                     color=COLOURS[c], label=SPECIES[c], linewidth=1.2)
    axes[0].set_xlabel('Retention time (min)')
    axes[0].set_ylabel('Normalised saliency')
    axes[0].set_title('RT saliency profile')
    axes[0].legend(frameon=False, ncol=2)
    sns.despine(ax=axes[0])

    for c in range(4):
        mz_sal = sal_mean[c].mean(axis=0)[:600]
        axes[1].plot(np.arange(600), mz_sal / mz_sal.max(),
                     color=COLOURS[c], label=SPECIES[c], linewidth=1.0, alpha=0.85)
    axes[1].set_xlabel('m/z bin')
    axes[1].set_ylabel('Normalised saliency')
    axes[1].set_title('m/z saliency profile')
    axes[1].legend(frameon=False, ncol=2)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_3d(sal_mean: np.ndarray, chromas: np.ndarray, y: np.ndarray, out: Path) -> None:
    MZ_LO, MZ_HI = 50, 450
    RT_LO, RT_HI = 5.0, 45.0
    S_RT, S_MZ   = 2, 2

    fig, axes_grid = plt.subplots(2, 2, figsize=(11, 10), subplot_kw={'projection': '3d'})
    fig.subplots_adjust(bottom=0.11, hspace=0.25, wspace=0.05, top=0.93)

    for c in range(4):
        ax = axes_grid[c // 2, c % 2]

        # Surface: first sample of the species (as in the original figure)
        idx         = int(np.where(y == c)[0][0])
        chroma_real = chromas[idx, :, MZ_LO:MZ_HI].astype(np.float32)

        sal = gaussian_filter(sal_mean[c][:, MZ_LO:MZ_HI].astype(np.float32), sigma=(1.5, 2.0))

        # Intensity: top ~20% of non-zero peaks; sqrt scale
        Z   = chroma_real.copy()
        nz  = Z[Z > 0]
        thr = np.percentile(nz, 80) if len(nz) else 0.05
        Z[Z < thr] = 0.0
        Z = np.sqrt(Z)

        rt_s  = np.linspace(RT_LO, RT_HI, Z.shape[0])[::S_RT]
        mz_s  = np.linspace(MZ_LO, MZ_HI, Z.shape[1])[::S_MZ]
        Z_s   = Z[::S_RT, ::S_MZ]
        sal_s = sal[::S_RT, ::S_MZ]
        RT_grid, MZ_grid = np.meshgrid(rt_s, mz_s)

        # Blues: transparent where saliency is low, deep blue where high
        sal_norm             = (sal_s - sal_s.min()) / (sal_s.max() - sal_s.min() + 1e-8)
        face_colours         = cm.Blues(sal_norm.T)
        face_colours[..., 3] = sal_norm.T ** 0.6

        ax.plot_surface(RT_grid, MZ_grid, Z_s.T, color='#d0d8e8', alpha=0.25,
                        rstride=1, cstride=1, shade=False, linewidth=0)
        ax.plot_surface(RT_grid, MZ_grid, Z_s.T, facecolors=face_colours,
                        rstride=1, cstride=1, shade=False, linewidth=0)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor('none')
        ax.zaxis.line.set_color((1, 1, 1, 0))
        ax.set_zticks([])
        ax.set_xlim(RT_LO, RT_HI)
        ax.set_ylim(MZ_LO, MZ_HI)
        ax.view_init(elev=42, azim=-60)
        ax.set_xticks([10, 20, 30, 40])
        ax.set_yticks([100, 200, 300, 400])
        ax.tick_params(labelsize=7)
        ax.set_xlabel('RT, min', fontsize=9, labelpad=10)
        ax.set_ylabel('m/z',     fontsize=9, labelpad=10)
        ax.set_title(SPECIES_LABEL[c], fontweight='bold', color=COLOURS[c], pad=4, fontsize=10)

    cbar_ax = fig.add_axes([0.18, 0.045, 0.64, 0.020])
    sm = cm.ScalarMappable(cmap='Blues', norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0\n(low)', '0.25', '0.5', '0.75',
                         '1.0\n(high)'])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Normalised saliency', fontsize=9, labelpad=4)
    fig.suptitle('Gradient saliency overlaid on GC-MS chromatograms',
                 fontsize=12, y=0.97)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


RT_MIN = np.linspace(5, 45, 200)  # retention time (min) of each RT bin, as in the figures
BANDS = [(18, 22), (24, 28), (33, 37), (40, 44)]  # the RT bands the paper discusses (min)
SPECIES_KEY = {0: 'snapper', 1: 'gurnard', 2: 'tarakihi', 3: 'blue-cod'}


def saliency_summary(sal_mean: np.ndarray, fold_ba: np.ndarray) -> dict:
    """The numbers the paper quotes about saliency.

    A species' RT profile is its saliency averaged over m/z, on RT_MIN; a band's share is the
    fraction of that profile inside the band. Profile similarity is the Pearson r between every
    pair of species, for the RT profiles and for the m/z profiles (saliency averaged over RT).
    """
    rt = sal_mean.mean(axis=2)
    mz = sal_mean.mean(axis=1)
    bands = {}
    for lo, hi in BANDS:
        inside = (RT_MIN >= lo) & (RT_MIN <= hi)
        share = {SPECIES_KEY[c]: float(rt[c][inside].sum() / rt[c].sum()) for c in range(len(rt))}
        bands[f'{lo}-{hi}min'] = share | {'min': min(share.values()), 'max': max(share.values())}
    pairs = [(a, b) for a in range(len(rt)) for b in range(a + 1, len(rt))]

    def corr(profiles: np.ndarray) -> dict:
        r = [float(np.corrcoef(profiles[a], profiles[b])[0, 1]) for a, b in pairs]
        return {'min': min(r), 'max': max(r), 'n': len(r)}

    return {
        'fold_balanced_accuracy': {'mean': float(np.mean(fold_ba)), 'n': int(len(fold_ba))},
        'rt_profile_corr': corr(rt),
        'mz_profile_corr': corr(mz),
        'band_share': bands,
    }


def write_summary(sal_mean: np.ndarray, fold_ba: np.ndarray) -> None:
    out = RESULTS_DIR / 'saliency_summary.json'
    out.write_text(json.dumps(saliency_summary(sal_mean, fold_ba), indent=2) + '\n')
    print(f'Summary saved → {out}')


def main(seed: int, device: str | None) -> None:
    sal_mean, chromas, y, fold_ba = compute_saliency(seed, device)
    print(f'Mean test BA over folds: {np.mean(fold_ba):.3f}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS_DIR / 'saliency_mean.npz', saliency=sal_mean,
                        fold_balanced_accuracy=np.array(fold_ba))
    write_summary(sal_mean, np.array(fold_ba))

    FIGS_DIR.mkdir(exist_ok=True)
    plot_heatmaps(sal_mean, FIGS_DIR / 'saliency-heatmaps.png')
    plot_marginals(sal_mean, FIGS_DIR / 'saliency-1d.png')
    plot_3d(sal_mean, chromas, y, FIGS_DIR / 'saliency-3d.png')
    print(f'Figures saved → {FIGS_DIR}/saliency-{{heatmaps,1d,3d}}.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Saliency figures for pretrained ChromaDCNN')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default=None, help='Force device: cuda, mps, cpu')
    ap.add_argument('--summary-only', action='store_true',
                    help='recompute saliency_summary.json from saliency_mean.npz (no training)')
    args = ap.parse_args()
    if args.summary_only:
        z = np.load(RESULTS_DIR / 'saliency_mean.npz')
        write_summary(z['saliency'], z['fold_balanced_accuracy'])
    else:
        main(args.seed, args.device)
