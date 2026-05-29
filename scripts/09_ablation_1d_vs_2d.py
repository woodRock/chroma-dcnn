"""
Ablation study: does ChromaDCNN need the RT axis?

Two conditions, from_scratch only:

  2D      original [200, 1000] chromatogram (control — reuses existing protocol)
  1D-mz   degenerate input: each sample's sum spectrum tiled across all 200 RT
           bins, then per-bin √+L2 normalisation applied.  Every row is
           identical, so the RT axis carries zero discriminative information.
           The model sees the same dimensionality as 2D; only the RT structure
           is removed.

If ChromaDCNN's gain over classical methods comes from exploiting the joint
RT × m/z structure, 1D-mz should drop toward the PLS-DA-on-sum-spectrum
baseline (~0.43).  If 1D-mz stays close to 2D, the RT axis was not doing
meaningful work and the result needs honest discussion in the paper.

Usage:
  python scripts/09_ablation_1d_vs_2d.py
  python scripts/09_ablation_1d_vs_2d.py --device cpu

Outputs → results/fish_oil/
  ablation_1d_vs_2d_results.json
  ablation_1d_vs_2d_scores.csv
  ablation_1d_vs_2d_summary.md
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_dcnn.models.chroma_cnn import ChromaCNNConfig, ChromatogramCNN
from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_paths

DATA_DIR    = Path('data/fish_oil')
RESULTS_DIR = Path('results/fish_oil')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Degenerate transform ──────────────────────────────────────────────────────

def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-bin √ + L2 normalisation — matches original preprocessing."""
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def make_1d_mz(chroma: np.ndarray) -> np.ndarray:
    """
    Collapse RT axis: tile the sum spectrum across all 200 RT bins,
    then re-apply per-bin √+L2 normalisation.

    Every row of the output is identical — the model cannot extract any
    RT-axis information.  The input dimensionality is unchanged [200, 1000].
    """
    sum_spec   = chroma.sum(axis=0)                                # [1000]
    degenerate = np.tile(sum_spec, (chroma.shape[0], 1))           # [200, 1000]
    return _sqrt_l2(degenerate)


# ── Training ──────────────────────────────────────────────────────────────────

def _class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    w = np.zeros(n_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        w[c] = len(labels) / (len(classes) * cnt)
    return torch.tensor(w)


def train_fold(
    X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_te: torch.Tensor, y_te: np.ndarray,
    cfg: ChromaCNNConfig,
    device: torch.device,
    epochs: int = 200,
    patience: int = 20,
    lr: float = 1e-3,
) -> dict[str, float]:
    model = ChromatogramCNN(cfg).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch   = CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss(
        weight=_class_weights(y_tr.cpu().numpy(), cfg.num_classes).to(device)
    )
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)

    best_loss, best_state, stale = 1e9, None, 0
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        with torch.no_grad():
            vl = crit(model(X_tr), y_tr).item()
        if vl < best_loss:
            best_loss, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).cpu().numpy()

    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_te, preds)),
        'macro_f1':          float(f1_score(y_te, preds, average='macro',
                                            zero_division=0)),
    }


def run_condition(
    chromas: np.ndarray,
    y: np.ndarray,
    name: str,
    seeds: list[int],
    n_folds: int,
    cfg: ChromaCNNConfig,
    device: torch.device,
    degenerate: bool = False,
) -> dict[str, list[float]]:
    print(f'\n--- Condition: {name} ---')

    if degenerate:
        print('  Applying 1D-mz degeneracy transform …')
        chromas_in = np.stack([make_1d_mz(c) for c in chromas])
        print(f'  All rows identical: {np.allclose(chromas_in[0, 0], chromas_in[0, 99])}')
    else:
        chromas_in = chromas

    all_ba, all_f1 = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(chromas_in, y):
            X_tr = torch.tensor(chromas_in[tr_idx], dtype=torch.float32).to(device)
            y_tr = torch.tensor(y[tr_idx],           dtype=torch.long).to(device)
            X_te = torch.tensor(chromas_in[te_idx], dtype=torch.float32).to(device)
            y_te = y[te_idx]

            m = train_fold(X_tr, y_tr, X_te, y_te, cfg, device)
            all_ba.append(m['balanced_accuracy'])
            all_f1.append(m['macro_f1'])

    ba = np.array(all_ba)
    print(f'  balanced_accuracy: {ba.mean():.3f} ± {ba.std():.3f}  (n={len(ba)})')
    return {'balanced_accuracy': all_ba, 'macro_f1': all_f1}


# ── Statistics ────────────────────────────────────────────────────────────────

def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1.0 - 2.0 * u / (n1 * n2)


def compute_stats(ba_2d: list[float], ba_1d: list[float]) -> dict:
    a, b = np.array(ba_2d), np.array(ba_1d)
    u, p_raw = mannwhitneyu(a, b, alternative='two-sided')
    return {
        'ba_2d_mean':    float(a.mean()),
        'ba_2d_std':     float(a.std()),
        'ba_1d_mean':    float(b.mean()),
        'ba_1d_std':     float(b.std()),
        'delta':         float(b.mean() - a.mean()),
        'U':             float(u),
        'p_raw':         float(p_raw),
        'p_corrected':   float(p_raw),   # single comparison — no correction needed
        'effect_r':      abs(_rank_biserial(u, len(a), len(b))),
        'significant':   bool(p_raw < 0.05),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(seeds: list[int], n_folds: int, device_str: str | None) -> None:
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    npz_paths, y = load_fish_oil_chroma_paths(DATA_DIR)
    chromas = np.stack([np.load(p)['chroma'] for p in npz_paths])
    print(f'Loaded {len(chromas)} samples  shape={chromas.shape}')

    cfg = ChromaCNNConfig(
        mz_max=1000, cnn_channels=128, kernel_size=7, num_classes=4, dropout=0.3
    )

    results = {
        '2D':    run_condition(chromas, y, '2D (from_scratch)',
                               seeds, n_folds, cfg, device, degenerate=False),
        '1D-mz': run_condition(chromas, y, '1D-mz (degenerate)',
                               seeds, n_folds, cfg, device, degenerate=True),
    }

    # ── Save raw results ───────────────────────────────────────────────────
    out_json = RESULTS_DIR / 'ablation_1d_vs_2d_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Per-fold CSV ───────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / 'ablation_1d_vs_2d_scores.csv'
    n_runs  = len(results['2D']['balanced_accuracy'])
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold_run', '2D', '1D-mz'])
        for i in range(n_runs):
            writer.writerow([i + 1,
                             results['2D']['balanced_accuracy'][i],
                             results['1D-mz']['balanced_accuracy'][i]])

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = compute_stats(results['2D']['balanced_accuracy'],
                          results['1D-mz']['balanced_accuracy'])

    print('\n=== Result ===')
    print(f'  2D      : {stats["ba_2d_mean"]:.3f} ± {stats["ba_2d_std"]:.3f}')
    print(f'  1D-mz   : {stats["ba_1d_mean"]:.3f} ± {stats["ba_1d_std"]:.3f}')
    print(f'  Δ       : {stats["delta"]:+.3f}')
    print(f'  p       : {stats["p_corrected"]:.4e}')
    print(f'  |r|     : {stats["effect_r"]:.3f}')
    print(f'  sig     : {stats["significant"]}')

    # ── Plain-text table ───────────────────────────────────────────────────
    print('\n=== Table (plain text) ===')
    print(f'{"Condition":<22} {"BA mean":>8} {"± SD":>7} {"Δ vs 2D":>9} '
          f'{"p":>10} {"  |r|":>6} {"sig":>5}')
    print('─' * 68)
    print(f'{"2D (from_scratch)":<22} {stats["ba_2d_mean"]:>8.3f} '
          f'{stats["ba_2d_std"]:>7.3f} {"—":>9} {"—":>10} {"—":>6} {"—":>5}')
    print(f'{"1D-mz (degenerate)":<22} {stats["ba_1d_mean"]:>8.3f} '
          f'{stats["ba_1d_std"]:>7.3f} {stats["delta"]:>+9.3f} '
          f'{stats["p_corrected"]:>10.4e} {stats["effect_r"]:>6.3f} '
          f'{"✓" if stats["significant"] else "✗":>5}')

    # ── LaTeX ──────────────────────────────────────────────────────────────
    p_str = f'{stats["p_corrected"]:.2e}'
    sig   = r'\checkmark' if stats['significant'] else r'$\times$'
    print('\n=== LaTeX row ===')
    print(r'\begin{tabular}{lccccc}')
    print(r'\hline')
    print(r'Condition & BA (mean) & BA (SD) & $\Delta$ vs 2D & $p$ & Sig. \\')
    print(r'\hline')
    print(f'2D (from scratch) & {stats["ba_2d_mean"]:.3f} & '
          f'{stats["ba_2d_std"]:.3f} & --- & --- & --- \\\\')
    print(f'1D-mz (degenerate) & {stats["ba_1d_mean"]:.3f} & '
          f'{stats["ba_1d_std"]:.3f} & {stats["delta"]:+.3f} & '
          f'{p_str} & {sig} \\\\')
    print(r'\hline')
    print(r'\end{tabular}')

    # ── Markdown summary ───────────────────────────────────────────────────
    pred_held   = stats['significant'] and stats['ba_1d_mean'] < stats['ba_2d_mean']
    pred_str    = 'held' if pred_held else 'did NOT hold'
    sig_str     = 'Yes'  if stats['significant'] else 'No'
    p_str2      = f"{stats['p_corrected']:.2e}"
    r_str2      = f"{stats['effect_r']:.3f}"
    interp_str  = (
        f'The RT axis carries significant discriminative information. '
        f'Removing it (1D-mz) causes a statistically significant drop '
        f'(p={p_str2}, |r|={r_str2}). This causally supports the claim '
        f'that ChromaDCNN exploits the joint RT x m/z structure rather '
        f'than spectral content alone.'
        if pred_held else
        'The RT axis did not provide a statistically significant benefit '
        'over the degenerate 1D-mz condition. This is an unexpected result '
        'that warrants honest discussion in the paper: the spectral content '
        'alone (tiled sum spectrum) may account for most of the performance gain.'
    )
    framing_str = (
        'Include as positive evidence for the 2D-structure hypothesis. '
        'The drop from 2D to 1D-mz confirms that the RT dimension is '
        'load-bearing.'
        if pred_held else
        'Report honestly. Consider whether the per-bin spectral projection '
        'is doing the heavy lifting and the dilated CNN over RT is providing '
        'marginal benefit. The result does not invalidate the paper but '
        'shifts the emphasis.'
    )
    summary = f"""## Ablation: Does ChromaDCNN need the RT axis?

**Protocol:** 10 seeds × 5-fold stratified CV (n=50), `from_scratch`,
identical hyperparameters to main experiment.

**1D-mz construction:** sum spectrum tiled across 200 RT bins →
per-bin √+L2 normalisation. Every RT bin is identical; the model
cannot extract RT-axis information.

| Condition | BA (mean ± SD) | Δ vs 2D | p | \\|r\\| | Sig. |
|-----------|---------------|---------|---|------|------|
| 2D (from scratch) | {stats['ba_2d_mean']:.3f} ± {stats['ba_2d_std']:.3f} | — | — | — | — |
| 1D-mz (degenerate) | {stats['ba_1d_mean']:.3f} ± {stats['ba_1d_std']:.3f} | {stats['delta']:+.3f} | {p_str2} | {r_str2} | {sig_str} |

**Prediction (from saliency analysis):** 1D-mz should drop toward
PLS-DA-on-sum-spectrum (~0.43). Prediction {pred_str}.

**Interpretation:**
{interp_str}

**Paper framing:**
{framing_str}
"""
    out_md = RESULTS_DIR / 'ablation_1d_vs_2d_summary.md'
    out_md.write_text(summary)

    print(f'\nOutputs saved to {RESULTS_DIR}/')
    print(f'  {out_json.name}')
    print(f'  {out_csv.name}')
    print(f'  {out_md.name}')
    print('\n' + summary)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds',  nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--folds',  type=int,  default=5)
    ap.add_argument('--device', type=str,  default=None)
    args = ap.parse_args()
    main(args.seeds, args.folds, args.device)
