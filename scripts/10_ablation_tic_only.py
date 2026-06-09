"""
Ablation study: does ChromaDCNN need the m/z dimension?

Two conditions, from_scratch only:

  2D      original [200, 1000] chromatogram (control — reuses existing protocol)
  TIC     Total Ion Chromatogram [200, 1]: each RT bin is collapsed to a single
          scalar by summing across all m/z channels.  The m/z fingerprint at
          every peak is discarded; only the total ion count at each RT bin is
          retained.  The model architecture is unchanged — spec_proj becomes
          Linear(1 → cnn_channels) — so the RT-axis CNN still operates on a
          200-element sequence.

Companion to 09_ablation_1d_vs_2d.py, which ablates the RT axis.
Together they isolate the contribution of each dimension:

  09 (1D-mz)  removes RT structure — keeps m/z information, identical per bin
  10 (TIC)    removes m/z structure — keeps RT information, single value per bin

Note: chroma_pretrain is not evaluated here.  The pretrained spec_proj is
Linear(1000 → 128); TIC-only requires Linear(1 → 128) — incompatible weights.

Usage:
  python scripts/10_ablation_tic_only.py
  python scripts/10_ablation_tic_only.py --device cpu

Outputs → results/fish_oil/
  ablation_tic_only_results.json
  ablation_tic_only_scores.csv
  ablation_tic_only_summary.md
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_dcnn.models.chroma_cnn import ChromaCNNConfig, ChromatogramCNN
from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_paths

DATA_DIR    = Path('data/fish_oil')
RESULTS_DIR = Path('results/fish_oil')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── TIC transform ─────────────────────────────────────────────────────────────

def make_tic(chroma: np.ndarray) -> np.ndarray:
    """
    Collapse m/z axis: sum across all 1000 m/z bins → [200, 1].

    The RT axis is preserved; each bin is reduced to its total ion count.
    The model receives [200, 1] — all compound-identity information in the
    m/z fingerprints is discarded.
    """
    return chroma.sum(axis=1, keepdims=True).astype(np.float32)   # [200, 1]


# ── Training (identical to 09_ablation_1d_vs_2d.py) ──────────────────────────

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
    tic_only: bool = False,
) -> dict[str, list[float]]:
    print(f'\n--- Condition: {name} ---')

    if tic_only:
        print('  Applying TIC-only transform (summing m/z axis) …')
        chromas_in = np.stack([make_tic(c) for c in chromas])
        print(f'  Input shape: {chromas_in.shape}  (was {chromas.shape})')
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


# ── Statistics (identical to 09_ablation_1d_vs_2d.py) ────────────────────────

def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1.0 - 2.0 * u / (n1 * n2)


def compute_stats(ba_2d: list[float], ba_tic: list[float]) -> dict:
    a, b = np.array(ba_2d), np.array(ba_tic)
    u, p_raw = mannwhitneyu(a, b, alternative='two-sided')
    return {
        'ba_2d_mean':  float(a.mean()),
        'ba_2d_std':   float(a.std()),
        'ba_tic_mean': float(b.mean()),
        'ba_tic_std':  float(b.std()),
        'delta':       float(b.mean() - a.mean()),
        'U':           float(u),
        'p_raw':       float(p_raw),
        'p_corrected': float(p_raw),
        'effect_r':    abs(_rank_biserial(u, len(a), len(b))),
        'significant': bool(p_raw < 0.05),
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

    cfg_2d  = ChromaCNNConfig(mz_max=1000, cnn_channels=128, kernel_size=7,
                               num_classes=4, dropout=0.3)
    cfg_tic = ChromaCNNConfig(mz_max=1,    cnn_channels=128, kernel_size=7,
                               num_classes=4, dropout=0.3)

    results = {
        '2D':  run_condition(chromas, y, '2D (from_scratch)',
                             seeds, n_folds, cfg_2d,  device, tic_only=False),
        'TIC': run_condition(chromas, y, 'TIC only (from_scratch)',
                             seeds, n_folds, cfg_tic, device, tic_only=True),
    }

    # ── Save raw results ───────────────────────────────────────────────────
    out_json = RESULTS_DIR / 'ablation_tic_only_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Per-fold CSV ───────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / 'ablation_tic_only_scores.csv'
    n_runs  = len(results['2D']['balanced_accuracy'])
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold_run', '2D', 'TIC'])
        for i in range(n_runs):
            writer.writerow([i + 1,
                             results['2D']['balanced_accuracy'][i],
                             results['TIC']['balanced_accuracy'][i]])

    # ── Stats ──────────────────────────────────────────────────────────────
    stats = compute_stats(results['2D']['balanced_accuracy'],
                          results['TIC']['balanced_accuracy'])

    print('\n=== Result ===')
    print(f'  2D  : {stats["ba_2d_mean"]:.3f} ± {stats["ba_2d_std"]:.3f}')
    print(f'  TIC : {stats["ba_tic_mean"]:.3f} ± {stats["ba_tic_std"]:.3f}')
    print(f'  Δ   : {stats["delta"]:+.3f}')
    print(f'  p   : {stats["p_corrected"]:.4e}')
    print(f'  |r| : {stats["effect_r"]:.3f}')
    print(f'  sig : {stats["significant"]}')

    # ── Plain-text table ───────────────────────────────────────────────────
    print('\n=== Table (plain text) ===')
    print(f'{"Condition":<24} {"BA mean":>8} {"± SD":>7} {"Δ vs 2D":>9} '
          f'{"p":>10} {"  |r|":>6} {"sig":>5}')
    print('─' * 70)
    print(f'{"2D (from_scratch)":<24} {stats["ba_2d_mean"]:>8.3f} '
          f'{stats["ba_2d_std"]:>7.3f} {"—":>9} {"—":>10} {"—":>6} {"—":>5}')
    print(f'{"TIC only (from_scratch)":<24} {stats["ba_tic_mean"]:>8.3f} '
          f'{stats["ba_tic_std"]:>7.3f} {stats["delta"]:>+9.3f} '
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
    print(f'TIC only (from scratch) & {stats["ba_tic_mean"]:.3f} & '
          f'{stats["ba_tic_std"]:.3f} & {stats["delta"]:+.3f} & '
          f'{p_str} & {sig} \\\\')
    print(r'\hline')
    print(r'\end{tabular}')

    # ── Markdown summary ───────────────────────────────────────────────────
    pred_held  = stats['significant'] and stats['ba_tic_mean'] < stats['ba_2d_mean']
    pred_str   = 'held' if pred_held else 'did NOT hold'
    sig_str    = 'Yes'  if stats['significant'] else 'No'
    p_str2     = f"{stats['p_corrected']:.2e}"
    r_str2     = f"{stats['effect_r']:.3f}"
    interp_str = (
        f'The m/z dimension carries significant discriminative information. '
        f'Collapsing to TIC (removing m/z fingerprints) causes a statistically '
        f'significant drop (p={p_str2}, |r|={r_str2}). The compound-identity '
        f'information encoded in the m/z spectra is load-bearing for classification.'
        if pred_held else
        'Collapsing to TIC did not cause a statistically significant drop. '
        'The RT elution profile alone may account for most of the classification '
        'signal, with the m/z dimension providing marginal additional benefit. '
        'This warrants honest discussion alongside the 09 (1D-mz) ablation.'
    )
    framing_str = (
        'Include as positive evidence that the m/z dimension is necessary. '
        'Together with 09 (1D-mz), this establishes that both the RT and m/z '
        'axes contribute independently to ChromaDCNN\'s performance.'
        if pred_held else
        'Report honestly alongside 09_ablation_1d_vs_2d. Consider whether the '
        'RT elution profile alone drives classification and whether the m/z '
        'projection is serving primarily as a noise-reduction step.'
    )
    summary = f"""## Ablation: Does ChromaDCNN need the m/z dimension?

**Protocol:** {len(seeds)} seeds × {n_folds}-fold stratified CV (n={len(seeds)*n_folds}), `from_scratch`,
identical hyperparameters to main experiment.

**TIC construction:** sum across all 1000 m/z channels → [200, 1] per sample.
The m/z fingerprint at every peak is discarded; only the total ion count
at each RT bin is retained. `spec_proj` becomes Linear(1 → {cfg_tic.cnn_channels}).

**Note:** `chroma_pretrain` is not evaluated — the pretrained `spec_proj` is
Linear(1000 → {cfg_2d.cnn_channels}), incompatible with the TIC-only mz_max=1 input.

| Condition | BA (mean ± SD) | Δ vs 2D | p | \\|r\\| | Sig. |
|-----------|---------------|---------|---|------|------|
| 2D (from scratch) | {stats['ba_2d_mean']:.3f} ± {stats['ba_2d_std']:.3f} | — | — | — | — |
| TIC only (from scratch) | {stats['ba_tic_mean']:.3f} ± {stats['ba_tic_std']:.3f} | {stats['delta']:+.3f} | {p_str2} | {r_str2} | {sig_str} |

**Prediction:** TIC-only should drop toward chance (0.25) or near the RF-on-TIC
baseline if the m/z axis is necessary. Prediction {pred_str}.

**Interpretation:**
{interp_str}

**Paper framing:**
{framing_str}

**Companion ablation:** see `09_ablation_1d_vs_2d.py` (ablates the RT axis).
"""
    out_md = RESULTS_DIR / 'ablation_tic_only_summary.md'
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
