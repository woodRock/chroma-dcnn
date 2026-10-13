"""
Export the fish oil results to docs/data.js for the project website (GitHub Pages).

Reads only committed outputs (results/fish_oil/*, data/fish_oil/*); no training.

Usage:
  python scripts/13_build_site_data.py

Output → docs/data.js  (window.SITE_DATA = {...})
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS = Path('results/fish_oil')
DATA    = Path('data/fish_oil')
OUT     = Path('docs/data.js')

SPECIES = ['SNA', 'GUR', 'TAR', 'BCO']
RT_LO, RT_HI = 5.0, 45.0
MZ_LO, MZ_HI = 50, 544            # non-zero m/z channels in the fish oil data
GRID_MZ = (50, 450, 2)            # 3D/heatmap grid: m/z 50–449, step 2
GRID_RT = 2                       # 200 RT bins → 100


def r3(a) -> list:
    return np.round(np.asarray(a, dtype=float), 4).tolist()


def quantise(m: np.ndarray) -> dict:
    """Matrix → ints 0–255 plus its max, to keep data.js small."""
    mx = float(m.max()) or 1.0
    return {'max': mx, 'q': np.rint(m / mx * 255).astype(int).tolist()}


def coarsen(m: np.ndarray) -> np.ndarray:
    """[200, 1000] → [100, 200]: max over 2 RT bins × 2 m/z channels in m/z 50–449."""
    lo, hi, st = GRID_MZ
    m = m[:, lo:hi]
    m = m.reshape(m.shape[0] // GRID_RT, GRID_RT, m.shape[1] // st, st)
    return m.max(axis=(1, 3))


def methods() -> list[dict]:
    chroma = json.loads((RESULTS / 'chroma_results.json').read_text())
    scores = {('ChromaDCNN (chroma_pretrain)', '2D chromatogram'): chroma['chroma_pretrain'],
              ('ChromaDCNN (from_scratch)', '2D chromatogram'): chroma['from_scratch']}
    names = {'plsda': 'PLS-DA', 'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'mlp': 'MLP'}
    reps = {'sum': 'sum spectrum', 'max_proj': 'max-projection', 'chroma_pca': 'chroma-PCA'}
    for rep, rep_name in reps.items():
        res = json.loads((RESULTS / f'baseline_gcms_{rep}_results.json').read_text())
        for key, name in names.items():
            scores[(name, rep_name)] = res[key]

    table = []
    with open(RESULTS / 'main_table.csv') as f:
        for row in csv.DictReader(f):
            s = scores[(row['method'], row['representation'])]
            table.append({
                'method': row['method'], 'representation': row['representation'],
                'ba': r3(s['balanced_accuracy']), 'f1': r3(s['macro_f1']),
                'mean': float(row['ba_mean']), 'sd': float(row['ba_std']),
                'abs_r': float(row['abs_r_vs_best']) if row['abs_r_vs_best'] else None,
                'p': float(row['p_bonferroni']) if row['p_bonferroni'] else None,
            })
    return table


def main() -> None:
    def lines(name: str) -> list[str]:
        return [s.strip() for s in (DATA / name).read_text().splitlines() if s.strip()]

    sample_ids = lines('sample_ids.txt')
    y = np.load(DATA / 'y.npy')
    parts = [s.split(' - ', 1)[1].split()[1] for s in sample_ids]
    parts = ['Rest of guts' if p in ('Rest', 'ROG') else p for p in parts]
    groups = lines('groups.txt')
    composition = Counter((SPECIES[int(c)], p) for c, p in zip(y, parts))
    individuals = Counter((SPECIES[int(c)], g) for c, g in zip(y, groups))

    paths = sorted((DATA / 'chroma').glob('*.npz'))
    sal = np.load(RESULTS / 'saliency_mean.npz')
    S, fold_ba = sal['saliency'], sal['fold_balanced_accuracy']

    rt_axis = np.linspace(RT_LO, RT_HI, 200)
    rt_prof = S.mean(axis=2)
    mz_prof = S.mean(axis=1)[:, MZ_LO:MZ_HI]
    windows = [(40, 44), (33, 37), (24, 28), (18, 22), (14, 18)]
    shares = {}
    for lo, hi in windows:
        w = (rt_axis >= lo) & (rt_axis < hi)
        shares[f'{lo}–{hi}'] = rt_prof[:, w].sum(axis=1) / rt_prof.sum(axis=1)

    # Example chromatogram per species: the first sample of each class (as in figs/saliency-3d.png)
    examples = []
    for c in range(4):
        i = int(np.where(y == c)[0][0])
        X = np.load(paths[i])['chroma'].astype(np.float32)
        examples.append({'species': SPECIES[c], 'sample': sample_ids[i],
                         'chroma': quantise(np.sqrt(coarsen(X))),
                         'saliency': quantise(coarsen(S[c]))})

    ablation = json.loads((RESULTS / 'ablation_1d_vs_2d_results.json').read_text())

    site = {
        'species': SPECIES,
        'n_samples': int(len(y)),
        'composition': [{'species': s, 'part': p, 'n': n}
                        for (s, p), n in sorted(composition.items())],
        'individuals': [{'species': s, 'individual': g, 'n': n}
                        for (s, g), n in sorted(individuals.items())],
        'methods': methods(),
        'ablation': {k: r3(v['balanced_accuracy']) for k, v in ablation.items()},
        'saliency': {
            'rt': r3(rt_axis), 'mz': list(range(MZ_LO, MZ_HI)),
            'rt_profile': [r3(p / p.max()) for p in rt_prof],
            'mz_profile': [r3(p / p.max()) for p in mz_prof],
            'rt_corr': r3(np.corrcoef(rt_prof)),
            'mz_corr': r3(np.corrcoef(S.mean(axis=1)[:, :600])),   # as in README / paper
            'window_share': {k: r3(v) for k, v in shares.items()},
            'fold_ba': r3(fold_ba),
            'grid_rt': r3(rt_axis[::GRID_RT] + (rt_axis[1] - rt_axis[0]) / 2),
            'grid_mz': list(range(GRID_MZ[0], GRID_MZ[1], GRID_MZ[2])),
            'examples': examples,
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text('// Generated by scripts/13_build_site_data.py — do not edit by hand.\n'
                   'window.SITE_DATA = ' + json.dumps(site, separators=(',', ':')) + ';\n')
    print(f'Wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)')


if __name__ == '__main__':
    main()
