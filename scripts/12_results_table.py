"""
Build the main results table (paper Table 1) from the saved fish oil results.

Ranks all 14 conditions (2 ChromaDCNN + 12 classical baselines) by mean
balanced accuracy and compares the best method against every other with a
two-sided Mann-Whitney U test, Bonferroni-corrected over the 13 comparisons.
Effect size is the rank-biserial |r| = |1 - 2U / (n1 n2)|.

Inputs (results/fish_oil/):
  chroma_results.json                       scripts/04_finetune_evaluate.py
  baseline_gcms_{sum,max_proj,chroma_pca}_results.json   scripts/05_run_baselines.py

Outputs (results/fish_oil/):
  main_table.csv
  main_table.md

Usage:
  python scripts/12_results_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path('results/fish_oil')
ALPHA = 0.05

CNN_CONDITIONS = {
    'chroma_pretrain': 'ChromaDCNN (chroma_pretrain)',
    'from_scratch':    'ChromaDCNN (from_scratch)',
}
CLASSIFIERS = {'plsda': 'PLS-DA', 'rf': 'Random Forest', 'svm': 'SVM (RBF)', 'mlp': 'MLP'}
REPRESENTATIONS = {'sum': 'sum spectrum', 'max_proj': 'max-projection',
                   'chroma_pca': 'chroma-PCA'}


def load_scores() -> list[tuple[str, str, np.ndarray]]:
    rows = []
    cnn = json.loads((RESULTS_DIR / 'chroma_results.json').read_text())
    for key, name in CNN_CONDITIONS.items():
        rows.append((name, '2D chromatogram', np.array(cnn[key]['balanced_accuracy'])))
    for rep, rep_name in REPRESENTATIONS.items():
        base = json.loads((RESULTS_DIR / f'baseline_gcms_{rep}_results.json').read_text())
        for clf, clf_name in CLASSIFIERS.items():
            rows.append((clf_name, rep_name, np.array(base[clf]['balanced_accuracy'])))
    return rows


def build_table(rows: list[tuple[str, str, np.ndarray]]) -> pd.DataFrame:
    rows = sorted(rows, key=lambda r: -r[2].mean())
    best = rows[0][2]
    n_comparisons = len(rows) - 1
    records = []
    for rank, (method, rep, ba) in enumerate(rows, start=1):
        rec = {'rank': rank, 'method': method, 'representation': rep,
               'ba_mean': ba.mean(), 'ba_std': ba.std(), 'n': len(ba),
               'abs_r_vs_best': np.nan, 'p_raw': np.nan, 'p_bonferroni': np.nan,
               'significant': None}
        if rank > 1:
            u, p = mannwhitneyu(best, ba, alternative='two-sided')
            p_corr = min(1.0, p * n_comparisons)
            rec.update(abs_r_vs_best=abs(1 - 2 * u / (len(best) * len(ba))),
                       p_raw=p, p_bonferroni=p_corr, significant=p_corr < ALPHA)
        records.append(rec)
    return pd.DataFrame(records)


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        '| Rank | Method | Representation | Balanced acc. '
        '| \\|r\\| vs best | p (Bonferroni) | Sig. |',
        '|:----:|--------|----------------|:-------------:'
        '|:-------------:|:--------------:|:----:|',
    ]
    for _, r in df.iterrows():
        ba = f'{r.ba_mean:.3f} ± {r.ba_std:.3f}'
        if r['rank'] == 1:
            lines.append(f'| 1 | **{r.method}** | {r.representation} | **{ba}** | — | — | — |')
        else:
            lines.append(f'| {r["rank"]} | {r.method} | {r.representation} | {ba} '
                         f'| {r.abs_r_vs_best:.3f} | {r.p_bonferroni:.2e} '
                         f'| {"Yes" if r.significant else "No"} |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    df = build_table(load_scores())
    df.to_csv(RESULTS_DIR / 'main_table.csv', index=False)
    (RESULTS_DIR / 'main_table.md').write_text(to_markdown(df))
    print(to_markdown(df))
    print(f'Saved → {RESULTS_DIR}/main_table.{{csv,md}}')


if __name__ == '__main__':
    main()
