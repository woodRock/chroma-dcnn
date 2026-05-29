"""
Run baselines (PLS-DA, RF, SVM, MLP) on the MTBLS288 rice cultivar dataset.

Three feature representations:
  sum        — per-m/z sum spectrum across all RT scans
  max_proj   — per-m/z maximum across RT bins (peak heights)
  chroma_pca — full 2D chromatogram flattened, 50 within-fold PCs

Usage:
  python scripts/mtbls288_run_baselines.py
  python scripts/mtbls288_run_baselines.py --config configs/finetune_rice.yaml

Results are written to results/mtbls288/baseline_gcms_{representation}_results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from chroma_dcnn.evaluation.baselines import baseline_cv
from chroma_dcnn.evaluation.stats import compare_conditions


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds        = cfg['task'].get('cv_seeds', list(range(10)))
    cv_strategy  = cfg['task'].get('cv_strategy', 'kfold')
    cv_folds     = cfg['task'].get('cv_folds', 4)
    data_dir     = Path(cfg['data']['data_dir'])

    from chroma_dcnn.downstream.rice import load_rice_data, load_rice_chroma_features

    X_sum, y, _, _ = load_rice_data(data_dir)
    chroma_features, _ = load_rice_chroma_features(data_dir)

    groups_path = data_dir / 'groups.txt'
    groups = np.array(groups_path.read_text().splitlines()) if groups_path.exists() else None
    if cv_strategy == 'grouped' and groups is not None:
        print(f'Using grouped CV — {len(set(groups))} biological replicates')

    representations = {
        'sum':        (X_sum,                          None),
        'max_proj':   (chroma_features['max_proj'],    None),
        'chroma_pca': (chroma_features['chroma_flat'], 50),
    }

    for rep_name, (X_rep, n_pca) in representations.items():
        print(f'\n=== Rice ({rep_name}) ===')
        rep_results = {}
        for model_name in ['plsda', 'rf', 'svm', 'mlp']:
            print(f'  Running {model_name} …')
            rep_results[model_name] = baseline_cv(
                X_rep, y, model_name, seeds=seeds,
                cv_strategy=cv_strategy, cv_folds=cv_folds,
                groups=groups, n_pca_components=n_pca,
            )
            ba = rep_results[model_name]['balanced_accuracy']
            print(f'    BA: {np.mean(ba):.3f} ± {np.std(ba):.3f}')

        _save_and_compare(rep_results, output_dir, f'baseline_gcms_{rep_name}')


def _save_and_compare(baseline_results: dict, output_dir: Path, name: str) -> None:
    ba_results = {k: v['balanced_accuracy'] for k, v in baseline_results.items()}
    comparison_df = compare_conditions(ba_results)
    print(f'\n--- Baseline comparisons ({name}) ---')
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(output_dir / f'{name}_comparisons.csv', index=False)
    with open(output_dir / f'{name}_results.json', 'w') as f:
        json.dump({
            k: {m: [float(v) for v in vals] for m, vals in metrics.items()}
            for k, metrics in baseline_results.items()
        }, f, indent=2)
    print(f'Saved → {output_dir}/{name}_results.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run baselines on MTBLS288 rice GC-MS')
    ap.add_argument('--config', default='configs/finetune_rice.yaml')
    args = ap.parse_args()
    main(args.config)
