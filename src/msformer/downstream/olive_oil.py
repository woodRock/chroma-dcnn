"""
Olive oil geographic origin — GC-IMS downstream task.

Dataset: Christmann & Weller, Mendeley Data DOI 10.17632/fr9t5fkkvz.3
         EVOO samples from Spain (0), Italy (1), Greece (2).

Each measurement is a (retention_times × drift_times) 2D intensity matrix,
e.g. shape (6939, 3150).  We reduce to 1D then bin to 1000 dims for the
pretrained model.

Two reduction strategies are evaluated and compared:
  sum  — collapse drift-time dimension by summation → TIC-like retention profile
  apex — extract the reactant-ion peak (RIP) apex row along the drift axis

Domain-mismatch note:
  The pretrained model learned from EI-MS spectra (m/z × intensity).
  The GC-IMS input is a retention-time profile or drift-time spectrum —
  a fundamentally different physical observable.  Positive transfer despite
  this gap would suggest the model learned generic intensity-pattern features
  that are not m/z-specific.  This must be stated explicitly in the paper.

Baseline: gc-ims-tools PLS-DA (Christmann et al., 2022, DOI 10.1016/j.foodchem.2022.133476).
  Reproduce using the published Python package on the same train/test splits.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from msformer.data.preprocess import sqrt_l2_normalize


# Expected class labels in the dataset filename or folder naming convention.
# Adjust if the Mendeley archive uses different naming.
_CLASS_MAP = {"spain": 0, "italy": 1, "greece": 2, "es": 0, "it": 1, "gr": 2}


def load_olive_oil_data(
    data_dir: str | Path,
    reduction: str = "sum",  # "sum" | "apex"
    target_dim: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load GC-IMS olive oil data and reduce to 1D feature vectors.

    Parameters
    ----------
    data_dir   : directory containing the Mendeley dataset files
    reduction  : 'sum' collapses drift-time by summation;
                 'apex' extracts the single drift-time slice with max total intensity
    target_dim : bin the 1D profile to this many dimensions (match pretrain mz_max)

    Returns
    -------
    X      : [N, target_dim]  float32 normalised feature matrix
    y      : [N]              int labels (0=Spain, 1=Italy, 2=Greece)
    labels : list of string sample IDs
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Olive oil data directory not found: {data_dir}\n"
            f"Download from https://data.mendeley.com/datasets/fr9t5fkkvz/3\n"
            f"Then run: python scripts/preprocess_olive_oil.py"
        )

    # Prefer pre-processed numpy arrays (output of preprocess_olive_oil.py)
    x_key = "X_sum" if reduction == "sum" else "X_apex"
    x_npy = data_dir / f"{x_key}.npy"
    y_npy = data_dir / "y.npy"
    ids_txt = data_dir / "sample_ids.txt"

    if x_npy.exists() and y_npy.exists():
        X = np.load(x_npy).astype(np.float32)
        y = np.load(y_npy).astype(np.int64)
        sample_ids = ids_txt.read_text().splitlines() if ids_txt.exists() else [str(i) for i in range(len(y))]
        if X.shape[1] != target_dim:
            X = np.stack([_bin_to_target_dim(row, target_dim) for row in X])
        print(f"Loaded olive oil from numpy arrays (reduction={reduction}): {X.shape}")
        _report_class_counts(y)
        return X, y, sample_ids

    # Fallback: scan for raw .npy / CSV matrix files
    return _load_from_raw_files(data_dir, reduction, target_dim)


def _load_via_gcims_tools(
    data_dir: Path, reduction: str, target_dim: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load using the gc_ims_tools package (preferred path — reproduces published baseline).

    gc_ims_tools stores measurements as .mea files and provides Dataset objects.
    See: https://github.com/breathe-and-tell/gc-ims-tools
    """
    import gc_ims_tools as git  # type: ignore

    dataset = git.Dataset.from_folder(str(data_dir))
    Xs, ys, sample_ids = [], [], []

    for sample in dataset:
        matrix = sample.values  # shape (n_ret, n_drift) float32
        label = _infer_label(sample.name)
        if label is None:
            print(f"  Warning: cannot infer class for sample '{sample.name}', skipping")
            continue

        vec_1d = _reduce_matrix(matrix, reduction)
        vec_binned = _bin_to_target_dim(vec_1d, target_dim)
        vec_norm = sqrt_l2_normalize(vec_binned)

        Xs.append(vec_norm)
        ys.append(label)
        sample_ids.append(sample.name)

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    print(f"Loaded {len(y)} olive oil samples via gc_ims_tools  (reduction={reduction})")
    _report_class_counts(y)
    return X, y, sample_ids


def _load_from_raw_files(
    data_dir: Path, reduction: str, target_dim: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fallback loader scanning for .npy, .csv, or .txt matrix files.
    Expects filenames to contain origin labels (spain/italy/greece or ES/IT/GR).
    """
    candidates = sorted(
        list(data_dir.rglob("*.npy"))
        + list(data_dir.rglob("*.csv"))
        + list(data_dir.rglob("*.txt"))
    )
    if not candidates:
        raise RuntimeError(
            f"No data files found in {data_dir}.  "
            f"Expected .npy, .csv, or .txt GC-IMS matrix files."
        )

    Xs, ys, sample_ids = [], [], []
    for path in candidates:
        label = _infer_label(path.stem)
        if label is None:
            continue

        if path.suffix == ".npy":
            matrix = np.load(path).astype(np.float32)
        else:
            matrix = np.loadtxt(path, delimiter=",").astype(np.float32)

        if matrix.ndim == 1:
            vec_1d = matrix
        elif matrix.ndim == 2:
            vec_1d = _reduce_matrix(matrix, reduction)
        else:
            print(f"  Skipping {path.name}: unexpected shape {matrix.shape}")
            continue

        vec_binned = _bin_to_target_dim(vec_1d, target_dim)
        Xs.append(sqrt_l2_normalize(vec_binned))
        ys.append(label)
        sample_ids.append(path.stem)

    if not Xs:
        raise RuntimeError("No labelled samples could be loaded from olive oil data directory.")

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    print(f"Loaded {len(y)} olive oil samples (fallback loader, reduction={reduction})")
    _report_class_counts(y)
    return X, y, sample_ids


def reproduce_gcims_plsda_baseline(
    data_dir: Path,
    n_components: int = 3,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Reproduce the PLS-DA baseline from Christmann et al. (2022)
    using gc_ims_tools on the same dataset.

    Returns dict with balanced_accuracy and macro_f1 per fold.
    """
    try:
        import gc_ims_tools as git
    except ImportError:
        raise ImportError(
            "gc_ims_tools is required to reproduce the published baseline. "
            "Install with: pip install gc_ims_tools"
        )

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    dataset = git.Dataset.from_folder(str(data_dir))

    # Use gc_ims_tools built-in feature extraction (RIP-relative areas or PARAFAC)
    # For a direct baseline match, use raw flattened matrices as in the paper.
    X_raw, ys, _ = [], [], []
    for sample in dataset:
        label = _infer_label(sample.name)
        if label is None:
            continue
        X_raw.append(sample.values.ravel())
        ys.append(label)

    X_raw = np.stack(X_raw)
    y = np.array(ys)

    from msformer.evaluation.baselines import PLSDAClassifier

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    ba_scores, f1_scores = [], []
    for tr, te in skf.split(X_raw, y):
        clf = PLSDAClassifier(n_components=n_components)
        clf.fit(X_raw[tr], y[tr])
        preds = clf.predict(X_raw[te])
        ba_scores.append(balanced_accuracy_score(y[te], preds))
        f1_scores.append(f1_score(y[te], preds, average="macro", zero_division=0))

    print(
        f"PLS-DA baseline (gc_ims_tools, n_comp={n_components}):  "
        f"BA={np.mean(ba_scores):.3f}±{np.std(ba_scores):.3f}  "
        f"F1={np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f}"
    )
    return {"balanced_accuracy": ba_scores, "macro_f1": f1_scores}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reduce_matrix(matrix: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "sum":
        return matrix.sum(axis=1)  # sum over drift-time → retention-time profile
    elif reduction == "apex":
        # Find drift-time slice with maximum total intensity (the RIP row)
        rip_idx = matrix.sum(axis=0).argmax()
        return matrix[:, rip_idx]  # retention-time profile at RIP apex drift time
    else:
        raise ValueError(f"Unknown reduction: {reduction}.  Choose 'sum' or 'apex'.")


def _bin_to_target_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Resample a 1D vector to target_dim using linear interpolation."""
    src_x = np.linspace(0, 1, len(vec))
    tgt_x = np.linspace(0, 1, target_dim)
    return np.interp(tgt_x, src_x, vec).astype(np.float32)


def _infer_label(name: str) -> int | None:
    name_lower = name.lower()
    for key, val in _CLASS_MAP.items():
        if key in name_lower:
            return val
    return None


def _report_class_counts(y: np.ndarray) -> None:
    classes, counts = np.unique(y, return_counts=True)
    inv_map = {v: k for k, v in _CLASS_MAP.items() if v in classes}
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({inv_map.get(c, '?')}): {n} samples")
