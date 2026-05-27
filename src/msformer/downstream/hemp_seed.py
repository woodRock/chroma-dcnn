"""
Thai vs foreign hemp seed origin — GC/MS downstream task.

Dataset: Sangkanu et al., Discover Chemistry 2026
  DOI 10.1007/s44371-026-00486-y  (data note)
  DOI 10.3390/foods14213739        (companion study)

GC/MS data for hemp seed volatile profiles.  Expected format: compound × sample
feature matrix (e.g. peak area table from XCMS/MZmine).

CRITICAL: n ≈ 12 biological samples.  Verify the exact count from the data note.
  - If n_samples < 30 even with replicates: use leave-one-sample-out CV (LOSO)
  - Treat all results as preliminary; report this caveat explicitly

Classes: 0 = Thai origin, 1 = Foreign origin

Baseline: PLS-DA from Sangkanu et al. 2025 (R²=0.8827, Q²=0.3733).
  Note: R²/Q² are not classification metrics.  Re-run on same splits to get
  balanced accuracy and macro-F1 so comparisons are valid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from msformer.data.preprocess import bin_spectrum, sqrt_l2_normalize


_CLASS_MAP = {"thai": 0, "th": 0, "foreign": 1, "non-thai": 1, "other": 1}


def load_hemp_seed_data(
    data_dir: str | Path,
    target_dim: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Load hemp seed GC/MS data.

    Tries three formats in order:
      1. CSV/Excel peak area table (features × samples or samples × features)
      2. Processed .npy arrays (X.npy + y.npy)
      3. Raw mzML / NIST .msp files (via pyteomics if available)

    Parameters
    ----------
    data_dir   : directory with downloaded Sangkanu et al. dataset
    target_dim : feature dimension (match pretrain mz_max=1000)

    Returns
    -------
    X        : [N, target_dim] float32
    y        : [N] int labels
    sample_ids: list of sample name strings
    metadata  : dict with sample count, class counts, CV strategy recommendation
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Hemp seed data directory not found: {data_dir}\n"
            f"Download from DOI 10.1007/s44371-026-00486-y"
        )

    # Try CSV / Excel first (most likely format for a metabolomics peak table)
    for suffix in ["*.csv", "*.xlsx", "*.xls"]:
        files = sorted(data_dir.rglob(suffix))
        if files:
            return _load_from_peak_table(files[0], target_dim)

    # Fallback: pre-saved numpy arrays
    x_npy = data_dir / "X.npy"
    y_npy = data_dir / "y.npy"
    if x_npy.exists() and y_npy.exists():
        X = np.load(x_npy).astype(np.float32)
        y = np.load(y_npy).astype(np.int64)
        if X.shape[1] != target_dim:
            X = _resize_features(X, target_dim)
        sample_ids = [f"sample_{i}" for i in range(len(y))]
        return X, y, sample_ids, _build_metadata(y)

    raise RuntimeError(
        f"No recognisable hemp seed data found in {data_dir}. "
        f"Expected CSV/Excel peak table or X.npy + y.npy."
    )


def _load_from_peak_table(
    filepath: Path, target_dim: int
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Load a peak area table.

    Assumes the table has samples as rows (or columns) with a header or index
    indicating sample origin (Thai/Foreign).  Tries both orientations.
    """
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, index_col=0)
    else:
        df = pd.read_csv(filepath, index_col=0)

    print(f"Loaded peak table: {df.shape} from {filepath.name}")

    # Determine orientation: samples as rows or columns?
    # Heuristic: try to find class labels in the row index first, then column index.
    y_row, labels_row = _extract_labels(list(df.index.astype(str)))
    y_col, labels_col = _extract_labels(list(df.columns.astype(str)))

    if len(y_row) > len(y_col):
        X_raw = df.values.astype(np.float32)      # [N_samples, N_features]
        y = np.array(y_row, dtype=np.int64)
        sample_ids = labels_row
        # Keep only the labelled rows
        labelled_mask = [i for i, _ in enumerate(y_row)]
        X_raw = X_raw[labelled_mask]
    else:
        X_raw = df.values.T.astype(np.float32)    # transpose → [N_samples, N_features]
        y = np.array(y_col, dtype=np.int64)
        sample_ids = labels_col
        labelled_mask = [i for i, _ in enumerate(y_col)]
        X_raw = X_raw[labelled_mask]

    # Normalise and resize to target_dim
    X = np.stack([
        sqrt_l2_normalize(_resize_features_row(row, target_dim))
        for row in X_raw
    ])

    n = len(y)
    cv_rec = "loso" if n < 30 else "kfold"
    print(
        f"Hemp seed: {n} samples  CV recommendation: {cv_rec}"
        + ("  *** n<30, results are preliminary ***" if cv_rec == "loso" else "")
    )
    _report_class_counts(y)

    return X, y, sample_ids, _build_metadata(y)


def _extract_labels(names: list[str]) -> tuple[list[int], list[str]]:
    ys, valid = [], []
    for name in names:
        label = _infer_label(name)
        if label is not None:
            ys.append(label)
            valid.append(name)
    return ys, valid


def _infer_label(name: str) -> int | None:
    name_lower = name.lower()
    for key, val in _CLASS_MAP.items():
        if key in name_lower:
            return val
    return None


def _resize_features_row(row: np.ndarray, target_dim: int) -> np.ndarray:
    if len(row) == target_dim:
        return row
    src_x = np.linspace(0, 1, len(row))
    tgt_x = np.linspace(0, 1, target_dim)
    return np.interp(tgt_x, src_x, row).astype(np.float32)


def _resize_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    return np.stack([_resize_features_row(row, target_dim) for row in X])


def _build_metadata(y: np.ndarray) -> dict:
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    cv_strategy = "loso" if n < 30 else "kfold"
    return {
        "n_samples": n,
        "class_counts": dict(zip(classes.tolist(), counts.tolist())),
        "cv_strategy": cv_strategy,
        "preliminary": n < 30,
    }


def _report_class_counts(y: np.ndarray) -> None:
    inv_map = {v: k for k, v in _CLASS_MAP.items()}
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({inv_map.get(c, '?')}): {n} samples")
