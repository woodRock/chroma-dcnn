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

import re
from pathlib import Path

import numpy as np
import pandas as pd

from msformer.data.preprocess import sqrt_l2_normalize


# Strict regex for Sangkanu et al. sample naming: TH01-O, FS02-MH, etc.
# Anchored to start of string to avoid matching compound names like "methyl"
_SAMPLE_RE = re.compile(r"^(TH|FS)(\d+)", re.IGNORECASE)
_CLASS_MAP = {"TH": 0, "FS": 1}  # prefix → class


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

    # Prefer the GC_MS_data subdirectory if present
    gc_ms_dir = data_dir / "GC_MS_data"
    search_dir = gc_ms_dir if gc_ms_dir.exists() else data_dir

    # Prefer the known-good non-transposed original file
    for candidate in [
        search_dir / "hemp-gcms-original.xlsx",
        search_dir / "hemp-gcms-threshold10.xlsx",
    ]:
        if candidate.exists():
            return _load_from_peak_table(candidate, target_dim)

    # Fallback: first CSV/Excel found
    for suffix in ["*.csv", "*.xlsx", "*.xls"]:
        files = [f for f in sorted(search_dir.rglob(suffix)) if "transpose" not in f.name]
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
    Load the Sangkanu et al. peak area table.

    Expected format: compounds as rows, samples as columns
    (e.g. hemp-gcms-original.xlsx with shape 61×12).
    Transposes to samples×features, infers class from column prefix (TH/FS).

    Biological groups (TH01, TH02, FS01, FS02) are tracked separately for
    leak-free LOSO-CV — all replicates of the same biological sample must
    stay on the same side of every train/test split.
    """
    if filepath.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, index_col=0)
    else:
        df = pd.read_csv(filepath, index_col=0)

    print(f"Loaded peak table: {df.shape} from {filepath.name}")

    # Determine orientation using strict TH/FS pattern matching
    col_names = list(df.columns.astype(str))
    row_names = list(df.index.astype(str))
    n_sample_cols = _count_sample_names(col_names)
    n_sample_rows = _count_sample_names(row_names)

    if n_sample_cols >= n_sample_rows:
        # Samples are columns (standard Sangkanu format: compounds×samples)
        sample_ids = [c for c in col_names if _is_sample_name(c)]
        df_samples = df[sample_ids].T             # [N_samples, N_features]
    else:
        # Samples are rows
        sample_ids = [r for r in row_names if _is_sample_name(r)]
        df_samples = df.loc[sample_ids, :]        # [N_samples, N_features]

    y = np.array([_infer_label(s) for s in sample_ids], dtype=np.int64)

    X_raw = df_samples.values.astype(np.float32)

    # Normalise and resize to target_dim
    X = np.stack([
        sqrt_l2_normalize(_resize_features_row(row, target_dim))
        for row in X_raw
    ])

    # Biological group IDs for leak-free LOSO splits
    bio_groups = [_biological_sample_id(s) for s in sample_ids]

    n = len(y)
    print(f"Hemp seed: {n} samples, {X_raw.shape[1]} features → {target_dim} dims")
    print(f"Biological groups: {sorted(set(bio_groups))}")
    print("*** n<30: LOSO-CV at biological-sample level recommended ***")
    _report_class_counts(y)

    meta = _build_metadata(y)
    meta["bio_groups"] = bio_groups
    meta["n_features_original"] = X_raw.shape[1]
    return X, y, sample_ids, meta


def _extract_labels(names: list[str]) -> tuple[list[int], list[str]]:
    ys, valid = [], []
    for name in names:
        label = _infer_label(name)
        if label is not None:
            ys.append(label)
            valid.append(name)
    return ys, valid


def _count_sample_names(names: list[str]) -> int:
    return sum(1 for n in names if _is_sample_name(n))


def _infer_label(name: str) -> int | None:
    """Return class label only for names matching the TH/FS sample pattern."""
    m = _SAMPLE_RE.match(str(name).strip())
    if m:
        return _CLASS_MAP.get(m.group(1).upper())
    return None


def _is_sample_name(name: str) -> bool:
    return _SAMPLE_RE.match(str(name).strip()) is not None


def _biological_sample_id(col_name: str) -> str:
    """'TH01-O' → 'TH01',  'FS02-MH' → 'FS02'"""
    m = _SAMPLE_RE.match(str(col_name).strip())
    if m:
        return m.group(0)  # e.g. 'TH01'
    return str(col_name).split("-")[0]


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
