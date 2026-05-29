"""
MTBLS71 urine metabolomics — GC-MS downstream task.

Dataset: Metz et al. (2014), Pacific Northwest National Laboratory
  MetaboLights study MTBLS71 — MaleVsFemale experiment
  160 GC-MS analyses of human urine (20 female + 20 male donors,
  2 treatments [NT/UT] × 2 replicates each).

Raw data: sparse ANDI/netCDF chromatograms (~4941 RT scans, 50–299 m/z)
Preprocessing: uniform 200-bin RT resampling, sqrt + L2 normalisation per bin

Classes:
  0 = Female
  1 = Male
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABEL_MAP = {0: "Female", 1: "Male"}


def load_urine_chroma_paths(
    data_dir: str | Path,
) -> tuple[list[Path], np.ndarray]:
    """
    Return ordered list of per-sample chroma .npz paths and matching label array.

    Expects data_dir/chroma/*.npz produced by scripts/download_preprocess_mtbls71.py,
    alongside sample_ids.txt and y.npy.
    """
    data_dir = Path(data_dir)
    chroma_dir = data_dir / "chroma"

    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_dir}\n"
            f"Run: python scripts/download_preprocess_mtbls71.py"
        )

    ids_path = data_dir / "sample_ids.txt"
    y_path = data_dir / "y.npy"

    if ids_path.exists() and y_path.exists():
        sample_ids = ids_path.read_text().splitlines()
        y = np.load(y_path).astype(np.int64)
        npz_paths, valid_y = [], []
        for sid, label in zip(sample_ids, y):
            p = chroma_dir / (sid + ".npz")
            if p.exists():
                npz_paths.append(p)
                valid_y.append(label)
        y = np.array(valid_y, dtype=np.int64)
    else:
        npz_paths = sorted(chroma_dir.glob("*.npz"))
        y = np.array(
            [0 if "Female" in p.stem else 1 for p in npz_paths], dtype=np.int64
        )

    print(f"Urine GC-MS: {len(npz_paths)} samples from {chroma_dir}")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} samples")

    return npz_paths, y


def load_urine_data(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Load preprocessed urine sum spectra.

    Expects X.npy, y.npy, sample_ids.txt produced by
    scripts/download_preprocess_mtbls71.py.
    """
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)

    ids_path = data_dir / "sample_ids.txt"
    sample_ids = ids_path.read_text().splitlines() if ids_path.exists() else [
        f"sample_{i}" for i in range(len(y))
    ]

    classes, counts = np.unique(y, return_counts=True)
    print(f"Urine GC-MS (sum spectra): {len(X)} samples, X shape {X.shape}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} samples")

    meta = {
        "n_samples": len(y),
        "class_counts": dict(zip(classes.tolist(), counts.tolist())),
    }
    return X, y, sample_ids, meta


def load_urine_chroma_features(
    data_dir: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load max-projection and flattened 2D chromatogram features for baselines.

    Returns
    -------
    features : dict with keys 'max_proj' [N, 1000] and 'chroma_flat' [N, 200*1000]
    y        : [N] int64 labels
    """
    npz_paths, y = load_urine_chroma_paths(data_dir)
    chromas = [np.load(p)["chroma"].astype(np.float32) for p in npz_paths]
    max_proj = np.stack([c.max(axis=0) for c in chromas])
    chroma_flat = np.stack([c.reshape(-1) for c in chromas])
    return {"max_proj": max_proj, "chroma_flat": chroma_flat}, y
