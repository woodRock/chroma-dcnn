"""
MTBLS1187 wheat flour GC-QTOF — downstream task.

Dataset: Longin et al. (2020), Food Res Int 129:108748
  MetaboLights MTBLS1187 — 120 GC-QTOF samples
  40 winter wheat cultivars × 3 German locations (GAL, HOH, IHO)

Classification target: wheat era (binary)
  0 = old    (18 cultivars; released 1962-1999)
  1 = modern (22 cultivars; released 2005-2014)

Groups (for grouped CV): cultivar PGL ID (e.g. "PGL_006")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LABEL_MAP = {0: "old", 1: "modern"}


def load_wheat_chroma_paths(
    data_dir: str | Path,
) -> tuple[list[Path], np.ndarray]:
    """Return ordered list of per-sample chroma .npz paths and matching label array."""
    data_dir   = Path(data_dir)
    chroma_dir = data_dir / "chroma"

    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_dir}\n"
            f"Run: python scripts/mtbls1187_preprocess.py"
        )

    ids_path = data_dir / "sample_ids.txt"
    y_path   = data_dir / "y.npy"

    if ids_path.exists() and y_path.exists():
        sample_ids = ids_path.read_text().splitlines()
        y          = np.load(y_path).astype(np.int64)
        npz_paths, valid_y = [], []
        for sid, label in zip(sample_ids, y):
            p = chroma_dir / (sid + ".npz")
            if p.exists():
                npz_paths.append(p)
                valid_y.append(label)
        y = np.array(valid_y, dtype=np.int64)
    else:
        npz_paths = sorted(chroma_dir.glob("*.npz"))
        y         = np.zeros(len(npz_paths), dtype=np.int64)

    print(f"Wheat GC-QTOF: {len(npz_paths)} samples from {chroma_dir}")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} samples")

    return npz_paths, y


def load_wheat_data(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load preprocessed wheat sum spectra (X.npy / y.npy)."""
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy").astype(np.float32)
    y = np.load(data_dir / "y.npy").astype(np.int64)

    ids_path   = data_dir / "sample_ids.txt"
    sample_ids = (
        ids_path.read_text().splitlines()
        if ids_path.exists()
        else [f"sample_{i}" for i in range(len(y))]
    )

    classes, counts = np.unique(y, return_counts=True)
    print(f"Wheat GC-QTOF (sum spectra): {len(X)} samples, X shape {X.shape}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({LABEL_MAP.get(c, '?')}): {n} samples")

    meta = {
        "n_samples":    len(y),
        "class_counts": dict(zip(classes.tolist(), counts.tolist())),
    }
    return X, y, sample_ids, meta


def load_wheat_chroma_features(
    data_dir: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load max-projection and flattened 2D chromatogram features for baselines."""
    npz_paths, y = load_wheat_chroma_paths(data_dir)
    chromas      = [np.load(p)["chroma"].astype(np.float32) for p in npz_paths]
    max_proj     = np.stack([c.max(axis=0) for c in chromas])
    chroma_flat  = np.stack([c.reshape(-1) for c in chromas])
    return {"max_proj": max_proj, "chroma_flat": chroma_flat}, y
