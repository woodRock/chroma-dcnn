"""
NZ fish species identification — GC-MS downstream task.

Dataset: Bach & Wood, Victoria University of Wellington (internal)
  Four NZ commercial fish species (Snapper, Gurnard, Tarakihi, Blue Cod)
  measured by GC-MS lipid profiling across 6 body parts.
  103 samples, 6 biological individuals (SNA1, GUR2, TAR3, BCO4/5/6).

Raw data: 2D GC-MS chromatograms (4800 RT scans × 344 m/z channels, 50–543 Da)
Preprocessing: sum across RT scans → 1000-bin m/z vector → sqrt + L2 normalise
  (matches EI-MS pretraining domain of MoNA/MassBank)

Classes:
  0 = SNA (Snapper)
  1 = GUR (Gurnard)
  2 = TAR (Tarakihi)
  3 = BCO (Blue Cod)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SPECIES_MAP = {0: "SNA", 1: "GUR", 2: "TAR", 3: "BCO"}
SPECIES_LABEL = {"SNA": 0, "GUR": 1, "TAR": 2, "BCO": 3}


def load_fish_oil_data(
    data_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """
    Load preprocessed fish oil GC-MS data.

    Expects data_dir to contain X.npy, y.npy, and groups.txt produced by
    scripts/preprocess_fish_oil.py.

    Returns
    -------
    X         : [N, 1000] float32 — sqrt+L2 normalised sum spectra
    y         : [N] int64 — class labels (0=SNA, 1=GUR, 2=TAR, 3=BCO)
    sample_ids: list of sample name strings
    metadata  : dict with sample count, class counts, biological groups
    """
    data_dir = Path(data_dir)
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Preprocessed arrays not found in {data_dir}.\n"
            f"Run: python scripts/preprocess_fish_oil.py"
        )

    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    ids_path = data_dir / "sample_ids.txt"
    sample_ids = ids_path.read_text().splitlines() if ids_path.exists() else [
        f"sample_{i}" for i in range(len(y))
    ]

    groups_path = data_dir / "groups.txt"
    bio_groups = groups_path.read_text().splitlines() if groups_path.exists() else None

    classes, counts = np.unique(y, return_counts=True)
    print(f"Fish oil GC-MS: {len(X)} samples, X shape {X.shape}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({SPECIES_MAP.get(c, '?')}): {n} samples")
    if bio_groups:
        print(f"Biological groups: {sorted(set(bio_groups))}")

    meta = {
        "n_samples": len(y),
        "class_counts": dict(zip(classes.tolist(), counts.tolist())),
        "bio_groups": bio_groups,
        "preliminary": False,
    }
    return X, y, sample_ids, meta


def load_fish_oil_chroma_features(
    data_dir: str | Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load feature matrices for baseline evaluation from the 2D chroma NPZ files.

    The sum spectrum in X.npy collapses 4800 RT scans with equal weight, so
    ~4600 background scans dominate and retention-time structure is lost entirely.
    These representations are more informative:

    max_proj   [N, 1000]       — per-m/z maximum across all RT bins.
                                  Picks up true peak heights; background bins have
                                  low max values and contribute minimally.
                                  Drop-in replacement for the sum spectrum.

    chroma_flat [N, 200*1000]  — full 2D chromatogram flattened.
                                  Preserves both spectral identity AND retention
                                  time; intended for within-fold PCA reduction
                                  (pass n_pca_components to baseline_cv).

    Returns
    -------
    features : dict with keys 'max_proj' and 'chroma_flat'
    y        : [N] int64 labels
    """
    npz_paths, y = load_fish_oil_chroma_paths(data_dir)

    chromas = [np.load(p)["chroma"].astype(np.float32) for p in npz_paths]
    # chromas[i]: [N_bins, mz_max]

    max_proj    = np.stack([c.max(axis=0) for c in chromas])          # [N, mz_max]
    chroma_flat = np.stack([c.reshape(-1)  for c in chromas])         # [N, N_bins*mz_max]

    return {"max_proj": max_proj, "chroma_flat": chroma_flat}, y


def load_fish_oil_chroma_paths(
    data_dir: str | Path,
) -> tuple[list[Path], np.ndarray]:
    """
    Return ordered list of per-sample chroma .npz paths and matching label array.

    Expects data_dir/chroma/{sample_id}.npz files produced by
    scripts/preprocess_fish_oil_chroma.py.
    """
    data_dir  = Path(data_dir)
    chroma_dir = data_dir / "chroma"

    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_dir}\n"
            f"Run: python scripts/preprocess_fish_oil_chroma.py"
        )

    ids_path = data_dir / "sample_ids.txt"
    y_path   = data_dir / "y.npy"

    if ids_path.exists() and y_path.exists():
        sample_ids = ids_path.read_text().splitlines()
        y = np.load(y_path).astype(np.int64)
        npz_paths, valid_y = [], []
        for sid, label in zip(sample_ids, y):
            p = chroma_dir / (sid.replace(" ", "_").replace("-", "_") + ".npz")
            if p.exists():
                npz_paths.append(p)
                valid_y.append(label)
        y = np.array(valid_y, dtype=np.int64)
    else:
        npz_paths = sorted(chroma_dir.glob("*.npz"))
        y_list = []
        for p in npz_paths:
            m = re.search(r"_([A-Z]{3})\d+_", p.stem.upper())
            y_list.append(SPECIES_LABEL.get(m.group(1), -1) if m else -1)
        y = np.array(y_list, dtype=np.int64)
        npz_paths = [p for p, lbl in zip(npz_paths, y) if lbl >= 0]
        y = y[y >= 0]

    print(f"Fish oil chroma data: {len(npz_paths)} samples from {chroma_dir}")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({SPECIES_MAP.get(c,'?')}): {n} samples")

    return npz_paths, y


_NPZ_RE = re.compile(r"\d+___(\w{3})\d+", re.IGNORECASE)


def load_fish_oil_scan_paths(
    data_dir: str | Path,
) -> tuple[list[Path], np.ndarray]:
    """
    Return ordered list of per-sample .npz paths and matching label array.

    Expects data_dir/scans/{sample_id}.npz files produced by
    scripts/preprocess_fish_oil_scans.py, alongside y.npy and sample_ids.txt.

    Returns
    -------
    npz_paths : list of Path, length N
    y         : int64 ndarray [N]
    """
    data_dir = Path(data_dir)
    scans_dir = data_dir / "scans"

    if not scans_dir.exists():
        raise FileNotFoundError(
            f"Scan directory not found: {scans_dir}\n"
            f"Run: python scripts/preprocess_fish_oil_scans.py"
        )

    # Match npz filenames back to labels using sample_ids.txt + y.npy
    ids_path = data_dir / "sample_ids.txt"
    y_path = data_dir / "y.npy"

    if ids_path.exists() and y_path.exists():
        sample_ids = ids_path.read_text().splitlines()
        y = np.load(y_path).astype(np.int64)
        npz_paths = []
        valid_y = []
        for sid, label in zip(sample_ids, y):
            npz_name = sid.replace(" ", "_").replace("-", "_") + ".npz"
            p = scans_dir / npz_name
            if p.exists():
                npz_paths.append(p)
                valid_y.append(label)
        y = np.array(valid_y, dtype=np.int64)
    else:
        # Fallback: parse labels from npz filenames directly
        npz_paths = sorted(scans_dir.glob("*.npz"))
        y_list = []
        for p in npz_paths:
            # filename pattern: "1___SNA1_Frame_A.npz"
            m = re.search(r"_([A-Z]{3})\d+_", p.stem.upper())
            label = SPECIES_LABEL.get(m.group(1), -1) if m else -1
            y_list.append(label)
        y = np.array(y_list, dtype=np.int64)
        npz_paths = [p for p, lbl in zip(npz_paths, y) if lbl >= 0]
        y = y[y >= 0]

    print(f"Fish oil scan data: {len(npz_paths)} samples from {scans_dir}")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({SPECIES_MAP.get(c,'?')}): {n} samples")

    return npz_paths, y
