"""
Spectrum preprocessing: binning, normalisation, HDF5 persistence.

Pipeline (follows standard MS spectral comparison practice):
  raw peaks → 1 Da integer binning → sqrt-intensity scaling → L2 normalisation

The 1000-dim output vector represents m/z 0–999 with 1 Da resolution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
from tqdm import tqdm


MZ_MAX = 1000  # exclusive upper bound; bins 0..999


def bin_spectrum(
    peaks: list[tuple[float, float]],
    mz_max: int = MZ_MAX,
) -> np.ndarray:
    """
    Convert a peak list to a fixed-length binned intensity vector.

    m/z values are rounded to the nearest integer and clipped to [0, mz_max-1].
    If multiple peaks map to the same bin, the maximum intensity wins.
    """
    vec = np.zeros(mz_max, dtype=np.float32)
    for mz, intensity in peaks:
        bin_idx = int(round(mz))
        if 0 <= bin_idx < mz_max and intensity > 0:
            vec[bin_idx] = max(vec[bin_idx], float(intensity))
    return vec


def sqrt_l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Square-root compress intensities then L2-normalise.

    sqrt compression reduces the dominance of base peaks and is standard
    for cosine-similarity spectral matching (NIST, MoNA).
    """
    vec = np.sqrt(np.maximum(vec, 0.0))
    norm = np.linalg.norm(vec)
    if norm > eps:
        vec = vec / norm
    return vec.astype(np.float32)


def preprocess_spectrum(
    peaks: list[tuple[float, float]], mz_max: int = MZ_MAX
) -> np.ndarray:
    """Full pipeline: bin → sqrt → L2."""
    return sqrt_l2_normalize(bin_spectrum(peaks, mz_max), )


def build_hdf5(
    records: list[dict],
    output_path: str | Path,
    val_fraction: float = 0.1,
    mz_max: int = MZ_MAX,
    seed: int = 42,
    chemical_class_key: str = "classyfire_class",
) -> Path:
    """
    Build an HDF5 dataset from a list of parsed spectrum records.

    HDF5 layout:
      /spectra      float32 [N, mz_max]
      /inchikeys    bytes   [N]
      /smiles       bytes   [N]
      /sources      bytes   [N]
      /split        bytes   [N]  ("train" | "val")
      attrs: mz_max, n_train, n_val, creation_date

    Val split is stratified by source if no class metadata is present.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = len(records)

    # Build spectra matrix
    spectra = np.zeros((n, mz_max), dtype=np.float32)
    inchikeys = []
    smiles_list = []
    sources = []

    for i, rec in enumerate(tqdm(records, desc="Preprocessing spectra")):
        spectra[i] = preprocess_spectrum(rec["peaks"], mz_max)
        inchikeys.append(rec["inchikey"].encode())
        smiles_list.append(rec.get("smiles", "").encode())
        sources.append(rec.get("source", "").encode())

    # Stratified val split by source
    source_arr = np.array([s.decode() for s in sources])
    split = np.array(["train"] * n, dtype=object)
    for src in np.unique(source_arr):
        mask = np.where(source_arr == src)[0]
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_fraction))
        split[mask[:n_val]] = "val"

    n_train = int((split == "train").sum())
    n_val = int((split == "val").sum())
    print(f"Split: {n_train} train / {n_val} val")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("spectra", data=spectra, compression="gzip", compression_opts=4)
        f.create_dataset("inchikeys", data=np.array(inchikeys))
        f.create_dataset("smiles", data=np.array(smiles_list))
        f.create_dataset("sources", data=np.array(sources))
        f.create_dataset("split", data=np.array([s.encode() for s in split]))
        f.attrs["mz_max"] = mz_max
        f.attrs["n_total"] = n
        f.attrs["n_train"] = n_train
        f.attrs["n_val"] = n_val

    print(f"Saved HDF5 → {output_path}  ({n} spectra, {mz_max} bins)")
    return output_path


def report_dataset_stats(h5_path: str | Path) -> None:
    """Print basic statistics about a pretraining HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        spectra = f["spectra"][:]
        sources = [s.decode() for s in f["sources"][:]]
        split_arr = [s.decode() for s in f["split"][:]]

    n = len(spectra)
    nonzero_per_spectrum = (spectra > 0).sum(axis=1)
    source_counts = {s: sources.count(s) for s in set(sources)}

    print(f"\n=== Dataset statistics ===")
    print(f"Total spectra     : {n}")
    print(f"Train / Val split : {split_arr.count('train')} / {split_arr.count('val')}")
    print(f"Source breakdown  : {source_counts}")
    print(f"Non-zero bins/spectrum  mean={nonzero_per_spectrum.mean():.1f}  "
          f"median={np.median(nonzero_per_spectrum):.1f}  "
          f"max={nonzero_per_spectrum.max()}")
    print(f"Spectrum L2 norms : mean={np.linalg.norm(spectra, axis=1).mean():.3f}")
