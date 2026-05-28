"""
Preprocess raw GC-MS fish oil CSVs into numpy arrays for MSFormer fine-tuning.

Each CSV (processed by Bach) is a 2D GC-MS chromatogram:
  rows    = RT scans (~4800, from 5 to 45 min)
  cols    = Time, TIC, then 344 m/z channels (50–543 Da, non-contiguous)

Processing:
  1. Sum intensities across all RT scans → one spectrum per sample
  2. Map to the 1000-bin m/z grid used by the pretrained encoder
  3. Apply sqrt + L2 normalisation (matches pretraining pipeline)

Output (written to data/fish_oil/):
  X.npy      [N, 1000]  float32  — normalised sum spectra
  y.npy      [N]        int64    — class labels (0=SNA, 1=GUR, 2=TAR, 3=BCO)
  groups.txt  N lines             — biological group (fish individual ID)

Usage:
  python scripts/preprocess_fish_oil.py
  python scripts/preprocess_fish_oil.py --raw-dir data/fish_oil/raw --output-dir data/fish_oil
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPECIES_MAP = {"SNA": 0, "GUR": 1, "TAR": 2, "BCO": 3}
_FNAME_RE = re.compile(r"\d+ - (\w{3})(\d+) .+\.csv", re.IGNORECASE)

MZ_MAX = 1000


def _parse_filename(fname: str) -> tuple[int, str] | None:
    """Return (label, group_id) or None if filename doesn't match."""
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    species = m.group(1).upper()
    fish_id = m.group(1).upper() + m.group(2)   # e.g. "SNA1", "BCO4"
    label = SPECIES_MAP.get(species)
    if label is None:
        return None
    return label, fish_id


def _read_mz_channels(filepath: Path) -> list[int]:
    """Read m/z channel indices from the second header row."""
    with open(filepath, encoding="utf-8") as fh:
        fh.readline()  # row 0: "X-data, Y-data, Packet #0, ..."
        row1 = fh.readline()
    parts = [p.strip() for p in row1.split(",")]
    mz_channels = []
    for p in parts[2:]:
        try:
            mz_channels.append(int(float(p)))
        except ValueError:
            pass
    return mz_channels


def process_file(filepath: Path, mz_channels: list[int]) -> np.ndarray:
    """Sum m/z intensities across all scans, map to 1000-bin grid."""
    df = pd.read_csv(filepath, skiprows=2, header=None, low_memory=False)
    mz_data = df.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    # Sum across RT scans → [344] sum spectrum
    sum_spec = mz_data.sum(axis=0)

    # Map to dense 1000-bin grid
    spectrum = np.zeros(MZ_MAX, dtype=np.float32)
    for i, mz in enumerate(mz_channels):
        if 0 <= mz < MZ_MAX and i < len(sum_spec):
            spectrum[mz] += sum_spec[i]

    return spectrum


def sqrt_l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.sqrt(np.maximum(vec, 0.0))
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


def main(raw_dir: Path, output_dir: Path) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {raw_dir}")

    # Read m/z channel layout from the first file (same across all files)
    first_csv = next((f for f in csv_files if _parse_filename(f.name)), None)
    if first_csv is None:
        raise RuntimeError("No parseable CSV files found.")
    mz_channels = _read_mz_channels(first_csv)
    print(f"m/z channels: {len(mz_channels)}, range {mz_channels[0]}–{mz_channels[-1]}")

    X_list, y_list, groups_list, ids_list = [], [], [], []

    for fpath in csv_files:
        result = _parse_filename(fpath.name)
        if result is None:
            print(f"  Skipping (no label): {fpath.name}")
            continue
        label, group_id = result

        spectrum = process_file(fpath, mz_channels)
        spectrum = sqrt_l2_normalize(spectrum)

        X_list.append(spectrum)
        y_list.append(label)
        groups_list.append(group_id)
        ids_list.append(fpath.stem)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    (output_dir / "groups.txt").write_text("\n".join(groups_list))
    (output_dir / "sample_ids.txt").write_text("\n".join(ids_list))

    inv_map = {v: k for k, v in SPECIES_MAP.items()}
    classes, counts = np.unique(y, return_counts=True)
    print(f"\nSaved {len(X)} samples → {output_dir}")
    print(f"X shape: {X.shape}  |  dtype: {X.dtype}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({inv_map[c]}): {n} samples")
    print(f"Biological groups: {sorted(set(groups_list))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/fish_oil/raw", type=Path)
    ap.add_argument("--output-dir", default="data/fish_oil", type=Path)
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir)
