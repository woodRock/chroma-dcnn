"""
Preprocess fish oil GC-MS CSVs into per-sample scan arrays for per-scan encoding.

For each sample:
  1. Read the 2D chromatogram (4800 scans × 344 m/z channels)
  2. Map m/z channels to the 1000-bin grid used by the pretrained encoder
  3. Select the top-N scans by TIC (total ion current) — these carry all the signal
  4. Normalize TIC weights so they sum to 1
  5. Apply sqrt + L2 normalisation to each scan (matches MSM pretraining)
  6. Save as a compressed .npz file

Output: data/fish_oil/scans/{sample_id}.npz  (one file per sample)
  scans       : float32 [n_top_scans, 1000]  — normalised m/z spectra
  tic_weights : float32 [n_top_scans]        — TIC weights, sums to 1

Usage:
  python scripts/preprocess_fish_oil_scans.py
  python scripts/preprocess_fish_oil_scans.py --n-top-scans 300
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MZ_MAX = 1000
_FNAME_RE = re.compile(r"\d+ - (\w{3}\d+) .+\.csv", re.IGNORECASE)


def _read_mz_channels(filepath: Path) -> list[int]:
    with open(filepath, encoding="utf-8") as fh:
        fh.readline()
        row1 = fh.readline()
    parts = [p.strip() for p in row1.split(",")]
    mz_channels = []
    for p in parts[2:]:
        try:
            mz_channels.append(int(float(p)))
        except ValueError:
            pass
    return mz_channels


def sqrt_l2_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def process_file(
    filepath: Path,
    mz_channels: list[int],
    n_top_scans: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scans       : [n_top_scans, MZ_MAX] float32  normalised
      tic_weights : [n_top_scans]         float32  sums to 1
    """
    df = pd.read_csv(filepath, skiprows=2, header=None, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    tic = df.iloc[:, 1].values.astype(np.float32)
    raw_mz = df.iloc[:, 2:].values.astype(np.float32)

    # Map non-contiguous m/z channels to the 1000-bin grid
    spectra = np.zeros((len(tic), MZ_MAX), dtype=np.float32)
    for col_idx, mz in enumerate(mz_channels):
        if 0 <= mz < MZ_MAX and col_idx < raw_mz.shape[1]:
            spectra[:, mz] = raw_mz[:, col_idx]

    # Select top-n scans by TIC
    n_top = min(n_top_scans, len(tic))
    top_indices = np.argpartition(tic, -n_top)[-n_top:]
    top_indices = top_indices[np.argsort(tic[top_indices])[::-1]]  # descending TIC

    top_scans = spectra[top_indices]          # [n_top, MZ_MAX]
    top_tic = tic[top_indices]                # [n_top]

    # Normalise each scan (sqrt + L2) — matches pretraining pipeline
    top_scans = sqrt_l2_normalize(top_scans)  # [n_top, MZ_MAX]

    # TIC weights sum to 1
    tic_sum = top_tic.sum()
    tic_weights = top_tic / (tic_sum + 1e-8)

    return top_scans, tic_weights


def main(raw_dir: Path, output_dir: Path, n_top_scans: int) -> None:
    csv_files = sorted(raw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    first_csv = next((f for f in csv_files if _FNAME_RE.match(f.name)), None)
    if first_csv is None:
        raise RuntimeError("No parseable CSV files found.")

    mz_channels = _read_mz_channels(first_csv)
    print(f"m/z channels: {len(mz_channels)}, range {mz_channels[0]}–{mz_channels[-1]}")
    print(f"Keeping top {n_top_scans} scans per sample by TIC")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for fpath in csv_files:
        m = _FNAME_RE.match(fpath.name)
        if not m:
            continue

        sample_id = fpath.stem.replace(" ", "_").replace("-", "_")
        out_path = output_dir / f"{sample_id}.npz"

        if out_path.exists():
            processed += 1
            continue

        scans, tic_weights = process_file(fpath, mz_channels, n_top_scans)
        np.savez_compressed(out_path, scans=scans, tic_weights=tic_weights)
        processed += 1

        if processed % 20 == 0:
            print(f"  {processed}/{len(csv_files)} done …")

    print(f"\nDone. {processed} samples → {output_dir}")
    print(f"Each sample: scans {(n_top_scans, MZ_MAX)}, tic_weights ({n_top_scans},)")
    total_mb = sum(p.stat().st_size for p in output_dir.glob("*.npz")) / 1e6
    print(f"Total size on disk: {total_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/fish_oil/raw", type=Path)
    ap.add_argument("--output-dir", default="data/fish_oil/scans", type=Path)
    ap.add_argument("--n-top-scans", default=200, type=int,
                    help="Number of highest-TIC scans to keep per sample")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_top_scans)
