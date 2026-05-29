"""
Download and preprocess MTBLS71 MaleVsFemale GC-MS urine data.

Source: MetaboLights MTBLS71 (https://www.ebi.ac.uk/metabolights/MTBLS71)
  235 GC-MS analyses of human urine; we use the 160-sample MaleVsFemale
  experiment (20 female + 20 male donors × 2 treatments × 2 replicates).

Filename convention:
  MaleVsFemale_{treatment}_{sex}_{ethnicity}_{age}_{donor_id}_{rep}.CDF
  treatment : NT (no treatment) | UT (urease-treated)
  sex       : Female | Male

Labels (binary):
  0 = Female
  1 = Male

Groups (for grouped CV): donor ID (BRH######)

CDF format: ANDI/NetCDF (sparse mass/intensity per scan)
  Variables used: scan_acquisition_time, scan_index, point_count,
                  mass_values, intensity_values, total_intensity

Processing pipeline (matches fish oil chroma preprocessing):
  1. Reconstruct dense [n_scans × 1000] spectra from sparse CDF data
  2. Bin RT axis into N_BINS (200) equal windows; keep highest-TIC scan per bin
  3. sqrt + L2 normalise each bin
  4. Save per-sample [200, 1000] float32 chromatogram as compressed .npz
  5. Also build sum-spectrum X.npy [N, 1000] for baseline methods

Output (data/mtbls71/):
  raw/           raw .CDF files (keep for reproducibility)
  chroma/        per-sample [200, 1000] chromatogram .npz files
  X.npy          [N, 1000] sqrt+L2-normalised sum spectra
  y.npy          [N] int64 labels (0=Female, 1=Male)
  groups.txt     donor ID per sample
  sample_ids.txt filename stem per sample

Usage:
  python scripts/mtbls71_download_preprocess.py
  python scripts/mtbls71_download_preprocess.py --n-bins 200 --workers 8
  python scripts/mtbls71_download_preprocess.py --no-download   # process already-downloaded files
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

FTP_BASE = "ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS71/FILES/"
MZ_MAX = 1000
N_BINS_DEFAULT = 200

FNAME_RE = re.compile(
    r"MaleVsFemale_(NT|UT)_(Female|Male)_\w+_\d+_(BRH\d+)_[A-D]\.CDF", re.IGNORECASE
)

SEX_LABEL = {"Female": 0, "Male": 1}


def _parse_filename(fname: str) -> tuple[int, str] | None:
    """Return (sex_label, donor_id) or None if filename doesn't match."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    sex = m.group(2).capitalize()
    donor_id = m.group(3).upper()
    label = SEX_LABEL.get(sex)
    if label is None:
        return None
    return label, donor_id


def _read_cdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a GC-MS ANDI/netCDF file.

    Returns
    -------
    spectra : [n_scans, MZ_MAX] float32  — dense intensity matrix
    tic     : [n_scans] float32           — total ion current per scan
    """
    with netcdf_file(str(path), "r", mmap=False) as f:
        rt = f.variables["scan_acquisition_time"].data.copy()
        scan_idx = f.variables["scan_index"].data.copy()
        pt_count = f.variables["point_count"].data.copy()
        mass_vals = f.variables["mass_values"].data.copy()
        int_vals = f.variables["intensity_values"].data.copy()
        tic_raw = f.variables["total_intensity"].data.copy()

    n_scans = len(rt)
    spectra = np.zeros((n_scans, MZ_MAX), dtype=np.float32)

    for i in range(n_scans):
        s = int(scan_idx[i])
        e = s + int(pt_count[i])
        mz = np.round(mass_vals[s:e]).astype(np.int32)
        intensity = int_vals[s:e].astype(np.float32)
        valid = (mz >= 0) & (mz < MZ_MAX)
        np.add.at(spectra[i], mz[valid], intensity[valid])

    tic = tic_raw.astype(np.float32)
    return spectra, tic


def _bin_chromatogram(spectra: np.ndarray, tic: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin RT axis into n_bins windows; keep highest-TIC scan per bin."""
    n_scans = len(tic)
    bin_size = n_scans / n_bins
    result = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        start = int(i * bin_size)
        end = min(int((i + 1) * bin_size), n_scans)
        if start >= end:
            continue
        best = start + int(np.argmax(tic[start:end]))
        result[i] = spectra[best]
    return result


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def _download(fname: str, dest: Path) -> None:
    if dest.exists():
        return
    url = FTP_BASE + fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _process(cdf_path: Path, chroma_dir: Path, n_bins: int) -> np.ndarray:
    """Process one CDF → save chroma npz, return sum spectrum."""
    out_npz = chroma_dir / (cdf_path.stem + ".npz")
    if out_npz.exists():
        chroma = np.load(out_npz)["chroma"].astype(np.float32)
    else:
        spectra, tic = _read_cdf(cdf_path)
        chroma = _bin_chromatogram(spectra, tic, n_bins)
        chroma = _sqrt_l2(chroma)
        np.savez_compressed(out_npz, chroma=chroma)

    sum_spec = spectra.sum(axis=0) if not out_npz.exists() else chroma.sum(axis=0)
    return sum_spec


def _get_filenames() -> list[str]:
    """Fetch MaleVsFemale CDF filenames from the EBI FTP index page."""
    import urllib.request
    url = "http://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS71/FILES/"
    with urllib.request.urlopen(url, timeout=30) as r:
        html = r.read().decode()
    return re.findall(r'href="(MaleVsFemale[^"]+\.CDF)"', html)


def main(raw_dir: Path, output_dir: Path, n_bins: int, workers: int, no_download: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching file list from EBI FTP …")
    fnames = sorted(_get_filenames())
    print(f"  Found {len(fnames)} MaleVsFemale CDF files")

    parseable = [(fn, _parse_filename(fn)) for fn in fnames]
    parseable = [(fn, meta) for fn, meta in parseable if meta is not None]
    print(f"  Parseable: {len(parseable)}")

    if not no_download:
        print(f"\nDownloading {len(parseable)} files ({workers} parallel workers) …")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download, fn, raw_dir / fn): fn
                for fn, _ in parseable
            }
            done = 0
            for fut in as_completed(futures):
                fn = futures[fut]
                done += 1
                if fut.exception():
                    print(f"  ERROR {fn}: {fut.exception()}")
                elif done % 20 == 0 or done == len(parseable):
                    print(f"  Downloaded {done}/{len(parseable)}")

    print(f"\nPreprocessing to {n_bins}-bin chromatograms …")
    X_list, y_list, groups_list, ids_list = [], [], [], []

    for i, (fn, (label, donor_id)) in enumerate(parseable, 1):
        cdf_path = raw_dir / fn
        if not cdf_path.exists():
            print(f"  SKIP (not downloaded): {fn}")
            continue

        out_npz = chroma_dir / (cdf_path.stem + ".npz")

        if out_npz.exists():
            chroma = np.load(out_npz)["chroma"].astype(np.float32)
            sum_spec = chroma.sum(axis=0)
        else:
            spectra, tic = _read_cdf(cdf_path)
            chroma = _bin_chromatogram(spectra, tic, n_bins)
            chroma = _sqrt_l2(chroma)
            np.savez_compressed(out_npz, chroma=chroma)
            sum_spec = spectra.sum(axis=0)

        sum_spec_norm = _sqrt_l2(sum_spec[None])[0]

        X_list.append(sum_spec_norm)
        y_list.append(label)
        groups_list.append(donor_id)
        ids_list.append(cdf_path.stem)

        if i % 20 == 0 or i == len(parseable):
            print(f"  {i}/{len(parseable)} done …")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    (output_dir / "groups.txt").write_text("\n".join(groups_list))
    (output_dir / "sample_ids.txt").write_text("\n".join(ids_list))

    label_names = {v: k for k, v in SEX_LABEL.items()}
    classes, counts = np.unique(y, return_counts=True)
    print(f"\nSaved {len(X)} samples → {output_dir}")
    print(f"X shape: {X.shape}  |  dtype: {X.dtype}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({label_names[c]}): {n} samples")
    print(f"Donors: {len(set(groups_list))}  |  Chromatograms: {len(list(chroma_dir.glob('*.npz')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir",     default="data/mtbls71/raw",    type=Path)
    ap.add_argument("--output-dir",  default="data/mtbls71",        type=Path)
    ap.add_argument("--n-bins",      default=N_BINS_DEFAULT,        type=int)
    ap.add_argument("--workers",     default=8,                     type=int)
    ap.add_argument("--no-download", action="store_true",
                    help="Skip downloading; process already-present CDF files")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_bins, args.workers, args.no_download)
