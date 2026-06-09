"""
Download and preprocess MTBLS1187 wheat bread GC-QTOF data for ChromatogramCNN evaluation.

Source: MetaboLights MTBLS1187
  Longin et al. (2020) "Aroma and quality of breads baked from old and modern wheat varieties
  and their prediction from genomic and flour-based metabolite profiles"
  Food Research International 129:108748  DOI: 10.1016/j.foodres.2019.108748

  120 GC-QTOF flour metabolite profiles: 40 winter wheat cultivars × 3 German growing
  locations (GAL = Gatersleben, HOH = Hohenheim, IHO = Ihinger Hof).
  Instrument: Agilent 7890B GC / 7200 Q-TOF; m/z 60-800; 27 min run.
  CDF files are MetAlign baseline-corrected exports from MassHunter.

Classification target: wheat era (binary)
  0 = old    (18 cultivars; year of release 1962–1999)
  1 = modern (22 cultivars; year of release 2005–2014)

  OLD_VARIETIES below is derived from the paper's supplementary cultivar list.
  If any assignment disagrees with the published table, update the set and re-run.

Groups (for grouped CV): cultivar PGL ID (e.g. "PGL_006")
  All three location-samples of one cultivar go into the same CV fold so the model
  cannot memorise location-specific metabolic variation for a cultivar seen in training.

Processing pipeline (identical to fish oil / MTBLS71 / MTBLS288):
  1. Reconstruct dense [n_scans × 1000] spectra from sparse ANDI/netCDF
  2. Bin RT axis into N_BINS (200) equal windows; keep highest-TIC scan per bin
  3. sqrt + L2 normalise each bin
  4. Save per-sample [200, 1000] float32 chromatogram as compressed .npz
  5. Build sum-spectrum X.npy [N, 1000] for classical baseline methods

Output (data/mtbls1187/):
  raw/           raw baseline-corrected .cdf files (~2 GB; skipped if present)
  chroma/        per-sample [200, 1000] chromatogram .npz files
  X.npy          [N, 1000] sqrt+L2-normalised sum spectra
  y.npy          [N] int64 labels (0=old, 1=modern)
  groups.txt     cultivar PGL ID per sample
  sample_ids.txt CDF stem per sample

Usage:
  python scripts/mtbls1187_preprocess.py
  python scripts/mtbls1187_preprocess.py --workers 8
  python scripts/mtbls1187_preprocess.py --no-download   # process already-downloaded files
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

MZ_MAX = 1000
N_BINS_DEFAULT = 200
CHUNK_SIZE = 1 << 20  # 1 MB streaming chunks

# HTTPS is significantly faster than FTP for EBI downloads.
HTTPS_BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1187/FILES/"
FILES_INDEX_URL = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1187/FILES/"

ASSAY_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1187/"
    "a_MTBLS1187_GC-MS___metabolite_profiling.txt"
)
SAMPLE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1187/"
    "s_MTBLS1187.txt"
)

# Old winter wheat cultivars (year of release 1962–1999).
# Source: Longin et al. (2020), Food Res Int 129:108748, supplementary cultivar table.
# Every cultivar in the study not listed here is classified as modern (2005–2014).
OLD_VARIETIES: frozenset[str] = frozenset({
    "Apache",        # France  1998
    "Colonia",       # Germany
    "Damier",        # France
    "Epiroux",       # France
    "Florida",       #
    "Hobbit",        # UK      1977
    "Horizon",       #
    "Hymack",        # UK      1998
    "KAUZ",          # CIMMYT  1990s
    "MarisKinsman",  # UK      1975
    "MarisMarksman", # UK      1974
    "Markant",       # Germany
    "Mission",       #
    "Muck",          # Germany
    "MvZelma",       # Hungary
    "Naturastar",    # Germany ~1999
    "Patras",        #
    "Sultan",        #
})

ERA_LABEL = {"old": 0, "modern": 1}
ERA_NAME  = {v: k for k, v in ERA_LABEL.items()}


def _fetch_tsv(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def _build_sample_meta(
    assay_rows: list[dict], sample_rows: list[dict]
) -> dict[str, dict]:
    """Return {cdf_stem: {label, group, variety, era, location}} for wheat samples."""
    # sample sheet: Sample Name → factor values
    sample_meta: dict[str, dict] = {}
    for row in sample_rows:
        name     = row.get("Sample Name", "").strip()
        genotype = row.get("Factor Value[Genotype]", "").strip()
        location = row.get("Factor Value[Location]", "").strip()
        if not genotype or not name:
            continue
        # Genotype format: "PGL NNN; VarietyName"
        m = re.match(r"PGL\s+(\d+);\s*(\S+)", genotype)
        if not m:
            continue
        pgl_num  = int(m.group(1))
        variety  = m.group(2)
        era      = "old" if variety in OLD_VARIETIES else "modern"
        sample_meta[name] = {
            "pgl_id":   f"PGL_{pgl_num:03d}",
            "variety":  variety,
            "era":      era,
            "location": location,
            "label":    ERA_LABEL[era],
        }

    # assay sheet: Sample Name → CDF file path
    cdf_meta: dict[str, dict] = {}
    for row in assay_rows:
        sample_name = row.get("Sample Name", "").strip()
        raw_file    = row.get("Raw Spectral Data File", "").strip()
        if sample_name not in sample_meta or not raw_file:
            continue
        cdf_stem = Path(raw_file).stem
        meta     = sample_meta[sample_name]
        cdf_meta[cdf_stem] = {
            "label":    meta["label"],
            "group":    meta["pgl_id"],
            "variety":  meta["variety"],
            "era":      meta["era"],
            "location": meta["location"],
        }
    return cdf_meta


def _get_filenames() -> list[str]:
    """List CDF filenames from the EBI HTTP FTP index."""
    with urllib.request.urlopen(FILES_INDEX_URL, timeout=30) as r:
        html = r.read().decode()
    return re.findall(r'href="(\d{6}_\d{2}\.cdf)"', html)


def _download(fname: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HTTPS_BASE + fname
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _read_cdf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with netcdf_file(str(path), "r", mmap=False) as f:
        rt       = f.variables["scan_acquisition_time"].data.copy()
        scan_idx = f.variables["scan_index"].data.copy()
        pt_count = f.variables["point_count"].data.copy()
        mass_vals = f.variables["mass_values"].data.copy()
        int_vals  = f.variables["intensity_values"].data.copy()
        tic_raw   = f.variables["total_intensity"].data.copy()

    n_scans = len(rt)
    spectra = np.zeros((n_scans, MZ_MAX), dtype=np.float32)
    for i in range(n_scans):
        s = int(scan_idx[i])
        e = s + int(pt_count[i])
        mz        = np.round(mass_vals[s:e]).astype(np.int32)
        intensity = int_vals[s:e].astype(np.float32)
        valid     = (mz >= 0) & (mz < MZ_MAX)
        np.add.at(spectra[i], mz[valid], intensity[valid])

    return spectra, tic_raw.astype(np.float32)


def _bin_chromatogram(spectra: np.ndarray, tic: np.ndarray, n_bins: int) -> np.ndarray:
    n_scans  = len(tic)
    bin_size = n_scans / n_bins
    result   = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        start = int(i * bin_size)
        end   = min(int((i + 1) * bin_size), n_scans)
        if start >= end:
            continue
        best      = start + int(np.argmax(tic[start:end]))
        result[i] = spectra[best]
    return result


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr   = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def main(
    raw_dir: Path, output_dir: Path, n_bins: int, workers: int, no_download: bool
) -> None:
    print("Fetching metadata from MetaboLights …")
    assay_rows  = _fetch_tsv(ASSAY_URL)
    sample_rows = _fetch_tsv(SAMPLE_URL)
    cdf_meta    = _build_sample_meta(assay_rows, sample_rows)
    print(f"  Metadata for {len(cdf_meta)} wheat samples loaded")
    old_n = sum(1 for m in cdf_meta.values() if m["label"] == 0)
    mod_n = sum(1 for m in cdf_meta.values() if m["label"] == 1)
    print(f"  Old: {old_n}  |  Modern: {mod_n}")

    if not no_download:
        print("\nFetching file list from EBI FTP …")
        all_fnames = sorted(_get_filenames())
        wanted     = sorted(fn for fn in all_fnames if Path(fn).stem in cdf_meta)
        print(f"  {len(all_fnames)} CDF files total; downloading {len(wanted)} wheat samples …")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download, fn, raw_dir / fn): fn for fn in wanted}
            done = 0
            for fut in as_completed(futures):
                fn = futures[fut]
                done += 1
                if fut.exception():
                    print(f"  ERROR {fn}: {fut.exception()}")
                elif done % 20 == 0 or done == len(wanted):
                    print(f"  Downloaded {done}/{len(wanted)}")

    chroma_dir = output_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    cdf_files = sorted(raw_dir.glob("*.cdf"))
    print(f"\nFound {len(cdf_files)} CDF files in {raw_dir}")
    print(f"Preprocessing to {n_bins}-bin chromatograms …")

    X_list, y_list, groups_list, ids_list = [], [], [], []
    skipped = 0

    for cdf_path in cdf_files:
        meta = cdf_meta.get(cdf_path.stem)
        if meta is None:
            skipped += 1
            continue

        out_npz = chroma_dir / (cdf_path.stem + ".npz")
        if out_npz.exists():
            chroma   = np.load(out_npz)["chroma"].astype(np.float32)
            sum_spec = chroma.sum(axis=0)
        else:
            spectra, tic = _read_cdf(cdf_path)
            chroma       = _bin_chromatogram(spectra, tic, n_bins)
            chroma       = _sqrt_l2(chroma)
            np.savez_compressed(out_npz, chroma=chroma)
            sum_spec     = spectra.sum(axis=0)

        sum_spec_norm = _sqrt_l2(sum_spec[None])[0]

        X_list.append(sum_spec_norm)
        y_list.append(meta["label"])
        groups_list.append(meta["group"])
        ids_list.append(cdf_path.stem)

        if len(X_list) % 20 == 0:
            print(f"  {len(X_list)}/{len(cdf_files) - skipped} processed …")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    (output_dir / "groups.txt").write_text("\n".join(groups_list))
    (output_dir / "sample_ids.txt").write_text("\n".join(ids_list))

    if skipped:
        print(f"  Skipped {skipped} non-wheat CDF files (blanks/standards/QC)")

    classes, counts = np.unique(y, return_counts=True)
    print(f"\nSaved {len(X)} samples → {output_dir}")
    print(f"X shape: {X.shape}  |  dtype: {X.dtype}")
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({ERA_NAME[c]}): {n} samples")
    cultivars = sorted(set(groups_list))
    print(f"Cultivars: {len(cultivars)}  |  Chromatograms: {len(list(chroma_dir.glob('*.npz')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir",     default="data/mtbls1187/raw",  type=Path)
    ap.add_argument("--output-dir",  default="data/mtbls1187",      type=Path)
    ap.add_argument("--n-bins",      default=N_BINS_DEFAULT,        type=int)
    ap.add_argument("--workers",     default=8,                     type=int)
    ap.add_argument("--no-download", action="store_true",
                    help="Skip downloading; process already-present CDF files")
    args = ap.parse_args()
    main(args.raw_dir, args.output_dir, args.n_bins, args.workers, args.no_download)
