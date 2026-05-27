"""
Preprocess olive oil GC-IMS .mea files from zip → numpy arrays.

Streams each .mea file from the zip, processes it in memory via a temp file,
and discards it — never extracts the full 6.9 GB to disk.

Reads:  data/olive_oil/Olive oil geography by GC-IMS analysis.zip
Writes: data/olive_oil/X_sum.npy   (N, 1000)
        data/olive_oil/X_apex.npy  (N, 1000)
        data/olive_oil/y.npy       (N,)
        data/olive_oil/sample_ids.txt

Expected zip structure:
  .../IMS_Olive_Oil_Geography/<Country>/<SampleID>/<timestamp>.mea

Labels: Spain=0, Italy=1, Greece=2
"""

import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import ims
import numpy as np
from tqdm import tqdm

LABEL_MAP = {"spain": 0, "italy": 1, "greece": 2}
TARGET_DIM = 1000
ZIP_PATH = Path("data/olive_oil/Olive oil geography by GC-IMS analysis.zip")
OUT_DIR = Path("data/olive_oil")


def reduce_sum(values: np.ndarray) -> np.ndarray:
    """Sum over drift-time axis → retention-time profile."""
    return values.sum(axis=1).astype(np.float32)


def reduce_apex(values: np.ndarray) -> np.ndarray:
    """Extract retention profile at drift-time slice with max total intensity (RIP apex)."""
    rip_idx = values.sum(axis=0).argmax()
    return values[:, rip_idx].astype(np.float32)


def bin_to_dim(vec: np.ndarray, target: int) -> np.ndarray:
    src_x = np.linspace(0, 1, len(vec))
    tgt_x = np.linspace(0, 1, target)
    return np.interp(tgt_x, src_x, vec).astype(np.float32)


def normalise(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.sqrt(np.maximum(vec, 0.0))
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > eps else vec


def infer_label(path_parts: list[str]) -> int | None:
    for part in path_parts:
        if part.lower() in LABEL_MAP:
            return LABEL_MAP[part.lower()]
    return None


def main() -> None:
    print(f"Reading from {ZIP_PATH}")

    X_sum_rows, X_apex_rows, ys, sample_ids = [], [], [], []

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        mea_names = [n for n in zf.namelist() if n.endswith(".mea")]
        print(f"Found {len(mea_names)} .mea files")

        for member in tqdm(mea_names, desc="Processing .mea files"):
            parts = member.replace("\\", "/").split("/")
            label = infer_label(parts)
            if label is None:
                print(f"  Skipping (no label): {member}")
                continue

            # Sample ID = country/samplefolder  e.g. "Italy/O-017"
            try:
                country_idx = next(
                    i for i, p in enumerate(parts) if p.lower() in LABEL_MAP
                )
                sample_id = f"{parts[country_idx]}/{parts[country_idx + 1]}"
            except (StopIteration, IndexError):
                sample_id = Path(member).stem

            # Extract to temp file, read, discard
            with tempfile.NamedTemporaryFile(suffix=".mea", delete=False) as tmp:
                tmp.write(zf.read(member))
                tmp_path = tmp.name

            try:
                spectrum = ims.Spectrum.read_mea(tmp_path)
                values = spectrum.values  # 2D array (ret_times × drift_times)
            except Exception as e:
                print(f"  Failed to read {member}: {e}")
                Path(tmp_path).unlink(missing_ok=True)
                continue
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            vec_sum = normalise(bin_to_dim(reduce_sum(values), TARGET_DIM))
            vec_apex = normalise(bin_to_dim(reduce_apex(values), TARGET_DIM))

            X_sum_rows.append(vec_sum)
            X_apex_rows.append(vec_apex)
            ys.append(label)
            sample_ids.append(f"{sample_id}/{Path(member).stem}")

    if not X_sum_rows:
        raise RuntimeError("No spectra were loaded — check zip structure.")

    X_sum = np.stack(X_sum_rows)
    X_apex = np.stack(X_apex_rows)
    y = np.array(ys, dtype=np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_sum.npy", X_sum)
    np.save(OUT_DIR / "X_apex.npy", X_apex)
    np.save(OUT_DIR / "y.npy", y)
    (OUT_DIR / "sample_ids.txt").write_text("\n".join(sample_ids))

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  X_sum:  {X_sum.shape}")
    print(f"  X_apex: {X_apex.shape}")
    print(f"  y:      {y.shape}")

    classes, counts = np.unique(y, return_counts=True)
    inv = {v: k for k, v in LABEL_MAP.items()}
    for c, n in zip(classes, counts):
        print(f"  Class {c} ({inv[c]}): {n} spectra")


if __name__ == "__main__":
    main()
