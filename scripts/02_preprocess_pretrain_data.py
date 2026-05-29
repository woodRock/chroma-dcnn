"""
Step 2: Preprocess raw spectra → normalised HDF5.

Reads data/pretraining/raw/combined_ei_spectra.json
Writes data/pretraining/spectra.h5
"""

import argparse
import json
from pathlib import Path

from chroma_dcnn.data.preprocess import build_hdf5, report_dataset_stats


def main(raw_json: str, output_h5: str, mz_max: int, val_fraction: float) -> None:
    raw_json = Path(raw_json)
    if not raw_json.exists():
        raise FileNotFoundError(
            f"{raw_json} not found — run 01_download_pretrain_data.py first"
        )

    print(f"Loading records from {raw_json} …")
    with open(raw_json) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    output_h5 = Path(output_h5)
    build_hdf5(records, output_h5, val_fraction=val_fraction, mz_max=mz_max)
    report_dataset_stats(output_h5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-json", default="data/pretraining/raw/combined_ei_spectra.json")
    ap.add_argument("--output-h5", default="data/pretraining/spectra.h5")
    ap.add_argument("--mz-max", type=int, default=1000)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    args = ap.parse_args()
    main(args.raw_json, args.output_h5, args.mz_max, args.val_fraction)
