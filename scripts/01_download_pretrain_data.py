"""
Step 1: Download EI-MS spectra from MoNA and MassBank EU.

Outputs (in data/pretraining/raw/):
  mona_gcms.json        — MoNA GC-MS bulk export
  massbank_records/     — MassBank EU .txt record files
"""

import argparse
from pathlib import Path

from chroma_dcnn.data.download import (
    deduplicate_records,
    download_massbank_eu,
    download_mona,
    parse_massbank_records,
    parse_mona_records,
)


def main(raw_dir: str, force: bool) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1a: MoNA ===")
    mona_json = download_mona(raw_dir, force=force)
    mona_records = list(parse_mona_records(mona_json)) if mona_json else []
    if not mona_records:
        print("  No MoNA records — continuing with MassBank only.")

    print("\n=== Step 1b: MassBank EU ===")
    mb_dir = download_massbank_eu(raw_dir, force=force)
    mb_records = list(parse_massbank_records(mb_dir))

    print("\n=== Step 1c: Deduplication ===")
    all_records = mona_records + mb_records
    unique_records = deduplicate_records(all_records)

    import json
    out_path = raw_dir / "combined_ei_spectra.json"
    with open(out_path, "w") as f:
        json.dump(unique_records, f)
    print(f"\nSaved {len(unique_records)} deduplicated records → {out_path}")
    print(f"\nExpected scale check: {len(unique_records)} spectra "
          f"(target: 15,000–30,000 after filtering)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/pretraining/raw")
    ap.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = ap.parse_args()
    main(args.raw_dir, args.force)
