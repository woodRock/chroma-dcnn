"""
Step 1: Download EI-MS spectra from MoNA and MassBank EU.

Outputs (in data/pretraining/raw/):
  mona_gcms.json        — MoNA GC-MS bulk export
  massbank_records/     — MassBank EU .txt record files
"""

import argparse
import json
from pathlib import Path

from chroma_dcnn.data.download import (
    deduplicate_records,
    download_massbank_eu,
    download_mona,
    parse_massbank_records,
    parse_mona_records,
)


def _assert_nonzero_yield(source: str, n_raw: int, n_kept: int) -> None:
    """Fail loudly rather than silently proceeding with a source that parsed to nothing.

    This is exactly the shape of a real regression we hit: MassBank's InChIKey parsing
    broke, every one of 139k raw records was silently dropped, and the pipeline happily
    produced a 100%-MoNA corpus with no error or warning anywhere. 0 kept from >0 raw is
    never correct for either source (both hold tens of thousands of EI-GC-MS records) and
    almost always means a parsing/filtering regression, not a legitimately empty source.
    """
    if n_raw > 0 and n_kept == 0:
        raise RuntimeError(
            f"{source}: {n_raw} raw records found but 0 were kept after parsing/filtering. "
            f"This almost always means a parsing regression (field name, filter logic), not "
            f"a legitimately empty source — fix the parser rather than ignoring this."
        )


def main(raw_dir: str, force: bool) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1a: MoNA ===")
    mona_json = download_mona(raw_dir, force=force)
    mona_records = list(parse_mona_records(mona_json)) if mona_json else []
    if mona_json:
        with open(mona_json) as f:
            n_raw_mona = len(json.load(f))
        _assert_nonzero_yield("MoNA", n_raw_mona, len(mona_records))
    else:
        print("  No MoNA records — continuing with MassBank only.")

    print("\n=== Step 1b: MassBank EU ===")
    mb_dir = download_massbank_eu(raw_dir, force=force)
    n_raw_mb = sum(1 for _ in mb_dir.rglob("*.txt"))
    mb_records = list(parse_massbank_records(mb_dir))
    _assert_nonzero_yield("MassBank", n_raw_mb, len(mb_records))

    print("\n=== Step 1c: Deduplication ===")
    all_records = mona_records + mb_records
    unique_records = deduplicate_records(all_records)

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
