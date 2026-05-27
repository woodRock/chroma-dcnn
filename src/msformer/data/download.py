"""
Download EI-MS spectra from MoNA and MassBank EU.

MoNA provides bulk JSON exports at:
  https://mona.fiehnlab.ucdavis.edu/downloads
  File: MoNA-export-GC-MS_Spectra.json.zip  (~200 MB)

MassBank EU records live on GitHub (tagged releases as zip):
  https://github.com/MassBank/MassBank-data/releases
  Records are plain-text .txt files in MassBank record format.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

MONA_BASE = "https://mona.fiehnlab.ucdavis.edu"
MONA_DOWNLOADS_API = f"{MONA_BASE}/rest/downloads"
MONA_SEARCH_API = f"{MONA_BASE}/rest/spectra/search"
# Public bulk export — no auth required, served directly (not via /rest/)
MONA_PUBLIC_EXPORT_URL = (
    "https://mona.fiehnlab.ucdavis.edu/downloads/retrieve/"
    "MoNA-export-GC-MS_Spectra.json.zip"
)
MASSBANK_RELEASE_API = (
    "https://api.github.com/repos/MassBank/MassBank-data/releases/latest"
)

# Mimic a browser so MoNA doesn't block the request
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
}

_EI_INSTRUMENT_PATTERNS = re.compile(
    r"GC-EI|GC/EI|EI-B|EI-EBEB|EI-QQEE|GC-MS|gas chromatograph", re.IGNORECASE
)
_MZ_INT_PATTERN = re.compile(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# MoNA
# ---------------------------------------------------------------------------

def download_mona(output_dir: str | Path, force: bool = False) -> Path | None:
    """
    Download MoNA GC-MS spectra.  Returns None if MoNA is unreachable.

    Strategy (tried in order):
      1. Public bulk export (no auth required).
      2. Discover URL from downloads list API.
      3. Paginated search API.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mona_gcms.json"

    if json_path.exists() and not force:
        print(f"MoNA JSON already present at {json_path}")
        return json_path

    # Strategy 1: direct public export URL (no auth)
    try:
        print(f"Trying MoNA public export URL …")
        _download_with_progress(MONA_PUBLIC_EXPORT_URL, output_dir / "mona_gcms.json.zip")
        with zipfile.ZipFile(output_dir / "mona_gcms.json.zip") as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            with zf.open(names[0]) as src, open(json_path, "wb") as dst:
                dst.write(src.read())
        (output_dir / "mona_gcms.json.zip").unlink()
        return json_path
    except Exception as e:
        print(f"  Public export failed: {e}")

    # Strategy 2: downloads list API
    try:
        return _download_mona_bulk(output_dir, json_path)
    except Exception as e:
        print(f"  Downloads list API failed: {e}")

    # Strategy 3: paginated search API
    try:
        return _download_mona_search(output_dir, json_path)
    except Exception as e:
        print(f"  Search API failed: {e}")

    print(
        "\n⚠  MoNA is unreachable from this host. Skipping MoNA.\n"
        "   To download manually: visit https://mona.fiehnlab.ucdavis.edu/downloads\n"
        "   and place the extracted JSON at:\n"
        f"   {json_path}\n"
        "   Then re-run this script — it will detect the file and skip the download.\n"
        "   Proceeding with MassBank only (should still yield ~10k–20k EI-MS spectra)."
    )
    return None


def _download_mona_bulk(output_dir: Path, json_path: Path) -> Path:
    """Discover and fetch the pre-built GC-MS JSON export from MoNA's downloads API."""
    print("Querying MoNA downloads list …")
    resp = requests.get(MONA_DOWNLOADS_API, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    downloads = resp.json()

    # Find a GC-MS JSON export entry
    gcms_entry = None
    for entry in downloads:
        label = entry.get("label", "") or entry.get("name", "") or ""
        filename = entry.get("filename", "") or entry.get("name", "") or ""
        if ("GC" in label or "GC" in filename) and (
            "json" in filename.lower() or "json" in label.lower()
        ):
            gcms_entry = entry
            break

    if gcms_entry is None:
        # Print available entries to help diagnose
        labels = [e.get("label", e.get("filename", str(e))) for e in downloads[:10]]
        raise RuntimeError(
            f"No GC-MS JSON export found in MoNA downloads list.\n"
            f"Available entries (first 10): {labels}\n"
            f"Check {MONA_DOWNLOADS_API} manually and set the URL."
        )

    # Build retrieve URL — try common patterns
    filename = gcms_entry.get("filename", gcms_entry.get("name", ""))
    retrieve_url = f"{MONA_BASE}/rest/downloads/retrieve/{filename}"
    print(f"Found export: {filename}")

    zip_path = output_dir / filename
    _download_with_progress(retrieve_url, zip_path)

    if filename.endswith(".zip"):
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            if not names:
                raise RuntimeError(f"No JSON found inside {zip_path}")
            with zf.open(names[0]) as src, open(json_path, "wb") as dst:
                dst.write(src.read())
        zip_path.unlink()
    else:
        zip_path.rename(json_path)

    return json_path


def _download_mona_search(output_dir: Path, json_path: Path) -> Path:
    """
    Download EI-MS GC spectra via the MoNA paginated search API.

    Queries for records where instrument type metadata matches GC variants.
    Paginates until the API returns an empty page.
    """
    # RSQL query: instrument type metadata value contains "GC"
    query = 'metaData=q=\'name=="instrument type" and value=~"GC"\''
    page_size = 100
    all_records: list[dict] = []
    page = 0

    print(f"Downloading MoNA GC-MS spectra via search API (page_size={page_size}) …")
    while True:
        resp = requests.get(
            MONA_SEARCH_API,
            params={"query": query, "size": page_size, "page": page},
            headers=_HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_records.extend(batch)
        print(f"  Page {page}: {len(batch)} records  (total so far: {len(all_records)})", end="\r")
        page += 1

    print(f"\nSearch API: downloaded {len(all_records)} raw MoNA records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f)

    return json_path


def parse_mona_records(json_path: str | Path) -> Iterator[dict]:
    """
    Yield dicts with keys: inchikey, smiles, peaks, instrument, splash.

    peaks is a list of (mz: float, intensity: float) tuples.
    Records with no InChIKey, no valid EI instrument tag, or m/z > 1000 Da
    are dropped silently.
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as fh:
        records = json.load(fh)

    print(f"Loaded {len(records)} raw MoNA records")
    kept = 0
    for rec in records:
        parsed = _parse_mona_record(rec)
        if parsed is not None:
            kept += 1
            yield parsed

    print(f"Retained {kept} EI-GC-MS records from MoNA after filtering")


def _parse_mona_record(rec: dict) -> dict | None:
    # ---- instrument type filter ----
    meta = {m["name"].lower(): m["value"] for m in rec.get("metaData", [])}
    instrument = meta.get("instrument type", "") or meta.get("instrument", "")
    if not _EI_INSTRUMENT_PATTERNS.search(instrument):
        return None

    # ---- InChIKey ----
    compound = rec.get("compound", [{}])
    if isinstance(compound, list):
        compound = compound[0] if compound else {}
    inchikey = compound.get("inchiKey", "")
    if not inchikey or len(inchikey) < 14:
        return None

    # ---- SMILES (optional) ----
    smiles = ""
    for m in compound.get("metaData", []):
        if m.get("name", "").lower() == "smiles":
            smiles = m.get("value", "")
            break

    # ---- parse peak string ----
    spectrum_str = rec.get("spectrum", "")
    peaks = _parse_peak_string(spectrum_str)
    if not peaks:
        return None

    return {
        "inchikey": inchikey,
        "smiles": smiles,
        "peaks": peaks,
        "instrument": instrument,
        "splash": rec.get("splash", {}).get("splash", ""),
        "source": "mona",
    }


def _parse_peak_string(spectrum_str: str) -> list[tuple[float, float]]:
    peaks = []
    for match in _MZ_INT_PATTERN.finditer(spectrum_str):
        mz, intensity = float(match.group(1)), float(match.group(2))
        if mz <= 1000:
            peaks.append((mz, intensity))
    return peaks


# ---------------------------------------------------------------------------
# MassBank EU
# ---------------------------------------------------------------------------

def download_massbank_eu(output_dir: str | Path, force: bool = False) -> Path:
    """
    Download the latest MassBank-data release zip from GitHub.

    Returns the path to the directory containing extracted .txt record files.
    """
    output_dir = Path(output_dir)
    records_dir = output_dir / "massbank_records"

    if records_dir.exists() and any(records_dir.rglob("*.txt")) and not force:
        n = sum(1 for _ in records_dir.rglob("*.txt"))
        print(f"MassBank records already present: {n} .txt files in {records_dir}")
        return records_dir

    print("Fetching MassBank-data latest release info …")
    resp = requests.get(MASSBANK_RELEASE_API, timeout=30)
    resp.raise_for_status()
    release = resp.json()
    zip_url = release["zipball_url"]
    tag = release["tag_name"]
    print(f"Latest release: {tag}")

    zip_path = output_dir / f"massbank_{tag}.zip"
    _download_with_progress(zip_url, zip_path)

    print("Extracting MassBank records …")
    records_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
        for member in tqdm(txt_members, desc="Extracting"):
            data = zf.read(member)
            dest = records_dir / Path(member).name
            dest.write_bytes(data)

    zip_path.unlink()
    n = sum(1 for _ in records_dir.rglob("*.txt"))
    print(f"Extracted {n} MassBank record files")
    return records_dir


def parse_massbank_records(records_dir: str | Path) -> Iterator[dict]:
    """
    Parse MassBank .txt records, yielding same dict schema as parse_mona_records.
    Filters to EI ionisation mode only.
    """
    records_dir = Path(records_dir)
    txt_files = list(records_dir.rglob("*.txt"))
    print(f"Parsing {len(txt_files)} MassBank record files …")

    kept = 0
    for txt_path in tqdm(txt_files, desc="Parsing MassBank"):
        parsed = _parse_massbank_txt(txt_path)
        if parsed is not None:
            kept += 1
            yield parsed

    print(f"Retained {kept} EI records from MassBank")


def _parse_massbank_txt(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    fields: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    in_peak_block = False

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("//") or not line:
            continue
        if line == "PK$PEAK: m/z int. rel.int.":
            in_peak_block = True
            continue
        if in_peak_block:
            if line.startswith("//"):
                in_peak_block = False
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz, intensity = float(parts[0]), float(parts[1])
                    if mz <= 1000:
                        peaks.append((mz, intensity))
                except ValueError:
                    pass
            continue

        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    # Filter: must be EI ionisation
    ac_ms = fields.get("AC$MASS_SPECTROMETRY", "")
    ion_mode_line = fields.get("AC$MASS_SPECTROMETRY: ION_MODE", ac_ms)
    ms_type = fields.get("AC$MASS_SPECTROMETRY: MS_TYPE", "")

    # Check for EI anywhere in fields
    full_text_upper = text.upper()
    if "IONIZATION: EI" not in full_text_upper and "ION_MODE: POSITIVE" not in full_text_upper:
        if "EI" not in full_text_upper:
            return None

    inchikey = fields.get("CH$INCHI_KEY", "") or fields.get("CH$INCHIKEY", "")
    if not inchikey or len(inchikey) < 14:
        return None

    if not peaks:
        return None

    return {
        "inchikey": inchikey,
        "smiles": fields.get("CH$SMILES", ""),
        "peaks": peaks,
        "instrument": fields.get("AC$INSTRUMENT", ""),
        "splash": fields.get("PK$SPLASH", ""),
        "source": "massbank",
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_records(records: list[dict]) -> list[dict]:
    """
    Keep one record per InChIKey (prefer MoNA over MassBank, then highest
    peak count as a proxy for spectrum quality).
    """
    from collections import defaultdict

    by_inchikey: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_inchikey[rec["inchikey"]].append(rec)

    deduped = []
    for inchikey, group in by_inchikey.items():
        # prefer mona source; then most peaks
        group.sort(key=lambda r: (r["source"] != "mona", -len(r["peaks"])))
        deduped.append(group[0])

    print(f"Deduplication: {len(records)} → {len(deduped)} unique InChIKeys")
    return deduped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=120, headers=_HEADERS)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))
