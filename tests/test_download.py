"""MoNA / MassBank download and parsing, with the network mocked out."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from chroma_dcnn.data import download as dl

# ---------------------------------------------------------------------------
# Fake network
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, content=b"", json_data=None, status=200):
        self.content = content
        self._json = json_data
        self.status = status
        self.headers = {"content-length": str(len(content))}

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _mona_record(instrument="GC-EI-TOF", inchikey="ABCDEFGHIJKLMN-UHFFFAOYSA-N",
                 spectrum="41:100 43.5:50 1200:9"):
    return {
        "metaData": [{"name": "Instrument Type", "value": instrument}],
        "compound": [{"inchiKey": inchikey,
                      "metaData": [{"name": "SMILES", "value": "CCO"}]}],
        "spectrum": spectrum,
        "splash": {"splash": "splash10-x"},
    }


# ---------------------------------------------------------------------------
# MoNA parsing
# ---------------------------------------------------------------------------

def test_parse_peak_string_drops_above_1000():
    assert dl._parse_peak_string("41:100 43.5:50 1200:9") == [(41.0, 100.0), (43.5, 50.0)]


def test_parse_mona_record_valid():
    rec = dl._parse_mona_record(_mona_record())
    assert rec["inchikey"].startswith("ABCDEFGHIJKLMN") and rec["smiles"] == "CCO"
    assert rec["peaks"] == [(41.0, 100.0), (43.5, 50.0)]
    assert rec["source"] == "mona" and rec["splash"] == "splash10-x"


@pytest.mark.parametrize("kwargs", [
    {"instrument": "LC-ESI-QTOF"},
    {"inchikey": ""},
    {"inchikey": "SHORT"},
    {"spectrum": ""},
])
def test_parse_mona_record_filters(kwargs):
    assert dl._parse_mona_record(_mona_record(**kwargs)) is None


def test_parse_mona_record_compound_as_dict_and_empty_list():
    rec = _mona_record()
    rec["compound"] = rec["compound"][0]
    assert dl._parse_mona_record(rec) is not None
    rec["compound"] = []
    assert dl._parse_mona_record(rec) is None


def test_parse_mona_records_from_file_and_parts(tmp_path):
    good, bad = _mona_record(), _mona_record(instrument="LC-ESI")
    (tmp_path / "m.json").write_text(json.dumps([good, bad]))
    assert len(list(dl.parse_mona_records(tmp_path / "m.json"))) == 1

    (tmp_path / "mona_gcms_part1.json").write_text(json.dumps([good]))
    (tmp_path / "mona_gcms_part2.json").write_text(json.dumps([good, bad]))
    assert len(list(dl.parse_mona_records(tmp_path / "missing.json"))) == 2


def test_parse_mona_records_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(dl.parse_mona_records(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# MassBank parsing
# ---------------------------------------------------------------------------

_MB_PARSER_FORMAT = """\
ACCESSION: TEST0001
CH$SMILES: CCO
CH$INCHI_KEY: LFQSCWFLJHTTHZ-UHFFFAOYSA-N
AC$INSTRUMENT: GC-EI-TOF
AC$MASS_SPECTROMETRY: IONIZATION EI
PK$SPLASH: splash10-y
PK$PEAK: m/z int. rel.int.
  31.0 100 999
  45.0 50 500
  bad line
  1500.0 1 1
//
"""

# Real MassBank format: the InChIKey lives on a CH$LINK line.
_MB_REAL_FORMAT = """\
ACCESSION: MSBNK-Test-TE000001
CH$NAME: Ethanol
CH$SMILES: CCO
CH$LINK: CAS 64-17-5
CH$LINK: INCHIKEY LFQSCWFLJHTTHZ-UHFFFAOYSA-N
AC$INSTRUMENT: GC-EI-TOF
AC$INSTRUMENT_TYPE: GC-EI-TOF
AC$MASS_SPECTROMETRY: MS_TYPE MS
AC$MASS_SPECTROMETRY: ION_MODE POSITIVE
AC$MASS_SPECTROMETRY: IONIZATION EI
PK$PEAK: m/z int. rel.int.
  31.0 100 999
//
"""

_MB_ESI = """\
ACCESSION: MSBNK-Test-TE000002
CH$NAME: Some protein fragment
CH$LINK: INCHIKEY LFQSCWFLJHTTHZ-UHFFFAOYSA-N
CH$INCHI_KEY: LFQSCWFLJHTTHZ-UHFFFAOYSA-N
AC$INSTRUMENT_TYPE: LC-ESI-QTOF
AC$MASS_SPECTROMETRY: MS_TYPE MS2
AC$MASS_SPECTROMETRY: ION_MODE POSITIVE
AC$MASS_SPECTROMETRY: IONIZATION ESI
PK$PEAK: m/z int. rel.int.
  120.0 100 999
//
"""


def test_parse_massbank_txt(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text(_MB_PARSER_FORMAT)
    rec = dl._parse_massbank_txt(p)
    assert rec["inchikey"] == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert rec["peaks"] == [(31.0, 100.0), (45.0, 50.0)]
    assert rec["source"] == "massbank" and rec["smiles"] == "CCO"


@pytest.mark.parametrize(("text", "why"), [
    ("ACCESSION: X\nPK$PEAK: m/z int. rel.int.\n  31.0 100 999\n//\n", "no InChIKey"),
    ("CH$INCHI_KEY: LFQSCWFLJHTTHZ-UHFFFAOYSA-N\nAC$X: EI\n", "no peaks"),
    ("CH$INCHI_KEY: LFQSCWFLJHTTHZ-UHFFFAOYSA-N\nPK$PEAK: m/z int. rel.int.\n  31 1 1\n",
     "no 'EI' anywhere"),
])
def test_parse_massbank_txt_rejects(tmp_path, text, why):
    p = tmp_path / "r.txt"
    p.write_text(text)
    assert dl._parse_massbank_txt(p) is None, why


def test_parse_massbank_txt_unreadable(tmp_path):
    assert dl._parse_massbank_txt(tmp_path / "missing.txt") is None


@pytest.mark.xfail(strict=True, reason="Known bug: parser reads CH$INCHI_KEY, but MassBank "
                   "stores InChIKeys as 'CH$LINK: INCHIKEY …', so real records are dropped")
def test_parse_massbank_real_inchikey_format(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text(_MB_REAL_FORMAT)
    assert dl._parse_massbank_txt(p) is not None


@pytest.mark.xfail(strict=True, reason="Known bug: EI filter accepts any record containing "
                   "'EI' (e.g. 'protEIn'), so LC-ESI MS/MS spectra pass")
def test_parse_massbank_rejects_esi(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text(_MB_ESI)
    assert dl._parse_massbank_txt(p) is None


def test_parse_massbank_records(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "r1.txt").write_text(_MB_PARSER_FORMAT)
    (tmp_path / "r2.txt").write_text("junk")
    assert len(list(dl.parse_massbank_records(tmp_path))) == 1


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def test_deduplicate_prefers_mona_then_most_peaks():
    recs = [
        {"inchikey": "A", "source": "massbank", "peaks": [1, 2, 3]},
        {"inchikey": "A", "source": "mona", "peaks": [1]},
        {"inchikey": "B", "source": "massbank", "peaks": [1]},
        {"inchikey": "B", "source": "massbank", "peaks": [1, 2]},
    ]
    out = {r["inchikey"]: r for r in dl.deduplicate_records(recs)}
    assert out["A"]["source"] == "mona"
    assert len(out["B"]["peaks"]) == 2


# ---------------------------------------------------------------------------
# download_mona strategies
# ---------------------------------------------------------------------------

def test_download_mona_existing_json(tmp_path):
    (tmp_path / "mona_gcms.json").write_text("[]")
    assert dl.download_mona(tmp_path) == tmp_path / "mona_gcms.json"


def test_download_mona_reassembles_parts(tmp_path):
    (tmp_path / "mona_gcms_part1.json").write_text(json.dumps([{"a": 1}]))
    (tmp_path / "mona_gcms_part2.json").write_text(json.dumps([{"b": 2}]))
    out = dl.download_mona(tmp_path)
    assert json.loads(out.read_text()) == [{"a": 1}, {"b": 2}]


def test_download_mona_extracts_existing_zip(tmp_path):
    (tmp_path / "x.zip").write_bytes(_zip_bytes({"inner.json": b"[1]"}))
    assert json.loads(dl.download_mona(tmp_path).read_text()) == [1]


def test_download_mona_public_export(tmp_path, monkeypatch):
    payload = _zip_bytes({"export.json": b"[2]"})
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(payload))
    out = dl.download_mona(tmp_path, force=True)
    assert json.loads(out.read_text()) == [2]
    assert not (tmp_path / "mona_gcms.json.zip").exists()


def _router(routes):
    def get(url, *a, **k):
        for prefix, resp in routes:
            if url.startswith(prefix):
                return resp(k) if callable(resp) else resp
        return FakeResponse(status=404)
    return get


def test_download_mona_falls_back_to_downloads_api(tmp_path, monkeypatch):
    listing = [{"label": "LC-MS", "filename": "lc.json"},
               {"label": "GC-MS Spectra", "filename": "gc.json.zip"}]
    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MONA_PUBLIC_EXPORT_URL, FakeResponse(status=503)),
        (dl.MONA_DOWNLOADS_API + "/retrieve/", FakeResponse(_zip_bytes({"a.json": b"[3]"}))),
        (dl.MONA_DOWNLOADS_API, FakeResponse(json_data=listing)),
    ]))
    assert json.loads(dl.download_mona(tmp_path, force=True).read_text()) == [3]


def test_downloads_api_plain_json_is_renamed(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MONA_DOWNLOADS_API + "/retrieve/", FakeResponse(b"[4]")),
        (dl.MONA_DOWNLOADS_API, FakeResponse(json_data=[{"label": "GC json",
                                                          "filename": "gc.json"}])),
    ]))
    out = dl._download_mona_bulk(tmp_path, tmp_path / "out.json")
    assert json.loads(out.read_text()) == [4]


def test_downloads_api_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MONA_DOWNLOADS_API, FakeResponse(json_data=[{"label": "LC", "filename": "x"}])),
    ]))
    with pytest.raises(RuntimeError, match="No GC-MS JSON"):
        dl._download_mona_bulk(tmp_path, tmp_path / "out.json")

    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MONA_DOWNLOADS_API + "/retrieve/", FakeResponse(_zip_bytes({"a.txt": b"x"}))),
        (dl.MONA_DOWNLOADS_API,
         FakeResponse(json_data=[{"label": "GC", "filename": "g.json.zip"}])),
    ]))
    with pytest.raises(RuntimeError, match="No JSON found"):
        dl._download_mona_bulk(tmp_path, tmp_path / "out.json")


def test_download_mona_falls_back_to_search(tmp_path, monkeypatch):
    def search(kwargs):
        page = kwargs["params"]["page"]
        return FakeResponse(json_data=[{"p": page}] if page < 2 else [])

    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MONA_PUBLIC_EXPORT_URL, FakeResponse(status=503)),
        (dl.MONA_SEARCH_API, search),
        (dl.MONA_DOWNLOADS_API, FakeResponse(status=500)),
    ]))
    out = dl.download_mona(tmp_path, force=True)
    assert json.loads(out.read_text()) == [{"p": 0}, {"p": 1}]


def test_download_mona_all_strategies_fail(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(status=500))
    assert dl.download_mona(tmp_path, force=True) is None
    assert "unreachable" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# MassBank download
# ---------------------------------------------------------------------------

def test_download_massbank_skips_when_present(tmp_path):
    (tmp_path / "massbank_records").mkdir()
    (tmp_path / "massbank_records" / "a.txt").write_text("x")
    assert dl.download_massbank_eu(tmp_path) == tmp_path / "massbank_records"


def test_download_massbank_fetches_release(tmp_path, monkeypatch):
    payload = _zip_bytes({"repo/sub/A.txt": b"one", "repo/B.txt": b"two", "repo/README": b"no"})
    release = {"zipball_url": "https://example.test/zip", "tag_name": "2024.1"}
    monkeypatch.setattr(dl.requests, "get", _router([
        (dl.MASSBANK_RELEASE_API, FakeResponse(json_data=release)),
        ("https://example.test/zip", FakeResponse(payload)),
    ]))
    out = dl.download_massbank_eu(tmp_path)
    assert sorted(p.name for p in out.iterdir()) == ["A.txt", "B.txt"]
    assert not (tmp_path / "massbank_2024.1.zip").exists()
