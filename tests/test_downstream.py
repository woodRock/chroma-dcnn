"""Downstream dataset loaders: label/file alignment is what matters here."""

from __future__ import annotations

import numpy as np
import pytest

from chroma_dcnn.downstream import fish_oil, rice, urine, wheat

# ---------------------------------------------------------------------------
# Fish oil
# ---------------------------------------------------------------------------

FISH_IDS = ["1 - SNA1 Frame A", "2 - GUR2  Frame A", "3 - TAR3 Head-B", "4 - BCO4 Skin A"]


def _write_fish(root, subdir="chroma", key="chroma", with_meta=True, skip=()):
    (root / subdir).mkdir(parents=True)
    for i, sid in enumerate(FISH_IDS):
        if sid in skip:
            continue
        fname = sid.replace(" ", "_").replace("-", "_") + ".npz"
        arr = np.full((3, 5), float(i), np.float32)
        if key == "scans":
            np.savez(root / subdir / fname, scans=arr, tic_weights=np.ones(3, np.float32) / 3)
        else:
            np.savez(root / subdir / fname, chroma=arr)
    if with_meta:
        (root / "sample_ids.txt").write_text("\n".join(FISH_IDS))
        np.save(root / "y.npy", np.array([0, 1, 2, 3]))


def test_fish_chroma_paths_align_labels_by_sample_id(tmp_path):
    _write_fish(tmp_path, skip=("3 - TAR3 Head-B",))
    paths, y = fish_oil.load_fish_oil_chroma_paths(tmp_path)
    assert y.tolist() == [0, 1, 3]
    for p, label in zip(paths, y):
        assert np.load(p)["chroma"][0, 0] == FISH_IDS.index(
            next(s for s in FISH_IDS if s.replace(" ", "_").replace("-", "_") == p.stem))
        assert fish_oil.SPECIES_MAP[label] in p.stem


def test_fish_chroma_paths_fallback_parses_filenames(tmp_path):
    _write_fish(tmp_path, with_meta=False)
    np.savez(tmp_path / "chroma" / "junk.npz", chroma=np.zeros((1, 1)))
    paths, y = fish_oil.load_fish_oil_chroma_paths(tmp_path)
    assert len(paths) == 4
    for p, label in zip(paths, y):
        assert fish_oil.SPECIES_MAP[label] in p.stem


def test_fish_chroma_paths_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_fish_oil_chroma"):
        fish_oil.load_fish_oil_chroma_paths(tmp_path)


def test_fish_chroma_features(tmp_path):
    _write_fish(tmp_path)
    feats, y = fish_oil.load_fish_oil_chroma_features(tmp_path)
    assert feats["max_proj"].shape == (4, 5) and feats["chroma_flat"].shape == (4, 15)
    assert feats["max_proj"][2, 0] == 2.0


def test_fish_scan_paths(tmp_path):
    _write_fish(tmp_path, subdir="scans", key="scans")
    paths, y = fish_oil.load_fish_oil_scan_paths(tmp_path)
    assert y.tolist() == [0, 1, 2, 3] and all(p.parent.name == "scans" for p in paths)


def test_fish_scan_paths_fallback_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="scans"):
        fish_oil.load_fish_oil_scan_paths(tmp_path)
    _write_fish(tmp_path, subdir="scans", key="scans", with_meta=False)
    _, y = fish_oil.load_fish_oil_scan_paths(tmp_path)
    assert sorted(y.tolist()) == [0, 1, 2, 3]


def test_fish_oil_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="preprocess_fish_oil"):
        fish_oil.load_fish_oil_data(tmp_path)
    np.save(tmp_path / "X.npy", np.ones((4, 6)))
    np.save(tmp_path / "y.npy", np.array([0, 1, 2, 3]))
    X, y, ids, meta = fish_oil.load_fish_oil_data(tmp_path)
    assert X.dtype == np.float32 and ids[0] == "sample_0" and meta["bio_groups"] is None
    (tmp_path / "sample_ids.txt").write_text("\n".join(FISH_IDS))
    (tmp_path / "groups.txt").write_text("SNA1\nGUR2\nTAR3\nBCO4")
    _, _, ids, meta = fish_oil.load_fish_oil_data(tmp_path)
    assert ids == FISH_IDS and meta["bio_groups"][3] == "BCO4"
    assert meta["class_counts"] == {0: 1, 1: 1, 2: 1, 3: 1}


# ---------------------------------------------------------------------------
# Rice / urine / wheat share one loader shape
# ---------------------------------------------------------------------------

LOADERS = [
    (rice, "load_rice_chroma_paths", "load_rice_data", "load_rice_chroma_features"),
    (urine, "load_urine_chroma_paths", "load_urine_data", "load_urine_chroma_features"),
    (wheat, "load_wheat_chroma_paths", "load_wheat_data", "load_wheat_chroma_features"),
]


@pytest.mark.parametrize(("mod", "paths_fn", "data_fn", "feat_fn"), LOADERS)
def test_other_loaders(tmp_path, mod, paths_fn, data_fn, feat_fn):
    with pytest.raises(FileNotFoundError):
        getattr(mod, paths_fn)(tmp_path)

    (tmp_path / "chroma").mkdir()
    ids = ["s_Female_1", "s_Male_2", "s_Male_3"]
    for i, sid in enumerate(ids[:2]):  # third sample missing on disk
        np.savez(tmp_path / "chroma" / f"{sid}.npz", chroma=np.full((2, 3), float(i), np.float32))

    paths, y = getattr(mod, paths_fn)(tmp_path)  # fallback: no sample_ids / y
    assert len(paths) == 2

    (tmp_path / "sample_ids.txt").write_text("\n".join(ids))
    np.save(tmp_path / "y.npy", np.array([0, 1, 1]))
    paths, y = getattr(mod, paths_fn)(tmp_path)
    assert [p.stem for p in paths] == ids[:2] and y.tolist() == [0, 1]

    feats, y = getattr(mod, feat_fn)(tmp_path)
    assert feats["max_proj"].shape == (2, 3) and feats["chroma_flat"].shape == (2, 6)

    np.save(tmp_path / "X.npy", np.ones((3, 4)))
    X, y, sids, meta = getattr(mod, data_fn)(tmp_path)
    assert X.shape == (3, 4) and sids == ids and meta["n_samples"] == 3
    (tmp_path / "sample_ids.txt").unlink()
    assert getattr(mod, data_fn)(tmp_path)[2][0] == "sample_0"


def test_urine_fallback_labels_from_filename(tmp_path):
    (tmp_path / "chroma").mkdir()
    for sid in ["a_Female", "b_Male"]:
        np.savez(tmp_path / "chroma" / f"{sid}.npz", chroma=np.zeros((1, 1)))
    _, y = urine.load_urine_chroma_paths(tmp_path)
    assert y.tolist() == [0, 1]
