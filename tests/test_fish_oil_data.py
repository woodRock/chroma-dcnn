"""Integrity checks on the committed fish-oil data used for the paper's results."""

from __future__ import annotations

import re

import numpy as np
import pytest

from chroma_dcnn.downstream.fish_oil import SPECIES_LABEL, load_fish_oil_chroma_paths

from .conftest import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "fish_oil"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not (DATA_DIR / "chroma").exists(), reason="fish-oil data not present"),
]


@pytest.fixture(scope="module")
def meta():
    ids = (DATA_DIR / "sample_ids.txt").read_text().splitlines()
    groups = (DATA_DIR / "groups.txt").read_text().splitlines()
    return ids, np.load(DATA_DIR / "y.npy"), groups


def test_metadata_lengths_agree(meta):
    ids, y, groups = meta
    assert len(ids) == len(y) == len(groups) == 103
    assert len(set(ids)) == len(ids)


def test_labels_match_sample_names(meta):
    ids, y, groups = meta
    for sid, label, group in zip(ids, y, groups):
        species = re.search(r"\b([A-Z]{3})\d", sid).group(1)
        assert SPECIES_LABEL[species] == label, sid
        assert group.startswith(species) and group in sid, sid


def test_every_sample_has_a_chromatogram(meta):
    ids, y, _ = meta
    paths, y_loaded = load_fish_oil_chroma_paths(DATA_DIR)
    assert len(paths) == len(ids)
    np.testing.assert_array_equal(y_loaded, y)
    for p, label in zip(paths, y_loaded):
        assert re.search(r"___([A-Z]{3})\d", p.stem).group(1) == \
            {v: k for k, v in SPECIES_LABEL.items()}[label]


def test_chromatograms_are_well_formed():
    paths, _ = load_fish_oil_chroma_paths(DATA_DIR)
    for p in paths:
        c = np.load(p)["chroma"]
        assert c.shape == (200, 1000) and c.dtype == np.float32, p.name
        assert np.isfinite(c).all() and (c >= 0).all(), p.name
        norms = np.linalg.norm(c, axis=1)
        assert np.all((norms < 1e-6) | (np.abs(norms - 1) < 1e-3)), p.name


def test_no_duplicate_chromatograms():
    paths, _ = load_fish_oil_chroma_paths(DATA_DIR)
    digests = {np.load(p)["chroma"].tobytes() for p in paths}
    assert len(digests) == len(paths)


def test_sum_spectra_aligned(meta):
    X = np.load(DATA_DIR / "X.npy")
    assert X.shape == (103, 1000) and np.isfinite(X).all()
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)
