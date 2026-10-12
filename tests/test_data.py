"""Spectrum preprocessing, HDF5 building, and the Dataset classes."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from chroma_dcnn.data.datasets import (
    AugmentationConfig,
    ChromaDataset,
    ContrastiveDataset,
    DownstreamDataset,
    MSMDataset,
    PretrainingDataset,
    ScanDataset,
    SyntheticChromaDataset,
)
from chroma_dcnn.data.preprocess import (
    bin_spectrum,
    build_hdf5,
    preprocess_spectrum,
    report_dataset_stats,
    sqrt_l2_normalize,
)

# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

def test_bin_spectrum_rounds_clips_and_keeps_max():
    vec = bin_spectrum([(10.4, 5.0), (9.6, 7.0), (-1, 3.0), (20.0, 1.0), (3.0, 0.0)], mz_max=20)
    assert vec.shape == (20,) and vec.dtype == np.float32
    assert vec[10] == 7.0  # 10.4 and 9.6 both round to 10; max wins
    assert vec.sum() == 7.0  # out-of-range and zero-intensity peaks dropped


def test_sqrt_l2_normalize():
    out = sqrt_l2_normalize(np.array([4.0, 0.0, -1.0, 9.0]))
    np.testing.assert_allclose(out, np.array([2, 0, 0, 3]) / np.sqrt(13), rtol=1e-6)
    assert out.dtype == np.float32


def test_sqrt_l2_normalize_zero_vector():
    assert not sqrt_l2_normalize(np.zeros(5)).any()


def test_preprocess_spectrum_unit_norm():
    out = preprocess_spectrum([(50, 100.0), (51, 25.0)], mz_max=100)
    assert np.linalg.norm(out) == pytest.approx(1.0)


def _records(n_mona=10, n_mb=5):
    recs = []
    for i in range(n_mona + n_mb):
        recs.append({"inchikey": f"KEY{i:011d}", "peaks": [(10 + i, 1.0), (40, 2.0)],
                     "smiles": "C", "source": "mona" if i < n_mona else "massbank"})
    return recs


def test_build_hdf5_layout_and_split(tmp_path, capsys):
    out = build_hdf5(_records(), tmp_path / "sub" / "s.h5", val_fraction=0.2, mz_max=64)
    with h5py.File(out) as f:
        assert f["spectra"].shape == (15, 64)
        split = [s.decode() for s in f["split"][:]]
        sources = [s.decode() for s in f["sources"][:]]
        assert f.attrs["n_train"] + f.attrs["n_val"] == 15
    # stratified by source: 20% of 10 mona = 2, max(1, 20% of 5) = 1 massbank
    assert sum(s == "val" and src == "mona" for s, src in zip(split, sources)) == 2
    assert sum(s == "val" and src == "massbank" for s, src in zip(split, sources)) == 1
    report_dataset_stats(out)
    assert "Total spectra     : 15" in capsys.readouterr().out


@pytest.fixture
def h5_path(tmp_path):
    return build_hdf5(_records(), tmp_path / "s.h5", val_fraction=0.2, mz_max=64)


# ---------------------------------------------------------------------------
# Pretraining datasets
# ---------------------------------------------------------------------------

def test_pretraining_dataset_splits(h5_path):
    train, val = PretrainingDataset(h5_path, "train"), PretrainingDataset(h5_path, "val")
    assert len(train) + len(val) == 15 and len(val) == 3
    assert train[0].shape == (64,) and train.mz_max == 64
    assert set(train.inchikeys).isdisjoint(val.inchikeys)


def test_msm_dataset_masks_only_nonzero(h5_path):
    ds = MSMDataset(PretrainingDataset(h5_path), mask_ratio=0.5)
    orig = ds.base[0]
    item = ds[0]
    assert len(ds) == len(ds.base)
    assert item["mask"].sum() >= 1
    assert torch.all(orig[item["mask"]] > 0)
    assert torch.all(item["spectrum"][item["mask"]] == 0)
    torch.testing.assert_close(item["targets"][item["mask"]], orig[item["mask"]])
    assert torch.all(item["targets"][~item["mask"]] == 0)


def test_contrastive_dataset_views(h5_path):
    ds = ContrastiveDataset(PretrainingDataset(h5_path),
                            AugmentationConfig(low_mass_cutoff=8, bin_dropout_rate=0.0))
    item = ds[0]
    assert len(ds) == 12
    assert item["view1"].shape == item["view2"].shape == (64,)
    assert item["view1"].norm() == pytest.approx(1.0, abs=1e-5)
    assert not torch.equal(item["view1"], item["view2"])


def test_contrastive_augment_all_dropped_stays_zero():
    ds = ContrastiveDataset(None, AugmentationConfig(low_mass_cutoff=2, low_mass_noise_sigma=0.0,
                                                     bin_dropout_rate=1.0))
    assert ds._augment(torch.ones(10)).abs().sum() == 0


def test_synthetic_chroma_dataset_is_deterministic(h5_path):
    ds = SyntheticChromaDataset(h5_path, n_samples=4, n_bins=10, n_compounds_range=(1, 3))
    x, label = ds[2]
    assert len(ds) == 4 and x.shape == (10, 64) and label.item() == 0
    torch.testing.assert_close(x, ds[2][0])
    assert torch.all((x.norm(dim=-1) - 1).abs() < 1e-4)


# ---------------------------------------------------------------------------
# Downstream datasets
# ---------------------------------------------------------------------------

def test_downstream_dataset():
    ds = DownstreamDataset(np.ones((3, 4), np.float64), np.array([0, 1, 2]))
    x, y = ds[1]
    assert len(ds) == 3 and x.dtype == torch.float32 and y.dtype == torch.int64 and y == 1


def test_chroma_dataset_loads_files_in_order(chroma_dataset):
    paths, y, _ = chroma_dataset
    ds = ChromaDataset(paths, y)
    x, label = ds[9]
    assert len(ds) == len(paths)
    np.testing.assert_array_equal(x.numpy(), np.load(paths[9])["chroma"])
    assert label == y[9]


def test_scan_dataset_truncates_and_renormalises(tmp_path):
    p = tmp_path / "s.npz"
    np.savez(p, scans=np.ones((5, 4), np.float32),
             tic_weights=np.array([0.4, 0.3, 0.1, 0.1, 0.1], np.float32))
    full = ScanDataset([p], np.array([1]))
    scans, w, y = full[0]
    assert scans.shape == (5, 4) and y == 1
    cut = ScanDataset([p], np.array([1]), max_scans=2)
    scans, w, _ = cut[0]
    assert len(cut) == 1 and scans.shape == (2, 4)
    assert w.sum() == pytest.approx(1.0, abs=1e-6) and w[0] == pytest.approx(4 / 7, abs=1e-6)
