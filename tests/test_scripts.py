"""Fish-oil pipeline scripts, run end to end on tiny synthetic inputs."""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pytest
import yaml

from .conftest import REPO_ROOT, write_chroma_dataset


def load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, n_scans=30, mz=(50, 51, 75), peak_scan=10, peak_mz_col=1, scale=1.0):
    """Mimic the instrument CSV export: 2 header rows, then Time, TIC, m/z columns."""
    rng = np.random.default_rng(len(path.name))
    data = rng.random((n_scans, len(mz))) * 0.01
    data[peak_scan, peak_mz_col] = 100.0 * scale
    lines = ["X-data, Y-data, " + ", ".join(f"Packet #{i}" for i in range(len(mz))),
             "Time, TIC, " + ", ".join(f"{m}.00" for m in mz)]
    for i, row in enumerate(data):
        lines.append(f"{i * 0.1:.2f}, {row.sum():.4f}, " + ", ".join(f"{v:.4f}" for v in row))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_csv(raw / "1 - SNA1 Frame A.csv")
    _write_csv(raw / "2 - GUR2  Frame A.csv", peak_scan=20, peak_mz_col=2)
    _write_csv(raw / "3 - TAR3 Head B.csv")
    (raw / "notes.csv").write_text("not a sample\n")
    return raw


# ---------------------------------------------------------------------------
# preprocess_fish_oil.py (sum spectra)
# ---------------------------------------------------------------------------

def test_preprocess_fish_oil_parse_filename():
    mod = load_script("preprocess_fish_oil")
    assert mod._parse_filename("5 - BCO5 Liver Male A.csv") == (3, "BCO5")
    assert mod._parse_filename("1 - XYZ1 Frame.csv") is None
    assert mod._parse_filename("readme.csv") is None


def test_preprocess_fish_oil_main(raw_dir, tmp_path):
    mod = load_script("preprocess_fish_oil")
    out = tmp_path / "out"
    mod.main(raw_dir, out)
    X, y = np.load(out / "X.npy"), np.load(out / "y.npy")
    assert X.shape == (3, 1000) and y.tolist() == [0, 1, 2]
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, rtol=1e-5)
    assert (out / "groups.txt").read_text().splitlines() == ["SNA1", "GUR2", "TAR3"]
    assert (out / "sample_ids.txt").read_text().splitlines()[1] == "2 - GUR2  Frame A"
    assert X[0].argmax() == 51 and X[1].argmax() == 75


def test_preprocess_fish_oil_no_files(tmp_path):
    with pytest.raises(RuntimeError, match="No parseable"):
        load_script("preprocess_fish_oil").main(tmp_path, tmp_path / "o")


# ---------------------------------------------------------------------------
# preprocess_fish_oil_chroma.py (RT-binned chromatograms)
# ---------------------------------------------------------------------------

def test_bin_chromatogram_keeps_max_tic_scan_per_window():
    mod = load_script("preprocess_fish_oil_chroma")
    spectra = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    tic = np.array([0, 5, 1, 0, 0, 9, 0, 2, 1, 0], np.float32)
    out = mod._bin_chromatogram(spectra, tic, n_bins=2)
    assert out[:, 0].tolist() == [1.0, 5.0]
    # more bins than scans → empty windows are zero rows
    assert mod._bin_chromatogram(spectra[:3], tic[:3], n_bins=5).shape == (5, 2)


def test_preprocess_fish_oil_chroma_main_and_loader_agree(raw_dir, tmp_path):
    """Chroma filenames must match what load_fish_oil_chroma_paths derives from sample_ids."""
    from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_paths

    data_dir = tmp_path / "fish"
    load_script("preprocess_fish_oil").main(raw_dir, data_dir)
    mod = load_script("preprocess_fish_oil_chroma")
    mod.main(raw_dir, data_dir / "chroma", n_bins=6)
    mod.main(raw_dir, data_dir / "chroma", n_bins=6)  # second run skips existing files

    paths, y = load_fish_oil_chroma_paths(data_dir)
    assert len(paths) == 3 and y.tolist() == [0, 1, 2]
    c = np.load(paths[1])["chroma"]
    assert c.shape == (6, 1000)
    peak_bin = 20 * 6 // 30
    assert c[peak_bin].argmax() == 75
    nonzero = np.linalg.norm(c, axis=1) > 0
    np.testing.assert_allclose(np.linalg.norm(c[nonzero], axis=1), 1.0, rtol=1e-5)


def test_preprocess_fish_oil_chroma_no_files(tmp_path):
    with pytest.raises(RuntimeError, match="No parseable"):
        load_script("preprocess_fish_oil_chroma").main(tmp_path, tmp_path / "o", 4)


# ---------------------------------------------------------------------------
# 04_finetune_evaluate.py / 05_run_baselines.py / 06_smoke_test.py
# ---------------------------------------------------------------------------

@pytest.fixture
def fish_project(tmp_path, finetune_cfg):
    """Tiny fish-oil-shaped data dir + config file, using real species filenames."""
    data_dir = tmp_path / "fish"
    paths, y, _ = write_chroma_dataset(data_dir / "chroma_tmp", n_per_class=6)
    (data_dir / "chroma").mkdir()
    species = ["SNA", "GUR", "TAR"]
    ids = []
    for i, (p, label) in enumerate(zip(paths, y)):
        sid = f"{label + 1} - {species[label]}{label + 1} Part {i}"
        ids.append(sid)
        p.rename(data_dir / "chroma" / (sid.replace(" ", "_").replace("-", "_") + ".npz"))
    (data_dir / "sample_ids.txt").write_text("\n".join(ids))
    np.save(data_dir / "y.npy", y)
    flat = np.stack([np.load(data_dir / "chroma" / (s.replace(" ", "_").replace("-", "_")
                                                    + ".npz"))["chroma"].sum(0) for s in ids])
    np.save(data_dir / "X.npy", flat)

    cfg = dict(finetune_cfg)
    cfg["data"] = {"data_dir": str(data_dir)}
    cfg["output"] = {"results_dir": str(tmp_path / "results")}
    cfg["task"]["cv_seeds"] = [0, 1]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, tmp_path / "results"


def test_finetune_evaluate_script(fish_project):
    cfg_path, results = fish_project
    load_script("04_finetune_evaluate").main(str(cfg_path), device="cpu")
    res = json.loads((results / "chroma_results.json").read_text())
    assert list(res) == ["from_scratch"]
    assert len(res["from_scratch"]["balanced_accuracy"]) == 6  # 2 seeds × 3 folds
    assert (results / "comparisons_chroma.csv").exists()


def test_run_baselines_script(fish_project, monkeypatch):
    from chroma_dcnn.evaluation import baselines as bl

    cfg_path, results = fish_project
    mod = load_script("05_run_baselines")
    # keep the expensive grid searches out of the test; the CV plumbing is what's tested
    fast = bl.run_random_forest
    monkeypatch.setattr(bl, "run_svm", lambda X, y: fast(X, y))
    monkeypatch.setattr(bl, "run_mlp", lambda X, y: fast(X, y))
    mod.main(str(cfg_path))
    for rep in ("sum", "max_proj", "chroma_pca"):
        res = json.loads((results / f"baseline_gcms_{rep}_results.json").read_text())
        assert set(res) == {"plsda", "rf", "svm", "mlp"}
        assert len(res["rf"]["balanced_accuracy"]) == 6


def test_smoke_test_script(fish_project, monkeypatch, capsys):
    cfg_path, _ = fish_project
    mod = load_script("06_smoke_test")
    monkeypatch.setattr(mod, "_device", lambda: __import__("torch").device("cpu"))
    mod.main(str(cfg_path), seed=0, fold_idx=1, epochs=2)
    out = capsys.readouterr().out
    assert "from_scratch" in out and "Test : 6" in out
    with pytest.raises(ValueError, match="out of range"):
        mod.main(str(cfg_path), seed=0, fold_idx=9, epochs=1)


def test_scripts_parse_cli(monkeypatch):
    """Each script's argparse block runs with --help without importing data."""
    import runpy

    for name in ("04_finetune_evaluate", "05_run_baselines", "06_smoke_test",
                 "preprocess_fish_oil", "preprocess_fish_oil_chroma"):
        monkeypatch.setattr(sys, "argv", [name, "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(REPO_ROOT / "scripts" / f"{name}.py"), run_name="__main__")
        assert exc.value.code == 0
