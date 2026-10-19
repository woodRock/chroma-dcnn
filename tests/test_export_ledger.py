"""scripts/export_ledger.py: the committed fish oil results → taura ledger results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ex():
    spec = importlib.util.spec_from_file_location(
        "export_ledger", ROOT / "scripts" / "export_ledger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results(ex):
    out = ex.results()
    assert ex.cv_design() == (10, 5)
    conds = [n for n in out if n.startswith(("chroma-", "plsda-", "rf-", "svm-", "mlp-"))]
    assert len(conds) == 14
    ba = out["chroma-pretrain"]["metrics"]["balanced-accuracy"]
    assert (ba["n"], ba["seeds"], ba["folds"], len(ba["values"])) == (50, 10, 5, 50)
    assert round(ba["mean"], 3) == 0.854 and round(ba["std"], 3) == 0.101
    table = out["main-table"]["metrics"]
    assert table["chroma-pretrain"] == {"rank": 1}
    assert (
        table["svm-chromapca"]["rank"] == 2 and round(table["svm-chromapca"]["abs-r"], 3) == 0.711
    )
    assert table["all"]["conditions"] == 14 and table["all"]["abs-r-max"] < 1
    gains = out["gains"]["metrics"]
    assert round(gains["pretrain-vs-scratch"]["points"], 1) == 25.1
    assert round(gains["pretrain-vs-best-baseline"]["points"], 1) == 15.6
    assert out["ablation"]["metrics"]["1d-vs-2d"]["abs-r"] == 0.617
    assert round(gains["ablation-rt-drop"]["points"], 1) == 19.7
    assert round(gains["ablation-1d-vs-2d"]["points"], 1) == -19.7
    # 26.24: the paper's 26.3 came from subtracting the rounded table values
    assert round(gains["plsda-chromapca-vs-sum"]["points"], 1) == 26.2
    assert round(gains["svm-chromapca-vs-sum"]["points"], 1) == 20.5
    sc = out["scratch-comparisons"]["metrics"]
    assert len(sc) == 12
    assert [
        round(sc[f"scratch-vs-{b}"]["p-bonf"], 2)
        for b in ("svm-chromapca", "rf-sum", "plsda-chromapca")
    ] == [0.15, 0.16, 0.44]
    sal = out["saliency"]["metrics"]
    assert round(sal["fold-balanced-accuracy"]["mean"], 3) == 0.887
    assert sal["mz-profile-corr"]["min"] >= 0.99 and sal["mz-profile-corr"]["n"] == 6
    assert (round(sal["rt-profile-corr"]["min"], 2), round(sal["rt-profile-corr"]["max"], 2)) == (
        0.74,
        0.88,
    )
    assert out["comparisons"]["metrics"]["scratch-vs-pretrain"]["abs-r"] == 0.7992


def test_table_must_match_raw_values(ex, tmp_path, monkeypatch):
    """A main_table.csv that disagrees with the raw per-fold values stops the export."""
    res = tmp_path / "fish_oil"
    res.mkdir()
    for f in (ROOT / "results" / "fish_oil").iterdir():
        if f.is_file():
            (res / f.name).write_bytes(f.read_bytes())
    csv = (res / "main_table.csv").read_text().replace("0.8540530303030303", "0.86")
    (res / "main_table.csv").write_text(csv)
    monkeypatch.setattr(ex, "RES", res)
    with pytest.raises(ValueError, match="chroma-pretrain ba_mean"):
        ex.results()


def test_main_writes_every_result(ex, tmp_path):
    names = {p.stem for p in ex.main(tmp_path)}
    assert {"main-table", "comparisons", "ablation", "gains", "chroma-pretrain"} <= names
    assert json.loads((tmp_path / "gains.json").read_text())["name"] == "gains"
