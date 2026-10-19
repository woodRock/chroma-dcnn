"""
Export the fish oil results as taura ledger results (ledger/results/*.json), for \\res{} keys.

Inputs (results/fish_oil/, all committed):
  chroma_results.json            scripts/04_finetune_evaluate.py (pretrained, from scratch)
  baseline_gcms_*_results.json   scripts/05_run_baselines.py (sum, max_proj, chroma_pca)
  main_table.csv                 scripts/12_results_table.py (ranks, |r| and p against the best)
  *_comparisons.csv, ablation_1d_vs_2d_*   scripts 04, 05 and 09
  configs/finetune.yaml          cv_seeds and cv_folds

Results:
  <condition>         balanced-accuracy and macro-f1: mean, std (population, as script 12), n,
                      seeds, folds and the raw per-fold values; conditions are chroma-pretrain,
                      chroma-scratch, {plsda,rf,svm,mlp}-{sum,maxproj,chromapca},
                      ablation-2d and ablation-1d-mz
  main-table          per condition: rank, abs-r and p (raw, Bonferroni) against the best;
                      all: the |r| range
  comparisons         pretrained vs from scratch; baselines within each representation
  ablation            1D-mz against 2D: delta, p, |r| (from the ablation summary)
  gains               percentage-point gains the text quotes

Keys look like \\res{svm-chromapca/balanced-accuracy/mean} and
\\res{main-table/svm-chromapca/abs-r}.
The means and SDs in main_table.csv are checked against the raw values; a mismatch stops the export.

Usage: python scripts/export_ledger.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "fish_oil"
CONFIG = ROOT / "configs" / "finetune.yaml"
OUT = ROOT / "ledger" / "results"

METHODS = {"plsda": "PLS-DA", "rf": "Random Forest", "svm": "SVM (RBF)", "mlp": "MLP"}
REPS = {
    "sum": ("sum", "sum spectrum"),
    "maxproj": ("max_proj", "max-projection"),
    "chromapca": ("chroma_pca", "chroma-PCA"),
}
CHROMA = {
    "chroma-pretrain": ("chroma_pretrain", "ChromaDCNN (pretrained)"),
    "chroma-scratch": ("from_scratch", "ChromaDCNN (from scratch)"),
}
METRICS = {"balanced_accuracy": "balanced-accuracy", "macro_f1": "macro-f1"}


def _source(*paths: Path) -> dict:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    rel = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths]
    return {"path": rel[0] if len(rel) == 1 else ", ".join(rel), "hash": "sha256:" + h.hexdigest()}


def cv_design(config: Path = CONFIG) -> tuple[int, int]:
    """(seeds, folds) from the fine-tuning config."""
    text = config.read_text()
    folds = int(re.search(r"^\s*cv_folds:\s*(\d+)", text, re.M).group(1))
    seeds = re.search(r"^\s*cv_seeds:\s*\[([^\]]*)\]", text, re.M).group(1)
    return len([s for s in seeds.split(",") if s.strip()]), folds


def summarise(values: list[float], seeds: int, folds: int) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "n": int(v.size),
        "seeds": seeds,
        "folds": folds,
        "values": [float(x) for x in v],
    }


def conditions(seeds: int, folds: int) -> dict[str, dict]:
    """{result name: {"label", "metrics", "source"}} for every condition and the ablation."""
    out = {}
    chroma = json.loads((RES / "chroma_results.json").read_text())
    for name, (key, label) in CHROMA.items():
        out[name] = {
            "label": label,
            "metrics": {METRICS[m]: summarise(v, seeds, folds) for m, v in chroma[key].items()},
            "source": _source(RES / "chroma_results.json"),
        }
    for rep, (file_key, rep_label) in REPS.items():
        path = RES / f"baseline_gcms_{file_key}_results.json"
        data = json.loads(path.read_text())
        for clf, clf_label in METHODS.items():
            out[f"{clf}-{rep}"] = {
                "label": f"{clf_label} on {rep_label}",
                "metrics": {METRICS[m]: summarise(v, seeds, folds) for m, v in data[clf].items()},
                "source": _source(path),
            }
    abl = RES / "ablation_1d_vs_2d_results.json"
    data = json.loads(abl.read_text())
    for key, name, label in (
        ("2D", "ablation-2d", "Ablation: 2D (from scratch)"),
        ("1D-mz", "ablation-1d-mz", "Ablation: 1D-mz (no RT axis)"),
    ):
        out[name] = {
            "label": label,
            "metrics": {METRICS[m]: summarise(v, seeds, folds) for m, v in data[key].items()},
            "source": _source(abl),
        }
    return out


def condition_of(method: str, representation: str) -> str:
    """main_table.csv's method and representation → the condition's result name."""
    if method.startswith("ChromaDCNN"):
        return "chroma-pretrain" if "pretrain" in method else "chroma-scratch"
    clf = next(k for k, v in METHODS.items() if v == method)
    rep = next(k for k, (_, label) in REPS.items() if label == representation)
    return f"{clf}-{rep}"


def main_table(conds: dict[str, dict]) -> dict:
    """Ranks, |r| and p against the best, from script 12's table, checked against the raw values."""
    path = RES / "main_table.csv"
    metrics: dict[str, dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = condition_of(row["method"], row["representation"])
            ba = conds[name]["metrics"]["balanced-accuracy"]
            for col, stat in (("ba_mean", "mean"), ("ba_std", "std")):
                if not np.isclose(float(row[col]), ba[stat], rtol=0, atol=1e-12):
                    raise ValueError(
                        f"{path.name}: {name} {col} {row[col]} != {ba[stat]} from the raw values"
                    )
            entry: dict = {"rank": int(row["rank"])}
            if row["abs_r_vs_best"]:
                entry |= {
                    "abs-r": float(row["abs_r_vs_best"]),
                    "p-raw": float(row["p_raw"]),
                    "p-bonf": float(row["p_bonferroni"]),
                }
            metrics[name] = entry
    rs = [m["abs-r"] for m in metrics.values() if "abs-r" in m]
    ps = [m["p-bonf"] for m in metrics.values() if "p-bonf" in m]
    metrics["all"] = {
        "abs-r-min": min(rs),
        "abs-r-max": max(rs),
        "p-bonf-max": max(ps),
        "conditions": len(metrics),
    }
    return {
        "label": "Table 1: rank, |r| and Bonferroni p against the best method",
        "metrics": metrics,
        "source": _source(path),
    }


def comparisons() -> dict:
    metrics: dict[str, dict] = {}
    paths = [RES / "comparisons_chroma.csv"] + [
        RES / f"baseline_gcms_{k}_comparisons.csv" for _, (k, _) in REPS.items()
    ]
    names = {"chroma_pretrain": "pretrain", "from_scratch": "scratch"}
    for path in paths:
        rep = next((r for r, (k, _) in REPS.items() if f"_{k}_" in path.name), None)
        with open(path) as f:
            for row in csv.DictReader(f):
                a = names.get(row["condition_a"], row["condition_a"])
                b = names.get(row["condition_b"], row["condition_b"])
                key = f"{a}-vs-{b}" + (f"-{rep}" if rep else "")
                metrics[key] = {
                    "u": float(row["U"]),
                    "p-raw": float(row["p_raw"]),
                    "p-corrected": float(row["p_corrected"]),
                    "effect-r": float(row["effect_r"]),
                    "abs-r": abs(float(row["effect_r"])),
                }
    return {"label": "Mann-Whitney U comparisons", "metrics": metrics, "source": _source(*paths)}


def ablation() -> dict:
    """1D-mz against 2D, from the ablation summary script 09 writes."""
    path = RES / "ablation_1d_vs_2d_summary.md"
    m = re.search(
        r"\|\s*1D-mz[^|]*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*([-+]?[\d.]+)\s*\|\s*([\d.e+-]+)\s*\|"
        r"\s*([\d.]+)\s*\|",
        path.read_text(),
    )
    if not m:
        raise ValueError(f"{path.name}: no 1D-mz row")
    return {
        "label": "Ablation: 1D-mz against 2D (from scratch)",
        "source": _source(path),
        "metrics": {
            "1d-vs-2d": {
                "delta": float(m.group(3)),
                "p": float(m.group(4)),
                "abs-r": float(m.group(5)),
            }
        },
    }


def gains(conds: dict[str, dict], table: dict) -> dict:
    """Percentage-point gains of pretrained ChromaDCNN that the text quotes."""
    ba = {n: c["metrics"]["balanced-accuracy"]["mean"] for n, c in conds.items()}
    best_baseline = max((n for n in ba if not n.startswith(("chroma", "ablation"))), key=ba.get)

    def pp(a: str, b: str) -> float:
        return 100 * (ba[a] - ba[b])

    return {
        "label": "Gains in balanced accuracy (percentage points)",
        "source": table["source"],
        "metrics": {
            "pretrain-vs-scratch": {"points": pp("chroma-pretrain", "chroma-scratch")},
            "pretrain-vs-best-baseline": {"points": pp("chroma-pretrain", best_baseline)},
            # the RT ablation: what removing the RT axis costs (and the signed change 1D - 2D)
            "ablation-rt-drop": {"points": pp("ablation-2d", "ablation-1d-mz")},
            "ablation-1d-vs-2d": {"points": pp("ablation-1d-mz", "ablation-2d")},
            # what the chroma-PCA representation gains over the sum spectrum
            "plsda-chromapca-vs-sum": {"points": pp("plsda-chromapca", "plsda-sum")},
            "svm-chromapca-vs-sum": {"points": pp("svm-chromapca", "svm-sum")},
        },
    }


def scratch_comparisons(conds: dict[str, dict], table: dict) -> dict:
    """From-scratch ChromaDCNN against every baseline: two-sided Mann-Whitney U, Bonferroni over the
    same number of comparisons as Table 1 (script 12: the conditions minus one)."""
    from scipy.stats import mannwhitneyu

    n_comparisons = table["metrics"]["all"]["conditions"] - 1
    scratch = np.asarray(conds["chroma-scratch"]["metrics"]["balanced-accuracy"]["values"])
    metrics = {}
    for name, c in conds.items():
        if name.startswith(("chroma", "ablation")):
            continue
        other = np.asarray(c["metrics"]["balanced-accuracy"]["values"])
        u, p = mannwhitneyu(scratch, other, alternative="two-sided")
        metrics[f"scratch-vs-{name}"] = {
            "u": float(u),
            "p-raw": float(p),
            "p-bonf": float(min(1.0, p * n_comparisons)),
            "abs-r": float(abs(1 - 2 * u / (scratch.size * other.size))),
        }
    return {
        "label": "From-scratch ChromaDCNN against each baseline (Mann-Whitney U, Bonferroni)",
        "metrics": metrics,
        "source": conds["chroma-scratch"]["source"],
    }


def saliency() -> dict:
    """What scripts/11_saliency.py reports about the seed-0 saliency models
    (saliency_summary.json): their fold accuracy, each RT band's share of every species' RT
    saliency, and how similar the species' profiles are."""
    path = RES / "saliency_summary.json"
    d = json.loads(path.read_text())
    metrics = {
        "fold-balanced-accuracy": d["fold_balanced_accuracy"],
        "rt-profile-corr": d["rt_profile_corr"],
        "mz-profile-corr": d["mz_profile_corr"],
    }
    for band, share in d["band_share"].items():
        metrics[f"band-{band}"] = share
    return {
        "label": "Gradient saliency (seed 0, 5 folds)",
        "source": _source(path),
        "metrics": metrics,
    }


def pretraining() -> dict | None:
    """Wall-clock time and final MSE of pretraining (scripts/03_pretrain_chroma.py), if recorded."""
    path = ROOT / "results" / "pretrain_summary.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    return {
        "label": "Pretraining (next-frame prediction on synthetic chromatograms)",
        "source": _source(path),
        "metrics": {
            "wall-minutes": d["wall_minutes"],
            "final-mse": d["final_mse"],
            "best-mse": d["best_mse"],
            "iterations": d["n_iterations"],
            "spectra": d["n_spectra"],
        },
    }


def results() -> dict[str, dict]:
    seeds, folds = cv_design()
    conds = conditions(seeds, folds)
    table = main_table(conds)
    out = dict(conds)
    out |= {
        "main-table": table,
        "comparisons": comparisons(),
        "ablation": ablation(),
        "gains": gains(conds, table),
        "scratch-comparisons": scratch_comparisons(conds, table),
        "saliency": saliency(),
        **({"pretraining": pre} if (pre := pretraining()) else {}),
    }
    return {
        name: {"name": name, "runs": [], "produced_by": "scripts/export_ledger.py", **r}
        for name, r in out.items()
    }


def main(out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, r in results().items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(r, indent=2) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in main():
        print(p.relative_to(ROOT))
