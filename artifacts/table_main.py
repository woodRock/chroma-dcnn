"""Table 1: balanced accuracy of all 14 methods, ranked, with |r| and Bonferroni p against the best.

Run by `taura build`; reads $TAURA_RESULTS (the conditions and main-table) and writes a taura
table spec to $TAURA_OUTPUT. Options ($TAURA_OPTIONS, from ledger/artifacts.toml): p_style = "sci"
(every p in scientific notation) or "threshold" (p below 0.001 as <0.001).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

METHODS = {"plsda": "PLS-DA", "rf": "Random Forest", "svm": "SVM (RBF)", "mlp": "MLP"}
REPS = {"sum": "sum spectrum", "maxproj": "max-projection", "chromapca": "chroma-PCA"}
CHROMA = {"chroma-pretrain": "ChromaDCNN (pretrained)",
          "chroma-scratch": "ChromaDCNN (from scratch)"}


def labels(name: str) -> tuple[str, str]:
    if name in CHROMA:
        return CHROMA[name], "2D chromatogram"
    clf, rep = name.split("-")
    return METHODS[clf], REPS[rep]


def main() -> None:
    R = json.loads(Path(os.environ["TAURA_RESULTS"]).read_text())
    opts = json.loads(os.environ.get("TAURA_OPTIONS", "{}"))
    p_style = opts.get("p_style", "sci")
    table = R["main-table"]["metrics"]
    order = sorted((n for n in table if n != "all"), key=lambda n: table[n]["rank"])
    cols = [{"label": "Rank"}, {"label": "Method"}, {"label": "Representation"},
            {"label": "Balanced acc.", "metric": "balanced-accuracy"},
            {"label": "$|r|$", "metric": "abs-r", "bold": "none"},
            {"label": "$p$ (Bonf.)", "metric": "p-bonf", "bold": "none"}]
    rows = []
    for name in order:
        t = table[name]
        method, rep = labels(name)
        ba = R[name]["metrics"]["balanced-accuracy"]
        if t["rank"] == 1:  # the best: bold row, nothing to compare against
            rows.append(["1", rf"\textbf{{{method}}}", rf"\textbf{{{rep}}}",
                         {"mean": ba["mean"], "std": ba["std"]}, "---", "---"])
            continue
        p = t["p-bonf"]
        p_cell = "$<0.001$" if p_style == "threshold" and p < 0.001 else p
        rows.append([str(t["rank"]), method, rep, {"mean": ba["mean"], "std": ba["std"]},
                     t["abs-r"], p_cell])
    spec = {"columns": cols, "rows": rows}
    Path(os.environ["TAURA_OUTPUT"]).write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
