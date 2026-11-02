"""
Generate the end-to-end pipeline overview figure for the paper (requested by a
reviewer: "another figure ... to show the process from the data point of
view" — how the two libraries are processed, how the model is pretrained,
and how the fish data is used for fine-tuning, testing and cross-validation).

Two rows:
  top    (pretraining, no fish data): MoNA/MassBank -> filter/dedup/bin ->
         synthetic chromatogram corpus -> self-supervised pretraining ->
         pretrained checkpoint
  bottom (fine-tuning/evaluation, 103 fish samples): fish 2D chromatograms ->
         5-fold stratified CV x 10 seeds -> {fine-tune on train folds,
         evaluate on held-out test fold} -> balanced accuracy

The pretrained checkpoint feeds into the fine-tuning box with a vertical
arrow, which is the only place the two rows connect.

Run from repo root:
  python3 figs/generate_pipeline.py
Output: figs/pipeline.pdf and figs/pipeline.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FS_BOX, FS_NOTE, FS_ROW = 13.5, 11, 15

TOP_COLOR = "#eef3fc"
TOP_EDGE = "#1f4e9d"
BOTTOM_COLOR = "#fdf1e8"
BOTTOM_EDGE = "#c9680a"
CKPT_COLOR = "#e9f7ec"
CKPT_EDGE = "#2e7d32"


def box(ax, xy, w, h, text, fc, ec, fontsize=FS_BOX, lw=1.6):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.06",
                            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, zorder=3, linespacing=1.3)
    return x, y, w, h


def arrow(ax, p0, p1, color="#333333", lw=1.8, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=1)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(16, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 32)
ax.axis("off")

# ── Row labels ──────────────────────────────────────────────────────────────
ax.text(1, 29.5, "Pretraining (no fish data)", fontsize=FS_ROW, fontweight="bold",
        color=TOP_EDGE, va="center", ha="left")
ax.text(1, 15.5, "Fine-tuning & evaluation (103 fish samples)", fontsize=FS_ROW,
        fontweight="bold", color=BOTTOM_EDGE, va="center", ha="left")

# ── Top row: synthetic-pretraining pipeline ──────────────────────────────────
b1 = box(ax, (13, 18), 15, 9,
         "MoNA +\nMassBank EI-MS\nlibraries\n(raw records)",
         TOP_COLOR, TOP_EDGE)
b2 = box(ax, (32, 18), 15, 9,
         "Filter to EI,\nde-duplicate by\nInChIKey, bin to\nshared $m/z$ grid\n(Sec.~IV-C)",
         TOP_COLOR, TOP_EDGE)
b3 = box(ax, (51, 18), 15, 9,
         "Synthetic chromato-\ngram corpus\n($\\sim$9,9xx spectra,\nFig.~3)",
         TOP_COLOR, TOP_EDGE)
b4 = box(ax, (70, 18), 15, 9,
         "Self-supervised\nnext-frame\nprediction\n(Sec.~IV-D)",
         TOP_COLOR, TOP_EDGE)
b5 = box(ax, (89, 18.5), 10, 8,
         "Pretrained\ncheck-\npoint", CKPT_COLOR, CKPT_EDGE)

for (bx, by, bw, bh), (cx, cy, cw, ch) in zip(
    [b1, b2, b3, b4], [b2, b3, b4, b5]
):
    arrow(ax, (bx + bw, by + bh / 2), (cx, cy + ch / 2), color=TOP_EDGE)

# ── Bottom row: fish fine-tuning / evaluation pipeline ───────────────────────
c1 = box(ax, (13, 2), 15, 9,
         "Fish 2D\nchromatograms\n(103 samples,\n4 species)",
         BOTTOM_COLOR, BOTTOM_EDGE)
c2 = box(ax, (32, 2), 20, 9,
         "5-fold stratified CV\n$\\times$ 10 seeds\n(50 train/test splits,\nSec.~V-C/D)",
         BOTTOM_COLOR, BOTTOM_EDGE)
c3 = box(ax, (57, 2), 16, 9,
         "Fine-tune on\ntrain folds, early\nstop on inner\nval split",
         BOTTOM_COLOR, BOTTOM_EDGE)
c4 = box(ax, (78, 2), 16, 9,
         "Evaluate on\nheld-out test\nfold: balanced\naccuracy",
         BOTTOM_COLOR, BOTTOM_EDGE)

for (bx, by, bw, bh), (cx, cy, cw, ch) in zip([c1, c2, c3], [c2, c3, c4]):
    arrow(ax, (bx + bw, by + bh / 2), (cx, cy + ch / 2), color=BOTTOM_EDGE)

# Checkpoint feeds the fine-tuning box.
ckpt_x = b5[0] + b5[2] / 2
arrow(ax, (ckpt_x, b5[1]), (c3[0] + c3[2] / 2, c3[1] + c3[3]),
      color=CKPT_EDGE, style="-|>")

fig.savefig("figs/pipeline.pdf", bbox_inches="tight", pad_inches=0.12)
fig.savefig("figs/pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print("Saved figs/pipeline.pdf and figs/pipeline.png")
