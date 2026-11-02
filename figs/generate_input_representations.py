"""
Visualise the four input representations compared in the paper (Sec. V-B):
2D chromatogram, sum spectrum, max-projection, and chroma-PCA.

(a)-(c) use one real fish sample's preprocessed 2D chromatogram; (d) is a
PCA scatter over all 103 real samples, coloured by species, since chroma-PCA
is a dataset-level representation rather than a per-sample curve.

Run from repo root:
  python3 figs/generate_input_representations.py
Output: figs/input-representations.pdf and figs/input-representations.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_features

SPECIES = {0: "Snapper", 1: "Gurnard", 2: "Tarakihi", 3: "Blue Cod"}
COLOURS = {0: "#2196f3", 1: "#ff9800", 2: "#4caf50", 3: "#e91e63"}
FS_LABEL, FS_TICK, FS_TITLE = 14, 11, 16

feats, y = load_fish_oil_chroma_features("data/fish_oil")
chroma_flat = feats["chroma_flat"]          # [103, 200000] = 200 RT bins x 1000 m/z
max_proj = feats["max_proj"]                # [103, 1000]

# One example sample for the per-sample panels (a)-(c).
example = int(np.where(y == 0)[0][0])       # first Snapper sample
chroma_2d = chroma_flat[example].reshape(200, 1000)
sum_spectrum = chroma_2d.sum(axis=0)
sum_spectrum = sum_spectrum / (np.linalg.norm(sum_spectrum) + 1e-8)
max_spectrum = max_proj[example]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))

# (a) 2D chromatogram
ax = axes[0]
nz = chroma_2d[chroma_2d > 0]
vmax = np.percentile(nz, 90) if len(nz) else 1.0
im = ax.imshow(chroma_2d[:, :600], aspect="auto", origin="lower", cmap="Blues",
                vmin=0, vmax=vmax, extent=[0, 600, 0, 200])
ax.set_xlabel("$m/z$", fontsize=FS_LABEL)
ax.set_ylabel("RT bin", fontsize=FS_LABEL)
ax.set_title("(a) 2D chromatogram", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(labelsize=FS_TICK)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=FS_TICK - 2)
cb.set_label("$\\sqrt{\\mathrm{intensity}}$", fontsize=FS_LABEL - 2)

# (b) Sum spectrum
ax = axes[1]
ax.plot(np.arange(600), sum_spectrum[:600], color="#1f4e9d", linewidth=1.1)
ax.fill_between(np.arange(600), 0, sum_spectrum[:600], color="#1f4e9d", alpha=0.25)
ax.set_xlabel("$m/z$", fontsize=FS_LABEL)
ax.set_ylabel("Normalised intensity", fontsize=FS_LABEL)
ax.set_title("(b) Sum spectrum", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(labelsize=FS_TICK)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# (c) Max-projection
ax = axes[2]
ax.plot(np.arange(600), max_spectrum[:600], color="#c9680a", linewidth=1.1)
ax.fill_between(np.arange(600), 0, max_spectrum[:600], color="#c9680a", alpha=0.25)
ax.set_xlabel("$m/z$", fontsize=FS_LABEL)
ax.set_ylabel("Normalised intensity", fontsize=FS_LABEL)
ax.set_title("(c) Max-projection", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(labelsize=FS_TICK)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# (d) Chroma-PCA: all 103 samples, standardised then reduced to 2 PCs (shown
# here across the whole dataset for a legible scatter; the paper refits the
# scaler/PCA inside each CV fold, Sec. V-C).
ax = axes[3]
Xc = chroma_flat - chroma_flat.mean(axis=0, keepdims=True)
Xc = Xc / (chroma_flat.std(axis=0, keepdims=True) + 1e-8)
pcs = PCA(n_components=2, random_state=0).fit_transform(Xc)
for c in range(4):
    mask = y == c
    ax.scatter(pcs[mask, 0], pcs[mask, 1], color=COLOURS[c], label=SPECIES[c],
               s=28, alpha=0.85, edgecolor="white", linewidth=0.3)
ax.set_xlabel("PC 1", fontsize=FS_LABEL)
ax.set_ylabel("PC 2", fontsize=FS_LABEL)
ax.set_title("(d) Chroma-PCA", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(labelsize=FS_TICK)
ax.legend(frameon=False, fontsize=FS_TICK, loc="best")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig("figs/input-representations.pdf", bbox_inches="tight", pad_inches=0.1)
fig.savefig("figs/input-representations.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("Saved figs/input-representations.pdf and figs/input-representations.png")
