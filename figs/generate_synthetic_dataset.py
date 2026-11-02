"""
Generate the "synthetic chromatogram generation" figure (Fig. 3) for the paper.

Four panels, left to right:
  1. A handful of reference EI-MS spectra sampled from the MoNA/MassBank corpus.
  2. The same spectra ("Sample K reference spectra").
  3. A random Gaussian elution profile assigned to each one.
  4. The resulting synthetic 2D chromatogram (their sum).

Mirrors the generation procedure in src/chroma_dcnn/training/pretrain_chroma.py
(Sample K ~ Uniform{5,...,30}; mu_k ~ Uniform(0, 200); sigma_k ~ Uniform(1, 15)).

Run from repo root:
  python3 figs/generate_synthetic_dataset.py
Output: figs/synthetic-dataset.png and figs/synthetic-dataset.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

rng = np.random.default_rng(7)

N_BINS = 200     # retention-time bins (matches configs/pretrain.yaml model.n_bins)
MZ_MAX = 1000    # m/z channels
K = 4            # spectra shown (full range is Uniform{5,...,30}; 4 keeps panels legible)

COLORS = ["#6a3d9a", "#33a02c", "#e08214", "#1f77b4"]


def make_spectrum(n_peaks):
    mz = rng.integers(20, 500, n_peaks)
    inten = rng.exponential(0.6, n_peaks)
    inten /= inten.max()
    spec = np.zeros(MZ_MAX)
    spec[mz] = inten
    return spec


spectra = [make_spectrum(rng.integers(10, 25)) for _ in range(K)]
mus = rng.uniform(20, 180, K)
sigmas = rng.uniform(3, 12, K)

t = np.arange(N_BINS)
elution = [np.exp(-0.5 * ((t - mu) / sig) ** 2) for mu, sig in zip(mus, sigmas)]

X = np.zeros((N_BINS, MZ_MAX))
for e, s in zip(elution, spectra):
    X += np.outer(e, s)

FS_LABEL, FS_TICK, FS_TITLE, FS_NOTE = 15, 12, 17, 12

fig = plt.figure(figsize=(16, 5.6))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 0.12], hspace=0.55, wspace=0.42,
                        top=0.80, bottom=0.17, left=0.045, right=0.985)

titles = ["MoNA / MassBank\nEI-MS libraries", "Sample $K$\nreference spectra",
          "Assign Gaussian\nelution profiles", "Synthetic\nchromatogram"]
notes = ["No fish data used\nin pretraining",
         r"$K \sim \mathrm{Uniform}\{5,\dots,30\}$",
         r"$\mu_k \sim \mathrm{Uniform}(0,200)$" "\n" r"$\sigma_k \sim \mathrm{Uniform}(1,15)$",
         "sqrt + L2 normalisation"]

panel_axes = {}

# ── Panel 1 & 2: stacked reference spectra (library pool, then the K sampled) ──
for col, jitter in ((0, True), (1, False)):
    ax = fig.add_subplot(gs[0, col])
    panel_axes[col] = ax
    for i, (spec, c) in enumerate(zip(spectra, COLORS)):
        offset = (K - 1 - i) * 1.25
        mz_idx = np.nonzero(spec)[0]
        ax.vlines(mz_idx, offset, offset + spec[mz_idx], color=c, linewidth=1.3)
        ax.axhline(offset, color=c, linewidth=0.4, alpha=0.3)
    if col == 0:
        ax.text(MZ_MAX * 0.5, (K - 1) * 1.25 + 1.5, "$\\vdots$", fontsize=FS_TITLE,
                 ha="center")
    ax.set_xlim(0, MZ_MAX)
    ax.set_ylim(-0.3, (K - 1) * 1.25 + 1.6)
    ax.set_xlabel("$m/z$", fontsize=FS_LABEL)
    if col == 0:
        ax.set_ylabel("Intensity", fontsize=FS_LABEL)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=FS_TICK)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(titles[col], fontsize=FS_TITLE, fontweight="bold")
    note_ax = fig.add_subplot(gs[1, col])
    note_ax.axis("off")
    note_ax.text(0.5, 0.5, notes[col], ha="center", va="center", fontsize=FS_NOTE,
                  transform=note_ax.transAxes,
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888",
                            linestyle="dashed"))

# ── Panel 3: Gaussian elution profiles ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
panel_axes[2] = ax
for i, (e, c) in enumerate(zip(elution, COLORS)):
    offset = (K - 1 - i) * 1.25
    ax.plot(t, offset + e, color=c, linewidth=1.6)
    ax.fill_between(t, offset, offset + e, color=c, alpha=0.25)
ax.set_xlim(0, N_BINS)
ax.set_ylim(-0.3, (K - 1) * 1.25 + 1.6)
ax.set_xlabel("RT bin", fontsize=FS_LABEL)
ax.set_yticks([])
ax.tick_params(axis="x", labelsize=FS_TICK)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.set_title(titles[2], fontsize=FS_TITLE, fontweight="bold")
note_ax = fig.add_subplot(gs[1, 2])
note_ax.axis("off")
note_ax.text(0.5, 0.5, notes[2], ha="center", va="center", fontsize=FS_NOTE,
              transform=note_ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888",
                        linestyle="dashed"))

# ── Panel 4: synthetic 2D chromatogram (coarse grid: schematic, not real data) ──
ax = fig.add_subplot(gs[0, 3], projection="3d")
panel_axes[3] = ax
S_T, S_M = 2, 5   # downsample for a light, fast-rendering vector figure
t_s = t[::S_T]
mz_s = np.arange(MZ_MAX)[::S_M]
Zs = X[::S_T, ::S_M]
T_grid, M_grid = np.meshgrid(t_s, mz_s, indexing="ij")
ax.plot_surface(T_grid, M_grid, Zs, cmap="Blues", linewidth=0, antialiased=True,
                 rstride=1, cstride=1)
ax.set_xlabel("RT bin", fontsize=FS_LABEL - 2, labelpad=8)
ax.set_ylabel("$m/z$", fontsize=FS_LABEL - 2, labelpad=8)
ax.set_zticks([])
ax.tick_params(axis="both", labelsize=FS_TICK - 2)
ax.view_init(elev=32, azim=-60)
ax.set_title(titles[3], fontsize=FS_TITLE, fontweight="bold", pad=0)
note_ax = fig.add_subplot(gs[1, 3])
note_ax.axis("off")
note_ax.text(0.5, 0.5, notes[3], ha="center", va="center", fontsize=FS_NOTE,
              transform=note_ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888888",
                        linestyle="dashed"))

fig.text(0.5, 0.97, f"{N_BINS} RT bins $\\times$ {MZ_MAX} $m/z$",
          fontsize=FS_NOTE, ha="center", style="italic")

# Arrows between consecutive panels, placed from their actual rendered positions
# (figure-fraction x, at vertical mid-height of the top row).
fig.canvas.draw()
top_y = panel_axes[0].get_position().y0 + 0.5 * panel_axes[0].get_position().height
for a, b, label in ((0, 1, None), (1, 2, None), (2, 3, "sum")):
    x0 = panel_axes[a].get_position().x1
    x1 = panel_axes[b].get_position().x0
    xm = 0.5 * (x0 + x1)
    fig.text(xm, top_y, r"$\rightarrow$", fontsize=22, ha="center", va="center")
    if label:
        fig.text(xm, top_y + 0.1, label, fontsize=FS_NOTE, ha="center", va="bottom",
                  style="italic")

fig.savefig("figs/synthetic-dataset.pdf", bbox_inches="tight", pad_inches=0.1)
fig.savefig("figs/synthetic-dataset.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
plt.close(fig)
print("Saved figs/synthetic-dataset.pdf and figs/synthetic-dataset.png")
