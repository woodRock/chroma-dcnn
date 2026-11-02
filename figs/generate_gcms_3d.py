"""
Generate the raw 2D chromatogram figure (Fig. 1) for the paper, from a real
fish sample rather than synthetic peaks, with axis ranges that match the
real data: retention time 5-45 min (the 200-bin grid used throughout, see
scripts/11_saliency.py's RT_MIN) and m/z 50-543, the instrument's observed
signal range (Section V-A). We show a real RT sub-window with visible peaks
(the 36-45 min region, which Section VI-C's saliency analysis identifies as
the most informative band) rather than the full 5-45 min range, so
individual peaks stay distinguishable at this bin resolution.

Run from repo root:
  python3 figs/generate_gcms_3d.py
Output: figs/gc-ms.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FS_LABEL, FS_TICK, crop_to_content

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chroma_dcnn.downstream.fish_oil import load_fish_oil_chroma_features

# ── Real data: one sample, a real RT sub-window ──────────────────────────────
MZ_LO, MZ_HI = 50, 543          # real observed m/z range (Section V-A)
RT_LO, RT_HI = 36.0, 45.0       # real RT sub-window with visible peaks

feats, y = load_fish_oil_chroma_features("data/fish_oil")
idx = int(np.where(y == 3)[0][0])  # a Blue Cod sample (most abundant species)
chroma = feats["chroma_flat"][idx].reshape(200, 1000)  # [RT bin, m/z bin]

rt_full = np.linspace(5.0, 45.0, 200)
rt_mask = (rt_full >= RT_LO) & (rt_full <= RT_HI)
rt_vals = rt_full[rt_mask]
mz_vals = np.arange(MZ_LO, MZ_HI + 1)
Z = chroma[rt_mask][:, MZ_LO:MZ_HI + 1]
Z = Z / Z.max()

# ── Wall projections ─────────────────────────────────────────────────────────
proj_rt = gaussian_filter1d(Z.sum(axis=1), sigma=1.0)   # TIC: sum over m/z per RT bin
proj_mz = gaussian_filter1d(Z.sum(axis=0), sigma=1.0)   # sum spectrum: sum over RT per m/z bin
z_max = Z.max()
proj_rt = proj_rt / proj_rt.max() * z_max * 0.88
proj_mz = proj_mz / proj_mz.max() * z_max * 0.88

# X = RT (left/depth axis), Y = m/z (front/bottom axis)
RT_grid, MZ_grid = np.meshgrid(rt_vals, mz_vals)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection="3d")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.xaxis.pane.set_edgecolor("none")
ax.yaxis.pane.set_edgecolor("none")
ax.zaxis.pane.set_edgecolor("none")
ax.zaxis.line.set_color((1, 1, 1, 0))
ax.set_zticks([])

FACE = "#5b9bd5"
EDGE = "#4a8ac4"

ax.plot_surface(
    RT_grid, MZ_grid, Z.T,
    color=FACE, edgecolor=EDGE, linewidth=0.15, alpha=0.92, shade=True,
    rstride=1, cstride=1,
)

# Shown in red: TIC (sum over m/z), at the far m/z wall.
ax.plot(rt_vals, proj_rt, zs=MZ_HI, zdir="y",
        color="#d62728", linewidth=1.1, alpha=0.95, zorder=10)
# Shown in green: sum spectrum (sum over RT), at the near RT wall.
ax.plot(mz_vals, proj_mz, zs=RT_LO, zdir="x",
        color="#2ca02c", linewidth=1.1, alpha=0.95, zorder=10)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(RT_LO, RT_HI)
ax.set_ylim(MZ_LO, MZ_HI)
ax.set_zlim(0, z_max * 1.08)

# Ticks kept strictly inside the axis limits (not at RT_LO/MZ_LO) so the two
# axes' tick labels don't collide where they'd otherwise sit on top of each
# other at the shared front corner of the plot.
ax.set_xticks([37, 39, 41, 43, 45])
ax.set_yticks([100, 200, 300, 400, 500])

ax.set_xlabel("RT, min", fontsize=FS_LABEL + 2, labelpad=14)
ax.set_ylabel("m/z", fontsize=FS_LABEL + 2, labelpad=14)
ax.tick_params(axis="x", labelsize=FS_TICK + 1, pad=2)
ax.tick_params(axis="y", labelsize=FS_TICK + 1, pad=6)

ax.view_init(elev=28, azim=-52)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.05)
plt.savefig("figs/gc-ms.png",
            bbox_inches="tight", pad_inches=0.15,
            dpi=300, transparent=True)
plt.close(fig)
crop_to_content("figs/gc-ms.png")
print("Saved figs/gc-ms.png")
