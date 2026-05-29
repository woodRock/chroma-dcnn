"""
Generate a 3D GC-MS chromatogram figure for the paper.

Uses a synthetic chromatogram built from Gaussian peaks to reproduce the
isolated-pyramid style of the original figure.  The synthetic data matches
the statistical properties of the fish oil data (sparse peaks, wide dynamic
range, m/z 50-450, typical lipid RT distribution) without requiring the
uncommitted raw CSV scans.

Run from repo root:
  python3 figs/generate_gcms_3d.py
Output: figs/gc-ms.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Synthetic chromatogram ─────────────────────────────────────────────────────
rng = np.random.default_rng(42)

N_RT  = 300     # fine RT grid for sharp peaks
N_MZ  = 400
RT_LO, RT_HI = 18.3, 18.8
MZ_LO, MZ_HI = 50, 450

rt_vals = np.linspace(RT_LO, RT_HI, N_RT)
mz_vals = np.linspace(MZ_LO, MZ_HI, N_MZ)

Z = np.zeros((N_RT, N_MZ), dtype=np.float64)

# Place ~45 compounds, each with a Gaussian RT profile and sparse mass spectrum
N_COMPOUNDS = 90
for _ in range(N_COMPOUNDS):
    rt_c   = rng.uniform(0.04, 0.96) * N_RT          # RT centre (bins)
    rt_sig = rng.uniform(2.5, 5.0)                    # narrow enough for isolated peaks
    amp    = rng.exponential(0.35) + 0.05
    amp    = min(amp, 1.0)

    # 1–4 sparse fragment ions — keeps peaks narrow and isolated in m/z
    mz_base = rng.uniform(0.05, 0.90) * N_MZ
    n_frags  = rng.integers(1, 5)
    frag_mz  = np.clip(
        mz_base + rng.normal(0, 10, n_frags),
        0, N_MZ - 1
    ).astype(int)
    frag_int = rng.exponential(0.5, n_frags)
    frag_int /= frag_int.max()

    # Gaussian RT envelope
    rt_idx = np.arange(N_RT)
    envelope = np.exp(-0.5 * ((rt_idx - rt_c) / rt_sig) ** 2)

    for j, mz_i in enumerate(frag_mz):
        Z[:, mz_i] += envelope * amp * frag_int[j]

# Normalise and clip
Z = np.clip(Z / Z.max(), 0, 1)

# ── Wall projections ─────────────────────────────────────────────────────────
from scipy.ndimage import gaussian_filter1d
proj_rt = Z.max(axis=1)     # chromatogram (TIC-like) — left wall
proj_mz = Z.max(axis=0)     # mass spectrum — back wall

# Light smoothing on RT projection so it looks like a clean chromatogram trace
proj_rt = gaussian_filter1d(proj_rt, sigma=1.5)
proj_mz = gaussian_filter1d(proj_mz, sigma=1.0)

z_max   = Z.max()
proj_rt = proj_rt / proj_rt.max() * z_max * 0.88
proj_mz = proj_mz / proj_mz.max() * z_max * 0.88

# ── Downsample ────────────────────────────────────────────────────────────────
S_RT, S_MZ = 1, 1
rt_s = rt_vals[::S_RT]
mz_s = mz_vals[::S_MZ]
Z_s  = Z[::S_RT, ::S_MZ]

# X = RT (left/depth axis, like original), Y = m/z (front/bottom axis, like original)
# meshgrid(rt_s, mz_s) → RT_grid[i,j]=rt_s[j], MZ_grid[i,j]=mz_s[i]
# Z_s.T[i,j] = Z_s[j,i] = value at rt_s[j], mz_s[i]  ✓
RT_grid, MZ_grid = np.meshgrid(rt_s, mz_s)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), dpi=300)
ax  = fig.add_subplot(111, projection="3d")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)
ax.xaxis.pane.set_edgecolor("none")
ax.yaxis.pane.set_edgecolor("none")
ax.zaxis.pane.set_edgecolor("none")
ax.zaxis.line.set_color((1, 1, 1, 0))
ax.set_zticks([])

# ── Surface ───────────────────────────────────────────────────────────────────
FACE = "#5b9bd5"
EDGE = "#4a8ac4"

ax.plot_surface(
    RT_grid, MZ_grid, Z_s.T,   # X=RT, Y=m/z; Z transposed to match
    color=FACE,
    edgecolor=EDGE,
    linewidth=0.15,
    alpha=0.92,
    shade=True,
    rstride=1,
    cstride=1,
)

# ── Wall projections ─────────────────────────────────────────────────────────
# Chromatogram (RT profile) — far m/z wall (Y=MZ_HI=450), traces in X-Z plane
# Shown in red: represents the max-projection / TIC along the RT axis
ax.plot(rt_vals, proj_rt, zs=MZ_HI, zdir="y",
        color="#d62728", linewidth=1.1, alpha=0.95, zorder=10)

# Mass spectrum — near RT wall (X=RT_LO=18.3), traces in Y-Z plane
# Shown in blue-grey: represents the sum-spectrum / max-projection along RT
ax.plot(mz_vals, proj_mz, zs=RT_LO, zdir="x",
        color="#2ca02c", linewidth=1.1, alpha=0.95, zorder=10)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(RT_LO, RT_HI)
ax.set_ylim(MZ_LO, MZ_HI)
ax.set_zlim(0, z_max * 1.08)

ax.set_xlabel("RT, min",  fontsize=11, labelpad=10)
ax.set_ylabel("m/z",      fontsize=11, labelpad=10)
ax.tick_params(axis="both", labelsize=8)

ax.view_init(elev=28, azim=-52)

plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.05)
plt.savefig("figs/gc-ms.png",
            bbox_inches="tight", pad_inches=0.15,
            dpi=300, transparent=True)
print("Saved figs/gc-ms.png")
