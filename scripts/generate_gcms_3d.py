import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

data_path = "data/fish_oil/chroma/1___SNA1_Frame_A.npz"
data = np.load(data_path)
chroma = data['chroma']

# Map bins to approximate real values based on user description
rt_vals = np.linspace(18.3, 18.8, chroma.shape[0])
mz_vals = np.linspace(50, 450, chroma.shape[1])

# Scale and smooth the 3D surface
Z_full = np.power(chroma, 0.4) # Power scaling to bring out smaller peaks
Z_full = gaussian_filter(Z_full, sigma=(1, 1.5))

# Calculate projections (averages) before downsampling
Z_mz_proj = np.mean(chroma, axis=0)
Z_rt_proj = np.mean(chroma, axis=1)

# Scale projections
Z_mz_proj = np.power(Z_mz_proj, 0.4)
Z_rt_proj = np.power(Z_rt_proj, 0.4)

Z_mz_proj = (Z_mz_proj / Z_mz_proj.max()) * Z_full.max()
Z_rt_proj = (Z_rt_proj / Z_rt_proj.max()) * Z_full.max()

# Downsample for the 3D surface mesh
stride_rt = 1
stride_mz = 4
rt_sub = rt_vals[::stride_rt]
mz_sub = mz_vals[::stride_mz]
Z_sub = Z_full[::stride_rt, ::stride_mz]

MZ, RT = np.meshgrid(mz_sub, rt_sub)

# Create figure with high DPI
fig = plt.figure(figsize=(7, 6), dpi=300)
ax = fig.add_subplot(111, projection='3d')

# ---------------------------------------------------------
# Clean up aesthetics to match OriginLab / high-end pubs
# ---------------------------------------------------------
# Remove default grey panes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Remove grid lines
ax.grid(False)

# Make pane edges invisible
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

# Make the Z-axis completely invisible (spine and ticks)
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.set_zticks([])

# Plot the blue 3D surface
# Use a slight light source (shade=True) for better 3D depth perception
ax.plot_surface(MZ, RT, Z_sub, color='dodgerblue', edgecolor='none', 
                alpha=0.9, antialiased=True, shade=True)

# Plot the 1D black traces on the background panes
# MZ projection on the back wall (where RT is maximum)
ax.plot(mz_vals, Z_mz_proj, zs=18.8, zdir='y', color='black', linewidth=1.2, alpha=0.9)
# RT projection on the side wall (where MZ is minimum)
ax.plot(rt_vals, Z_rt_proj, zs=50, zdir='x', color='black', linewidth=1.2, alpha=0.9)

# Set axes limits
ax.set_xlim(50, 450)
ax.set_ylim(18.3, 18.8)
ax.set_zlim(0, Z_full.max() * 1.05)

# Axis formatting
ax.set_xlabel('m/z', fontsize=11, labelpad=8)
ax.set_ylabel('Retention Time (min)', fontsize=11, labelpad=8)
ax.tick_params(axis='both', which='major', labelsize=9)

# Adjust viewing angle
ax.view_init(elev=30, azim=-55)

# Tight layout and save with transparent background
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('figs/gc-ms.png', bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
print("Saved pristine 3D gc-ms.png")
