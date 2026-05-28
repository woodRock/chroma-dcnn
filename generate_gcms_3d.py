import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_path = "data/fish_oil/chroma/1___SNA1_Frame_A.npz"
data = np.load(data_path)
chroma = data['chroma']

# The shape is (200, 1000). A full 3D surface of 200,000 points might be too dense/slow
# to render nicely or might look like a black blob of edges.
# Let's downsample slightly for the visualization to look like a clean 3D mesh.
stride_rt = 2
stride_mz = 10

Z = chroma[::stride_rt, ::stride_mz]
Z = np.log1p(Z) # Log scale to make peaks visible against baseline

rt_bins = np.arange(Z.shape[0])
mz_bins = np.arange(Z.shape[1])
X, Y = np.meshgrid(mz_bins, rt_bins)

fig = plt.figure(figsize=(8, 8), dpi=300)
# Create a 3D axis with a specific viewing angle to mimic typical GC-MS 3D plots
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# cmap='viridis' or 'plasma' or 'jet' are good for this.
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)

# Remove axes/grids to make it look like a clean floating object for the architecture diagram
ax.set_axis_off()

# Set a nice viewing angle
ax.view_init(elev=35, azim=45)

# Save with a tight bounding box and transparent background
plt.savefig('figs/gc-ms.png', transparent=True, bbox_inches='tight', pad_inches=0)
print("Saved 3D gc-ms.png")
