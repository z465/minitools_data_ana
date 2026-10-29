import sys
sys.path.insert(0, "/home/bubble/RoboRacer-3D-Isaac-Sim/simulation/isaac_sim")
import object_splats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

obj = object_splats.ObjectSplat.load("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/tube_pipe_aligned.npz")

means = obj.means

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# x-y
ax1.scatter(means[:, 0], means[:, 1], s=0.3, alpha=0.2)
ax1.set_aspect("equal")
ax1.set_title("x-y Plane")

# x-z
ax2.scatter(means[:, 0], means[:, 2], s=0.3, alpha=0.2)
ax2.set_aspect("equal")
ax2.set_title("x-z Plane")

fig.savefig("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/calibrated_tube.png", dpi=150)


# inspect length of waves
counts, edges = np.histogram(means[:, 0], bins=500, range=(-0.5, 0.5))
centers = (edges[:-1] + edges[1:]) / 2
fig2, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(centers, counts)
ax3.set_xlim(0.2, 0.5)

fig2.savefig("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/splat_count.png", dpi=150)

# smoothing counts
counts_smooth = uniform_filter1d(counts, 3)

fig3, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4))
axL.plot(centers, counts_smooth)
axL.set_xlim(-0.52, -0.40)
axR.plot(centers, counts_smooth)
axR.set_xlim(0.40, 0.52)

fig3.savefig("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/detail.png", dpi=150)
