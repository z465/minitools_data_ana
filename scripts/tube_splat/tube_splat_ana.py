import sys
import numpy as np
sys.path.insert(0, "/home/bubble/RoboRacer-3D-Isaac-Sim/simulation/isaac_sim")
import object_splats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

splat_info_dict = object_splats.load_splat_ply("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/tube_pipe_3dgs.ply")


print(f"Mean of the Gaussian object is: {splat_info_dict['means']}")
for key, value in splat_info_dict.items():
    print(f"Shape of {key} is: {value.shape}")

print(f"scales_log: {splat_info_dict['scales_log']}")
print(f"quats: {splat_info_dict['quats']}")
print(f"opacities: {splat_info_dict['opacities']}")
print(f"sh: {splat_info_dict['sh']}")



# transparent number
opac = 1 / (1 + np.exp(-splat_info_dict["opacities"]))
print(f"max of opacities after sigmoid: {opac.max()}")
print(f"min of opacities after sigmoid: {opac.min()}")

# mask out opacities lower than 0.3
mask = opac > 0.2
print(f"mask: {mask}")
means_masked = splat_info_dict["means"][mask]
scales_log_masked = splat_info_dict["scales_log"][mask]

print(f"shape of means after masking: {means_masked.shape}")
print(f"shape of scales_log after masking: {scales_log_masked.shape}")


# distribution inspection
for i in range(3):
    axis = means_masked[:, i]
    print(f"axis: {i}, max value: {axis.max()}, min value: {axis.min()}")


# remove discret points
median = np.median(means_masked, axis=0)
print(f"median: {median}")

median = median.reshape(1, -1)
l = np.linalg.norm((means_masked - median), axis=1)
print(f"distance with median: {l}")

# percentil
print(f"50%: {np.percentile(l, 50)}, 80%: {np.percentile(l, 80)}, 90%: {np.percentile(l, 90)}, 93%: {np.percentile(l, 93)}, 94%: {np.percentile(l, 94)}")

# using results from above, get the parameter mask for filtering out far distance from median (l > 10)
mask_distance = (l < 10)

means_scanned = means_masked[mask_distance]
print(f"mean values of the scanned object is: {means_scanned}, shape: {means_scanned.shape}")
for i in range(3):
    print(f"axis{i}, max: {means_scanned[:, i].max()}, min: {means_scanned[:, i].min()}")


# plotting scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# x-z
ax1.scatter(means_scanned[:, 0], means_scanned[:, 2], s=0.3, alpha=0.2)
ax1.set_aspect("equal")
ax1.set_title("x-z Plane")
ax1.set_xlim(-0.9, 0.87)
ax1.set_ylim(-0.1, 0.4)

# x-y
ax2.scatter(means_scanned[:, 0], means_scanned[:, 1], s=0.3, alpha=0.2)
ax2.set_aspect("equal")
ax2.set_title("x-y Plane")
ax2.set_xlim(-0.9, 0.87)
ax2.set_ylim(-0.3, 0.3)

fig.savefig("/home/bubble/minitools_data_ana/outputs/tube_pipe_3dgs/view.png", dpi=150)

# Making assets: so using all means value
# cropping splatting objects
means = splat_info_dict["means"]
mask_x = (-1.3 < means[:, 0]) & (means[:, 0] < 1.3)
mask_y = (-0.5 < means[:, 1]) & (means[:, 1] < 0.5)
mask_z = (-0.2 < means[:, 2]) & (means[:, 2] < 0.6)

mask = mask_x & mask_y & mask_z

means_crop = means[mask]
    
