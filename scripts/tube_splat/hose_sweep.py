import numpy as np
import sys
sys.path.insert(0, "/home/bubble/RoboRacer-3D-Isaac-Sim/simulation/isaac_sim")
import object_splats as obs

# ---------- paths: only edit here when switching scene or hose tile ----------
SCENE = "xgrid_wormhole"            # which track the hose axes belong to
TILE_NAME = "hose_tile"             # which hose scan is used as the tile
WORK_DIR = "/home/bubble/minitools_data_ana/tube_splat"
OBJ_DIR = "/home/bubble/RoboRacer-3D-Isaac-Sim/data/local/objects"
AXES_PATH = f"{WORK_DIR}/hose_axes_{SCENE}.npz"           # written by hose_path_ana.py
TILE_PATH = f"{OBJ_DIR}/{TILE_NAME}.npz"                  # written by import_object_splat.py
OUT_PATH = f"{OBJ_DIR}/hoses_{SCENE}_{TILE_NAME}.npz"     # written by this script

hose_axes = np.load(AXES_PATH)
obj = obs.ObjectSplat.load(TILE_PATH)
tube_left_end = obj.meta["segment_m"][0]
tube_right_end = obj.meta["segment_m"][1]
TILE_LEN = tube_right_end - tube_left_end

inner_axis = hose_axes["inner"]
outer_axis = hose_axes["outer"]

inner_seg = np.linalg.norm((np.roll(inner_axis, -1, axis=0) - inner_axis), axis=1)
L_inner = np.sum(inner_seg)     # perimeter of inner 

outer_seg = np.linalg.norm((np.roll(outer_axis, -1, axis=0) - outer_axis), axis=1)
L_outer = np.sum(outer_seg)

K_inner = round(L_inner / TILE_LEN)
K_outer = round(L_outer / TILE_LEN)

tile_len_inner = L_inner / K_inner
tile_len_outer = L_outer / K_outer

stretch_inner = tile_len_inner / TILE_LEN
stretch_outer = tile_len_outer / TILE_LEN

inner_closed = np.vstack([inner_axis, inner_axis[:1]])
outer_closed = np.vstack([outer_axis, outer_axis[:1]])

print(f"stretch number of inner bound: {stretch_inner}")
print(f"stretch number of outer bound: {stretch_outer}")
print(inner_seg.shape)

# s
s_inner = np.zeros(len(inner_seg) + 1)
s_inner[1:] = np.cumsum(inner_seg, axis=0)

s_outer = np.zeros(len(outer_seg) + 1)
s_outer[1:] = np.cumsum(outer_seg, axis=0)

# directions: t_hat, n_hat

t_hat_inner = (np.roll(inner_axis, -1, axis=0) - inner_axis) / inner_seg.reshape(-1, 1)
t_hat_outer = (np.roll(outer_axis, -1, axis=0) - outer_axis) / outer_seg.reshape(-1, 1)

n_hat_inner = np.stack([-t_hat_inner[:, 1], t_hat_inner[:, 0]], axis=-1)
n_hat_outer = np.stack([-t_hat_outer[:, 1], t_hat_outer[:, 0]], axis=-1)

def pos_at(s_query, s, pts):
    x = np.interp(s_query, s, pts[:, 0])    # mapping from s to e.g. pts[:, 0] at s_query
    y = np.interp(s_query, s, pts[:, 1])
    return np.stack([x, y], axis=-1)

def dir_at(s_query, s, t_hat, n_hat):
    index_insert = np.searchsorted(s, s_query, side='right')
    seg_index = (index_insert - 1) % len(n_hat)
    return t_hat[seg_index], n_hat[seg_index]

# start allocating the first tile of tube
def allocate_tube(tube_asset_path, s_start, tube_left_end, stretch, s, closed_pts, t_hat, n_hat):
    tube_asset = np.load(tube_asset_path)
    # x of tube asset -> s
    means = tube_asset["means"]
    quats = tube_asset["quats"]
    sh = tube_asset["sh"]
    scales_log = tube_asset["scales_log"]
    opacities = tube_asset["opacities"]
    x = means[:, 0]
    s_query = s_start + (x - tube_left_end) * stretch
    # query for position at tube axis map and its direction n_hat
    pos_map = pos_at(s_query, s, closed_pts)  # array with x, y info
    t_hat_map, n_hat_map = dir_at(s_query, s, t_hat, n_hat)

    new_pos = pos_map + means[:, 1].reshape(-1, 1) * n_hat_map

    means = np.column_stack((new_pos, means[:, 2]))

    # rotate quats and sh
    axis_angles = np.arctan2(t_hat_map[:, 1], t_hat_map[:, 0])
    axis_quat = np.zeros((len(axis_angles), 4))
    axis_quat[:, 0] = np.cos(axis_angles / 2)
    axis_quat[:, 3] = np.sin(axis_angles / 2)
    gaussian_quat = quats
    quats = obs.quat_mul(axis_quat, gaussian_quat)

    
    axis_angles_unique = np.unique(axis_angles)
    for i in range(len(axis_angles_unique)):
        mask_angle = (axis_angles == axis_angles_unique[i])
        R_z_i = obs.rot_z(axis_angles_unique[i])
        sh[mask_angle] = obs.rotate_sh(sh[mask_angle], R_z_i)    


    return obs.ObjectSplat(means, scales_log, quats, opacities, sh)


# odometry calculation
def cal_odom(pts):
    seg = np.linalg.norm((np.roll(pts, -1, axis=0) - pts), axis=1)
    s = np.zeros(len(seg) + 1)
    s[1:] = np.cumsum(seg, axis=0)
    return s

def main():
    # inner
    tube_asset_path = TILE_PATH
    obj_inner_list = []
    for k in range(K_inner):
        s_start = k * tile_len_inner
        obj_inner_list.append(allocate_tube(tube_asset_path, s_start, tube_left_end, stretch_inner, s_inner, inner_closed, t_hat_inner, n_hat_inner))
    
    means_inner = [obj.means for obj in obj_inner_list]
    scales_log_inner = [obj.scales_log for obj in obj_inner_list]
    quats_inner = [obj.quats for obj in obj_inner_list]
    opacities_inner = [obj.opacities for obj in obj_inner_list]
    sh_inner = [obj.sh for obj in obj_inner_list]


    obj_outer_list = []
    for k in range(K_outer):
        s_start = k * tile_len_outer
        obj_outer_list.append(allocate_tube(tube_asset_path, s_start, tube_left_end, stretch_outer, s_outer, outer_closed, t_hat_outer, n_hat_outer))
    
    means_outer = [obj.means for obj in obj_outer_list]
    scales_log_outer = [obj.scales_log for obj in obj_outer_list]
    quats_outer = [obj.quats for obj in obj_outer_list]
    opacities_outer = [obj.opacities for obj in obj_outer_list]
    sh_outer = [obj.sh for obj in obj_outer_list]


    all_means = means_inner + means_outer
    all_scales_log = scales_log_inner + scales_log_outer
    all_quats = quats_inner + quats_outer
    all_opacities = opacities_inner + opacities_outer
    all_sh = sh_inner + sh_outer

    all_means = np.concatenate(all_means, axis=0)
    all_scales_log = np.concatenate(all_scales_log, axis=0)
    all_quats = np.concatenate(all_quats, axis=0)
    all_opacities = np.concatenate(all_opacities)
    all_sh = np.concatenate(all_sh, axis=0)

    all_obj = obs.ObjectSplat(all_means, all_scales_log, all_quats, all_opacities, all_sh)

    all_obj.save(OUT_PATH)


if __name__ == "__main__":
    main()