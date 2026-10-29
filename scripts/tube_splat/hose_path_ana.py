import sys
sys.path.insert(0, "/home/bubble/RoboRacer-3D-Isaac-Sim/scripts")
from generate_racelines import cal_raceline_info
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

# ---------- paths: only edit here when switching scene ----------
SCENE = "xgrid_wormhole"
SCENARIO_JSON = "/home/bubble/kaidong/xgrid_wormhole_map/scenario/racing_xgrid_wormhole.json"
WORK_DIR = "/home/bubble/minitools_data_ana/tube_splat"
AXES_PATH = f"{WORK_DIR}/hose_axes_{SCENE}.npz"

with open(SCENARIO_JSON, "r") as file:
    config = json.load(file)
    reference_path = np.array(config["reference_path"])


s, n_hat, _ = cal_raceline_info(reference_path)

new_pos_inner = reference_path + 0.95 * n_hat
new_pos_outer = reference_path - 0.95 * n_hat

# drawing figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_aspect("equal")
ax.plot(new_pos_inner[:, 0], new_pos_inner[:, 1], label="tube axis inner")
ax.plot(new_pos_outer[:, 0], new_pos_outer[:, 1], label="tube axis outter")
ax.plot(reference_path[:, 0], reference_path[:, 1], "--", label="center line")
ax.legend()
fig.savefig(f"{WORK_DIR}/raceline_with_tube_{SCENE}.png", dpi=150)

#inner
poly_inner = Polygon(new_pos_inner)
print(poly_inner.is_valid)    # checking whether the boundary of the polygon through polygon itself

R_MIN = 0.30
poly_round = poly_inner.buffer(-R_MIN).buffer(R_MIN)
print(poly_round.geom_type)
inner_round = np.array(poly_round.exterior.coords)[:-1] # left the last point out, because, is the same with the first one

# resample evenly: because buffer making the points distribute unevenly
line = LineString(np.vstack([inner_round, inner_round[:1]]))
inner_round = np.array([line.interpolate(t).coords[0] for t in np.linspace(0, line.length, 
                                                                           round(line.length/0.05), endpoint=False)])   # number of step is calculated by: L/0.05

# checking direction (clockwise or counter clockwise)
if not Polygon(inner_round).exterior.is_ccw:
    inner_round = inner_round[::-1]

# outer
poly_outer = Polygon(new_pos_outer)
print(poly_outer.is_valid)

poly_round = poly_outer.buffer(R_MIN).buffer(-R_MIN)
print(poly_round.geom_type)
outer_round = np.array(poly_round.exterior.coords)[:-1]

line = LineString(np.vstack([outer_round, outer_round[:1]]))
outer_round = np.array([line.interpolate(t).coords[0] for t in np.linspace(0, line.length, 
                                                                           round(line.length/0.05), endpoint=False)])

if not Polygon(outer_round).exterior.is_ccw:
    outer_round = outer_round[::-1]

np.savez(AXES_PATH,
         inner=inner_round, outer=outer_round)