# minitools_data_ana

Small analysis scripts for the RoboRacer / F1TENTH simulator data
([RoboRacer-3D-Isaac-Sim](https://github.com/fjahncke/RoboRacer-3D-Isaac-Sim), branch `data-generation`).
Each script does one job and saves its figures or files. Generated data goes to `outputs/`, which is not in git.

## What is here

### `scripts/tube_splat/`: hose scan and hoses along a track

The track boundaries in the simulator are hoses: a short scanned hose (a 3D Gaussian splat) is repeated along each
boundary. These scripts were used to check the scan and build the first version of the hoses.

| Script | What it does |
|---|---|
| `tube_splat_ana.py` | Loads the hose scan (`.ply`) and prints/plots how its Gaussians are distributed: shapes, scales, opacities, positions. |
| `calibrated_tube_ana.py` | Plots the aligned hose (`tube_pipe_aligned.npz`) from the top and the side, the number of Gaussians along the hose, and a close-up of both ends, to find the rib period and where to cut one tile. |
| `hose_path_ana.py` | From a scenario's centre line, computes the two hose axes (inner and outer boundary, rounded corners, evenly resampled) and saves them as `hose_axes_<scene>.npz`. |
| `hose_sweep.py` | Repeats the hose tile along both axes (bending it with the track) and saves all hoses as one splat `.npz` for the scenario's `static_objects`. |

`hose_path_ana.py` and `hose_sweep.py` are now part of the simulator repo as `scripts/hoses.py`
(`generate_hose_axis` + `sweep_tube`), which `generate_tracks.py` calls for every generated track.
The versions here are kept as the original, step-by-step working.

### `scripts/racing_data/ego_racing_data/`: driving data

| Script | What it does |
|---|---|
| `ego_traj_ana.py` | Speed profile of a raceline scenario json: prints the speed range, the share of the lap at top speed and the lap time, and plots speed and curvature along the lap. With `--a-lat` it recomputes the profile for other lateral accelerations to compare them. |

```bash
python scripts/racing_data/ego_racing_data/ego_traj_ana.py <raceline.json>
python scripts/racing_data/ego_racing_data/ego_traj_ana.py <raceline.json> --a-lat 2.0 1.5 1.0
```

The figure is saved to `~/Desktop/charts/<json name>_speed.png` (change it with `--out`).

## Requirements

- Python with `numpy`, `scipy`, `matplotlib`, `shapely` (the simulator's `roboracer` environment has all of them).
- A checkout of RoboRacer-3D-Isaac-Sim: the scripts import its code (`object_splats`, `generate_racelines`).
  `ego_traj_ana.py` finds it through `--roboracer-root`, `$ROBORACER_ROOT` or `~/RoboRacer-3D-Isaac-Sim`.

The `tube_splat` scripts still have their input and output paths written at the top of each file
(absolute paths on the machine they were written on); edit them before running elsewhere.
