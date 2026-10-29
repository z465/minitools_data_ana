"""Speed profile of a raceline scenario json (written by RoboRacer-3D-Isaac-Sim scripts/generate_racelines.py).

Prints a few numbers (speed range, share of the lap at v_max, lap time) and saves one figure: speed along the lap,
with the curvature on a second axis, so you can see where the car slows down for a bend.

    python ego_traj_ana.py <raceline.json>
    python ego_traj_ana.py <raceline.json> --a-lat 2.0 1.5 1.0     # also recompute the profile for other a_lat values

--a-lat imports generate_vel_profile from the simulator repo (--roboracer-root, default $ROBORACER_ROOT or
~/RoboRacer-3D-Isaac-Sim), so the comparison uses exactly the code that writes the json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lap_geometry(path):
    """Segment lengths L, arc length s at each point, and curvature kappa (same formula as generate_vel_profile)."""
    e = np.roll(path, -1, axis=0) - path
    L = np.linalg.norm(e, axis=1)
    s = np.concatenate([[0.0], np.cumsum(L[:-1])])
    h = np.arctan2(e[:, 1], e[:, 0])
    h_diff = (np.roll(h, -1) - h + np.pi) % (2 * np.pi) - np.pi    # heading change, wrapped into (-pi, pi]
    kappa = np.abs(h_diff) / L
    return L, s, kappa


def summary(v, L, label):
    v_max = v.max()
    at_max = L[v >= v_max - 0.05].sum() / L.sum() * 100     # share of the lap length, not of the points
    lap_time = (L / v).sum()
    print(f"{label:>14s}: v min {v.min():.2f}  median {np.median(v):.2f}  max {v_max:.2f} m/s, "
          f"{at_max:.0f}% of the lap at v_max, lap time {lap_time:.1f} s, mean speed {L.sum() / lap_time:.2f} m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path, help="raceline scenario json with reference_path and speed_profile")
    parser.add_argument("--a-lat", type=float, nargs="+", default=[],
                        help="recompute the speed profile with these lateral accelerations [m/s2] for comparison")
    parser.add_argument("--roboracer-root", type=Path,
                        default=Path(os.environ.get("ROBORACER_ROOT", Path.home() / "RoboRacer-3D-Isaac-Sim")),
                        help="RoboRacer-3D-Isaac-Sim checkout, only needed for --a-lat")
    parser.add_argument("--out", type=Path, help="figure path (default ~/Desktop/charts/<json name>_speed.png)")
    args = parser.parse_args()

    scenario = json.loads(args.json.read_text())
    path = np.array(scenario["reference_path"])
    v = np.array(scenario["speed_profile"])
    L, s, kappa = lap_geometry(path)

    summary(v, L, "in json")
    recomputed = {}
    if args.a_lat:
        sys.path.insert(0, str(args.roboracer_root / "scripts"))
        from generate_racelines import generate_vel_profile
        for a_lat in args.a_lat:
            recomputed[a_lat], _ = generate_vel_profile(path, a_lat=a_lat)
            summary(recomputed[a_lat], L, f"a_lat {a_lat:g}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s, v, color="black", linewidth=2, label="speed profile (json)")
    for a_lat, v_new in recomputed.items():
        ax.plot(s, v_new, "--", label=f"recomputed, a_lat = {a_lat:g} m/s²")
    ax.set_xlabel("distance along the lap s [m]")
    ax.set_ylabel("speed [m/s]")
    ax.set_xlim(0, L.sum())
    ax.grid(alpha=0.3)

    ax_k = ax.twinx()    # curvature on the right axis, drawn under the speed curves
    ax_k.fill_between(s, kappa, color="tab:blue", alpha=0.15, label="curvature")
    ax_k.set_ylabel("curvature κ [1/m]", color="tab:blue")
    ax_k.set_ylim(0, kappa.max() * 2.5)    # keep the curvature in the lower part of the figure
    ax.set_zorder(ax_k.get_zorder() + 1); ax.patch.set_visible(False)

    lines, labels = ax.get_legend_handles_labels()
    lines_k, labels_k = ax_k.get_legend_handles_labels()
    ax.legend(lines + lines_k, labels + labels_k, loc="lower right", fontsize=9)
    ax.set_title(f"{args.json.stem}: lap {L.sum():.1f} m, lap time {(L / v).sum():.1f} s")
    fig.tight_layout()

    out = args.out or Path.home() / "Desktop/charts" / f"{args.json.stem}_speed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
