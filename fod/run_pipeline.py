"""
run_pipeline.py -- end-to-end: scene -> voxel windows -> clusters -> 3D track.

    python -m fod.run_pipeline --dataset 3 --window 1.0 --voxel-size 1.0
    python -m fod.run_pipeline --dataset 5 ...      # once calibration+detections land
"""

import argparse
import json
import os
import numpy as np

from fod.sources.scene import Dataset3Scene, Dataset5Scene
from fod.voxel.grid import GridSpec, accumulate_windows
from fod.voxel.cluster import extract_clusters
from fod.track3d.kalman import Tracker3D


def build_scene(args):
    if args.dataset == 3:
        return Dataset3Scene(det_dir=args.det_dir, poses_json=args.poses)
    elif args.dataset == 5:
        return Dataset5Scene(root=args.det_dir, calib_dir=args.calib_dir,
                             rotations_json=args.poses)
    raise ValueError(f"dataset {args.dataset} not supported")


def estimate_volume(scene, t0, t1, n=200):
    """Rough flight volume from pairwise ray midpoints, used to size the grid."""
    pts = []
    for t in np.linspace(t0, t1, n):
        rays = scene.rays(t)
        if len(rays) < 3:
            continue
        O = np.array([r.origin for r in rays])
        D = np.array([r.direction for r in rays])
        A = np.zeros((3, 3)); b = np.zeros(3)
        for o, u in zip(O, D):
            P = np.eye(3) - np.outer(u, u)
            A += P; b += P @ o
        try:
            X = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if np.linalg.norm(X) < 500:
            pts.append(X)
    pts = np.array(pts)
    lo = np.percentile(pts, 2, axis=0)
    hi = np.percentile(pts, 98, axis=0)
    return lo, hi, pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=int, default=3)
    p.add_argument("--det-dir", default="/mnt/user-data/uploads")
    p.add_argument("--calib-dir", default="/mnt/user-data/uploads")
    p.add_argument("--poses", default="/mnt/user-data/uploads/solved_camera_poses.json")
    p.add_argument("--window", type=float, default=1.0, help="seconds")
    p.add_argument("--voxel-size", type=float, default=1.0)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--min-cameras", type=int, default=3)
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--ray-sigma", type=float, default=7.0,
                   help="per-camera orientation uncertainty in degrees; "
                        "7 = measured dataset 3 value, <1 once dataset 5 PnP lands")
    p.add_argument("--t-start", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--out", default="/home/claude/out")
    p.add_argument("--dump-grids", type=int, default=3,
                   help="write this many voxel_grid .bin files for the viewers")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene = build_scene(args)
    spt = scene.seconds_per_time_unit()

    lo_t, hi_t = scene.time_range()
    t0 = args.t_start if args.t_start is not None else lo_t
    t1 = args.t_end if args.t_end is not None else hi_t
    window_units = args.window / spt

    print(f"scene      : dataset {args.dataset}, {len(scene.cameras)} cameras")
    print(f"time axis  : {scene.time_unit()}")
    print(f"range      : {t0:.1f} .. {t1:.1f}  ({(t1-t0)*spt:.1f} s)")
    print(f"window     : {args.window}s = {window_units:.1f} time units")

    lo, hi, seed = estimate_volume(scene, t0, t1)
    spec = GridSpec.from_volume(lo, hi, voxel_size=args.voxel_size, margin=15.0)
    print(f"volume     : {np.round(lo,1)} .. {np.round(hi,1)}")
    print(f"grid       : N={spec.N}, voxel={spec.voxel_size} m, "
          f"center={np.round(spec.center,1)}, {spec.nbytes()/1e6:.1f} MB/window")
    print(f"             (original ray_voxel.cpp: N=500, voxel=6.0 -> 500.0 MB)")
    print(f"ray sigma  : {args.ray_sigma} deg -> cone radius {np.tan(np.radians(args.ray_sigma))*60:.1f} m at 60 m range")

    tracker = Tracker3D(gate=20.0, max_misses=3, dt=args.window, q=2.0, r=4.0)
    rows, n_dumped, n_windows, n_clusters = [], 0, 0, 0

    for ts, te, acc, n_rays in accumulate_windows(
            scene, spec, t0, t1, window_units,
            samples_per_window=args.samples, min_cameras=args.min_cameras,
            ray_sigma_deg=args.ray_sigma):
        n_windows += 1
        cl = extract_clusters(acc, ts, te, percentile=args.percentile,
                              min_cameras=args.min_cameras)
        n_clusters += len(cl)
        if n_dumped < args.dump_grids and cl:
            acc.write_bin(os.path.join(args.out, f"voxel_grid_{n_dumped:03d}.bin"))
            n_dumped += 1
        if cl:
            tracker.update(cl, (ts + te) * 0.5 * spt)
            c = cl[0]
            rows.append(dict(t=(ts + te) * 0.5 * spt,
                             x=float(c.position[0]), y=float(c.position[1]),
                             z=float(c.position[2]), n_cameras=c.n_cameras,
                             n_voxels=c.n_voxels, intensity=c.intensity,
                             extent=[float(v) for v in c.extent]))

    print(f"\nwindows    : {n_windows}")
    print(f"clusters   : {n_clusters}  ({n_clusters/max(n_windows,1):.2f} per window)")

    tracks = tracker.finished(min_points=3)
    print(f"tracks     : {len(tracks)}")
    for tr in sorted(tracks, key=lambda t: -len(t.times))[:5]:
        print("  " + tr.summary())

    with open(os.path.join(args.out, "detections_3d.json"), "w") as f:
        json.dump(rows, f, indent=1)

    best = max(tracks, key=lambda t: len(t.times)) if tracks else None
    if best:
        np.savetxt(os.path.join(args.out, "track_3d.csv"),
                   np.column_stack([best.times, np.array(best.filtered),
                                    best.speeds]),
                   delimiter=",", header="t,x,y,z,speed_mps", comments="")
    print(f"\nwrote      : {args.out}/detections_3d.json, track_3d.csv, "
          f"{n_dumped} voxel_grid_*.bin")
    return spec, rows, tracks


if __name__ == "__main__":
    main()
