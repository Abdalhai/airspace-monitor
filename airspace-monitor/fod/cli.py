"""
Unified command-line interface for the airspace monitoring system.

DESIGN INTENT
-------------
One entry point, subcommands mapping to the three tiers of the system:

    detect / angular   Tier 1 -- single camera, live or from file
    calibration        Tier 2 -- camera geometry, cached results
    reconstruct        Tier 3 -- multi-camera 3D, cached or live run
    demo               guided walkthrough for presentation

The cached commands exist because a full reconstruction takes minutes and a
presentation slot does not tolerate that. `demo` and the `--cached` paths load
precomputed output and render instantly; the live paths are there when someone
asks to see it actually compute.

WINDOWS NOTES
-------------
Two things break Python demos on Windows and both are handled here:

1. Console encoding. cmd.exe defaults to a code page that cannot encode the
   degree sign or box-drawing characters, and raises UnicodeEncodeError partway
   through printing -- which looks like a crash. _init_console() forces UTF-8
   where possible and falls back to ASCII output where not.

2. Multiprocessing. Windows spawns rather than forks, so any module-level code
   re-executes in every child process. Every entry point is behind
   `if __name__ == "__main__":` for that reason. Do not remove the guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# console setup -- must happen before any printing
# --------------------------------------------------------------------------

_ASCII_ONLY = False


def _init_console() -> None:
    """Make the console safe to print to on Windows."""
    global _ASCII_ONLY
    if sys.platform == "win32":
        os.system("")            # enables ANSI escape handling on Win10+
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            _ASCII_ONLY = True
    try:
        "°".encode(sys.stdout.encoding or "ascii")
    except Exception:
        _ASCII_ONLY = True


def deg(s: str = "") -> str:
    return ("deg" if _ASCII_ONLY else "°") + s


def rule(width: int = 72, ch: str = "-") -> str:
    return ch * width


def header(title: str) -> str:
    return f"\n{title}\n{rule(max(len(title), 40))}"


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def results_dir(override: str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("FOD_RESULTS")
    if env:
        return Path(env)
    return package_root()


# ==========================================================================
# TIER 1 -- single camera
# ==========================================================================

def cmd_angular(args) -> int:
    """Tier 1: real detector (fod.detect) + angular velocity.

    This calls the SAME pipeline as `python -m fod.detect.pipeline`. There is one
    detector in this project, not a demo copy and a real one.
    """
    import cv2
    from fod.angular import AngularVelocityEstimator, Calibration
    from fod.detect.pipeline import iter_frames, load_config
    from fod.detect.visualization import draw_tracks

    cfg = load_config(args.config) if args.config else _default_config()

    src = args.source
    if src.isdigit():
        cfg["source"] = {"type": "webcam", "index": int(src)}
        label = f"webcam {src}"
    elif src.startswith("rtsp://"):
        cfg["source"] = {"type": "rtsp", "url": src,
                         "reconnect_attempts": 3, "reconnect_delay_s": 1.0}
        label = src
    elif src.startswith("http://") or src.startswith("https://"):
        cfg["source"] = {"type": "ip_http", "url": src,
                         "reconnect_attempts": 3, "reconnect_delay_s": 1.0}
        label = src
    else:
        cfg["source"] = {"type": "file", "path": src,
                         "pace_realtime": args.pace_realtime}
        label = src

    # CLI overrides win over the config file.
    if args.min_area is not None:
        cfg["background_subtraction"]["min_blob_area"] = args.min_area
    if args.history is not None:
        cfg["background_subtraction"]["history"] = args.history
    if args.var_threshold is not None:
        cfg["background_subtraction"]["var_threshold"] = args.var_threshold
    if args.no_clahe:
        cfg["standardization"]["brightness_method"] = "none"
    cfg["output"]["show_window"] = False

    print(header("TIER 1 -- single-camera detection and angular velocity"))
    print(f"source     : {label}")
    print(f"brightness : {cfg['standardization']['brightness_method']}"
          f"   (clahe costs ~43% of the frame budget)")

    # ---- calibration --------------------------------------------------
    calib = None
    if args.calib:
        calib = Calibration.load(args.calib)
        print(f"calib      : fx={calib.fx:.1f} fy={calib.fy:.1f} "
              f"hfov={calib.hfov_deg:.1f}{deg()} at "
              f"{calib.width}x{calib.height} -> reporting {deg('/s')}")
    elif args.hfov:
        print(f"calib      : APPROXIMATE from hfov={args.hfov:.1f}{deg()} "
              f"-> {deg('/s')} good to ~10%")
    else:
        print(f"calib      : none -- reporting px/s, not {deg('/s')}")
        print("             (pass --calib phone.json; see `fod calibrate-camera`)")

    est = AngularVelocityEstimator(None, smoothing=args.smoothing)
    lookback = cfg["tracking"].get("direction_lookback", 5)

    print(f"\n{'frame':>7}  detections")
    print(rule())

    t0 = time.perf_counter()
    n = 0
    calib_ready = False

    try:
        for out in iter_frames(cfg, max_frames=args.max_frames or None):
            n = out.frame_index + 1

            # Calibration is bound on the first frame, once the native resolution
            # is actually known. Intrinsics are in PIXELS, so a calibration taken
            # at another resolution has to be rescaled -- reusing a mismatched K
            # is a silent error that yields plausible wrong angles.
            if not calib_ready:
                nw, nh = _native_size(out)
                if args.calib and calib is not None:
                    if (calib.width, calib.height) != (nw, nh):
                        print(f"note       : rescaling calibration "
                              f"{calib.width}x{calib.height} -> {nw}x{nh}")
                        calib = calib.scaled_to(nw, nh)
                    est.calib = calib
                elif args.hfov:
                    est.calib = Calibration.from_hfov(nw, nh, args.hfov)
                print(f"native     : {nw}x{nh}")
                calib_ready = True

            # Angular velocity uses NATIVE coordinates. Standardization is a speed
            # optimisation; if its scale leaked into the geometry every angle would
            # be wrong by the scale ratio and nothing would look broken.
            confirmed = {t.track_id: out.centroids_native[t.track_id]
                         for t in out.confirmed_tracks
                         if t.track_id in out.centroids_native}
            t_s = out.media_time_s if out.media_time_s >= 0 else out.wall_time_s
            readings = est.update(t_s, confirmed)

            if readings and out.frame_index % args.print_every == 0:
                for r in readings.values():
                    print(f"{out.native_frame_id:>7}  {est.format_line(r)}")

            if args.show:
                vis = draw_tracks(out.frame, out.tracks, mask=out.mask,
                                  lookback=lookback, only_confirmed=True)
                sx, sy = out.scale
                for tid, r in readings.items():
                    if r["rate"] is None:
                        continue
                    px = int(r["centroid"][0] * sx)
                    py = int(r["centroid"][1] * sy)
                    cv2.putText(vis, f"{r['rate']:.1f} {r['unit']}",
                                (px + 10, py + 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Tier 1 - detection + angular velocity", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        if isinstance(src, str) and src.startswith("http"):
            print("\nPhone streaming over the local network -- check:")
            print("  * phone and laptop on the SAME network")
            print("  * the IP has not changed (the MiFi reassigns it)")
            print("  * URL ends in the app's video path, often /video")
            print("  * Windows Firewall is not blocking Python")
        return 2
    finally:
        if args.show:
            cv2.destroyAllWindows()

    dt = time.perf_counter() - t0
    print(rule())
    print(f"processed {n} frames in {dt:.1f}s ({n / max(dt, 1e-9):.1f} fps)")
    if est.calib is None:
        print(f"\nReported in px/s. Physical {deg('/s')} needs the focal length: "
              "see `fod calibrate-camera --help`.")
    return 0


def _native_size(out) -> tuple:
    """Native sensor resolution, recovered from the standardized frame and scale."""
    h, w = out.frame.shape[:2]
    sx, sy = out.scale
    return int(round(w / sx)), int(round(h / sy))


def _default_config() -> dict:
    """Used when no --config is given, so the CLI runs from a bare checkout."""
    return {
        "source": {},
        "standardization": {"target_width": 960, "target_height": 540,
                            "brightness_method": "none", "clahe_clip_limit": 2.0,
                            "clahe_tile_grid": 8, "preserve_aspect": True},
        "background_subtraction": {"history": 500, "var_threshold": 16.0,
                                   "detect_shadows": False, "bg_sub_scale": 1.0,
                                   "morph_kernel_size": 3, "min_blob_area": 25},
        "tracking": {"max_match_distance": 60.0, "max_missed_frames": 10,
                     "history_len": 15, "direction_lookback": 5, "min_hits": 3},
        "output": {"show_window": False, "write_video_path": None},
    }


def cmd_calibrate_camera(args) -> int:
    """Calibrate a camera from checkerboard images."""
    import cv2
    import numpy as np
    from fod.angular import Calibration

    print(header("Camera calibration from checkerboard images"))
    imgs = sorted([p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not imgs:
        print(f"No images found in {args.images}")
        return 2

    nx, ny = args.corners
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * args.square_size

    objpoints, imgpoints = [], []
    shape = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(f"  {p.name:<32} {'found' if ok else 'NOT FOUND'}")
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            objpoints.append(objp)
            imgpoints.append(corners)

    print(f"\nusable images: {len(objpoints)} of {len(imgs)}")
    if len(objpoints) < 8:
        print("Need at least ~8 usable views for a stable result, ideally 15-20")
        print("spread across angles and across different parts of the frame.")
        if len(objpoints) < 4:
            return 2

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, shape, None, None)

    calib = Calibration(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                        shape[0], shape[1], dist)
    print(f"\nreprojection RMS : {rms:.3f} px "
          f"({'good' if rms < 1.0 else 'high -- check the board is rigid and flat'})")
    print(f"resolution       : {shape[0]}x{shape[1]}")
    print(f"fx, fy           : {calib.fx:.1f}, {calib.fy:.1f}")
    print(f"cx, cy           : {calib.cx:.1f}, {calib.cy:.1f}")
    print(f"horizontal FOV   : {calib.hfov_deg:.1f}{deg()}")
    if not (35.0 < calib.hfov_deg < 140.0):
        print("  WARNING: that FOV is outside the plausible range for a "
              "consumer camera.\n  Check the corner count matches your board "
              "(inner corners, not squares).")

    calib.save(args.output)
    print(f"\nwritten to {args.output}")
    return 0


# ==========================================================================
# TIER 2 -- geometry, cached
# ==========================================================================

CALIB_SUMMARY = {
    "dataset3": {
        "solved": "18 rotational DOF (6 cameras), positions known at metric scale",
        "method": "joint refinement over reprojection residual, "
                  "seeded from pairwise essential-matrix decomposition",
        "validation": "out-of-sample, 553 frames not used by the optimiser",
        "median_deg": 6.53,
        "mean_deg": 7.04,
        "p90_deg": 11.6,
        "p95_deg": 14.3,
        "mean_ray_miss_m": 6.72,
        "per_camera": [("cam5", 4.94), ("cam2", 5.36), ("cam0", 6.39),
                       ("cam4", 6.58), ("cam1", 7.91), ("cam3", 11.27)],
    }
}


def cmd_calibration(args) -> int:
    d = CALIB_SUMMARY["dataset3"]
    print(header("TIER 2 -- camera geometry (dataset 3)"))
    print(f"solved     : {d['solved']}")
    print(f"method     : {d['method']}")
    print(f"validated  : {d['validation']}")

    print(f"\norientation error")
    print(rule(52))
    print(f"  median              {d['median_deg']:6.2f}{deg()}")
    print(f"  mean                {d['mean_deg']:6.2f}{deg()}")
    print(f"  90th percentile     {d['p90_deg']:6.2f}{deg()}")
    print(f"  95th percentile     {d['p95_deg']:6.2f}{deg()}")
    print(f"  mean ray miss       {d['mean_ray_miss_m']:6.2f} m")

    print(f"\nper camera")
    print(rule(52))
    for name, err in d["per_camera"]:
        bar = "#" * int(err * 3)
        print(f"  {name}  {err:5.2f}{deg()}  {bar}")

    print(f"\nnotes")
    print(rule(52))
    print("  An earlier in-sample figure of 4.5deg was measured on the same")
    print("  points the optimiser was free to move, and understated the error.")
    print("  The 7.04deg figure above is measured on held-out frames and is the")
    print("  one the reconstruction accuracy is derived from.")
    print()
    print("  Error propagates linearly to position: 1deg -> 0.53 m,")
    print("  7deg -> 3.68 m for this camera geometry. There is no threshold")
    print("  effect, so voxel size is chosen from the calibration error rather")
    print("  than made as fine as memory allows.")
    print()
    print("  Reconstructions using only 2 cameras land underground 13.4% of")
    print("  the time; with 3 or more, 0.0%. The pipeline therefore requires")
    print("  3+ cameras per cluster, which costs ~15% of frames and removes")
    print("  every gross error.")
    return 0


# ==========================================================================
# TIER 3 -- 3D reconstruction
# ==========================================================================

def _load_track(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def cmd_reconstruct(args) -> int:
    rd = results_dir(args.results)
    track_csv = rd / "track_3d.csv"
    det_json = rd / "detections_3d.json"

    if not args.cached:
        print("Live reconstruction is available via fod.run_pipeline.")
        print("For presentation use --cached, which loads the stored result")
        print("instead of spending several minutes recomputing it.\n")
        return 2

    if not track_csv.exists():
        print(f"No cached result at {track_csv}")
        return 2

    rows = _load_track(track_csv)
    dets = json.loads(det_json.read_text(encoding="utf-8")) if det_json.exists() else []

    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    zs = [r["z"] for r in rows]
    sp = [r["speed_mps"] for r in rows if r["speed_mps"] > 0]
    ts = [r["t"] for r in rows]

    path_len = sum(
        ((rows[i]["x"] - rows[i - 1]["x"]) ** 2 +
         (rows[i]["y"] - rows[i - 1]["y"]) ** 2 +
         (rows[i]["z"] - rows[i - 1]["z"]) ** 2) ** 0.5
        for i in range(1, len(rows)))

    print(header("TIER 3 -- multi-camera 3D reconstruction (dataset 3)"))
    print(f"track points        {len(rows)}")
    print(f"duration            {ts[-1] - ts[0]:.1f} s "
          f"(t = {ts[0]:.1f} to {ts[-1]:.1f})")
    print(f"path length         {path_len:.1f} m")
    print(f"speed               mean {sum(sp)/len(sp):.2f} m/s, max {max(sp):.2f} m/s")
    print(f"extent  X           {min(xs):8.1f} to {max(xs):8.1f} m")
    print(f"        Y           {min(ys):8.1f} to {max(ys):8.1f} m")
    print(f"        Z           {min(zs):8.1f} to {max(zs):8.1f} m")

    if dets:
        ncam = [d["n_cameras"] for d in dets]
        nvox = [d["n_voxels"] for d in dets]
        print(f"\ncluster statistics")
        print(rule(52))
        print(f"  clusters accepted   {len(dets)}")
        print(f"  cameras per cluster min {min(ncam)}, "
              f"mean {sum(ncam)/len(ncam):.1f}, max {max(ncam)}")
        print(f"  voxels per cluster  min {min(nvox)}, "
              f"mean {sum(nvox)/len(nvox):.0f}, max {max(nvox)}")

    if args.plot:
        _ascii_plot(xs, ys, "top-down view (X-Y)", 62, 20)
        _ascii_plot(ts, zs, "altitude over time (t-Z)", 62, 14)

    print(f"\nstated accuracy     4-7 m, derived from the 7.04{deg()} "
          "orientation error")
    print("                    above and the linear error propagation")
    fig = rd / "reconstruction.png"
    if fig.exists():
        print(f"\nfigure              {fig}")
    return 0


def _ascii_plot(xs, ys, title, w=62, h=18) -> None:
    """Terminal scatter. Renders anywhere, needs no display, survives being
    run over SSH or on a machine with no GUI toolkit installed."""
    print(f"\n{title}")
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    sx = (w - 1) / (x1 - x0) if x1 > x0 else 0.0
    sy = (h - 1) / (y1 - y0) if y1 > y0 else 0.0
    grid = [[" "] * w for _ in range(h)]
    for i, (x, y) in enumerate(zip(xs, ys)):
        cx = int((x - x0) * sx)
        cy = h - 1 - int((y - y0) * sy)
        ch = "o" if i in (0, len(xs) - 1) else "."
        grid[cy][cx] = ch
    print("+" + "-" * w + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * w + "+")
    print(f" x: {x0:.1f} to {x1:.1f}     y: {y0:.1f} to {y1:.1f}"
          f"     'o' = endpoints")


# ==========================================================================
# demo
# ==========================================================================

def cmd_demo(args) -> int:
    print(rule(72, "="))
    print("  MULTI-CAMERA AIRSPACE MONITORING SYSTEM")
    print("  real-time per-camera detection feeding windowed 3D localisation")
    print(rule(72, "="))
    print("""
  The system assumes a deliberately surveyed camera cluster: known
  positions, known intrinsics, and a shared time base. Each public
  dataset violates a different part of that premise, which is what
  makes them useful as staged tests rather than a single benchmark.

      dataset 1, 2   no synchronisation
      dataset 3      positions given, orientations solved here
      dataset 4      no camera positions
      dataset 5      no 2D labels; detector must supply them
""")
    args.plot = True
    args.cached = True
    args.results = getattr(args, "results", None)
    cmd_calibration(args)
    print()
    cmd_reconstruct(args)
    print(header("TIER 1 -- live"))
    print("Run separately against a camera or stream:")
    print("  fod angular --source 0 --show")
    print("  fod angular --source http://<phone-ip>:8080/video "
          "--calib phone.json --show")
    return 0


# ==========================================================================
# argument parsing
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fod",
        description="Multi-camera airspace monitoring system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  fod demo                                  full walkthrough, cached
  fod angular --source 0 --show             webcam, live
  fod angular --source video.mp4 --calib phone.json
  fod angular --source http://192.168.1.42:8080/video --show
  fod calibrate-camera --images ./board --output phone.json
  fod calibration                           tier 2 results
  fod reconstruct --cached --plot           tier 3 results
""")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("angular", help="tier 1: detection + angular velocity")
    a.add_argument("--source", required=True,
                   help="video path, stream URL, or webcam index")
    a.add_argument("--config", default=None,
                   help="config YAML (default: built-in defaults)")
    a.add_argument("--calib", help="calibration JSON; enables deg/s")
    a.add_argument("--hfov", type=float,
                   help="approximate horizontal FOV in degrees if uncalibrated")
    a.add_argument("--show", action="store_true", help="display video window")
    a.add_argument("--no-clahe", action="store_true",
                   help="disable CLAHE (it costs ~43%% of the frame budget)")
    a.add_argument("--pace-realtime", action="store_true",
                   help="for file sources, sleep to match native fps")
    a.add_argument("--history", type=int, default=None)
    a.add_argument("--var-threshold", type=float, default=None)
    a.add_argument("--min-area", type=float, default=None)
    a.add_argument("--smoothing", type=int, default=5)
    a.add_argument("--print-every", type=int, default=5)
    a.add_argument("--max-frames", type=int, default=0)
    a.set_defaults(func=cmd_angular)

    c = sub.add_parser("calibrate-camera",
                       help="calibrate from checkerboard images")
    c.add_argument("--images", required=True, help="directory of board photos")
    c.add_argument("--output", default="calibration.json")
    c.add_argument("--corners", type=int, nargs=2, default=[9, 6],
                   metavar=("NX", "NY"),
                   help="INNER corner count, not square count (default 9 6)")
    c.add_argument("--square-size", type=float, default=1.0,
                   help="square edge length; units only affect translation")
    c.set_defaults(func=cmd_calibrate_camera)

    g = sub.add_parser("calibration", help="tier 2: camera geometry results")
    g.set_defaults(func=cmd_calibration)

    r = sub.add_parser("reconstruct", help="tier 3: 3D reconstruction results")
    r.add_argument("--cached", action="store_true")
    r.add_argument("--plot", action="store_true", help="ASCII trajectory plots")
    r.add_argument("--results", help="directory holding cached outputs")
    r.set_defaults(func=cmd_reconstruct)

    d = sub.add_parser("demo", help="guided walkthrough for presentation")
    d.add_argument("--results", help="directory holding cached outputs")
    d.set_defaults(func=cmd_demo)

    return p


def main(argv=None) -> int:
    _init_console()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


# The guard is required, not stylistic: Windows spawns child processes by
# re-importing this module, and without it any parallel stage re-runs the CLI
# in every child.
if __name__ == "__main__":
    sys.exit(main())
