"""
scene.py -- the three interfaces every downstream stage consumes.

Downstream code (voxel accumulation, clustering, 3D tracking) never learns which
dataset it is running on. It asks a Scene for:

    scene.cameras          -> list[Camera]      (center, R, K, dist  -- all resolved)
    scene.rays(t)          -> list[(cam_id, origin, direction)]  in WORLD frame
    scene.time_range()     -> (t_min, t_max)    in the scene's own time units

Three concrete Scenes exist:

    Dataset3Scene   -- fully working. Positions from campos.txt, rotations from
                       the Phase 2 solve, detections from cam*.txt, time axis is
                       cam0's frame index (via the README alpha/beta sync table).

    Dataset5Scene   -- structurally complete, BLOCKED on two missing inputs
                       (see docstring). Time axis is project seconds, which is
                       simpler than dataset 3: cam*_frame_ts.txt is already in
                       project time, so no sync coefficients are applied.

    LiveScene       -- method definitions only, intentionally not implemented.
                       Documents the runtime-configurable live path so the shape
                       of the extension is visible in the code rather than in a
                       promise. See fod/sources/live.py.

The split exists because these three differ ONLY in how (camera pose, detection,
timestamp) triples are obtained. Everything after this file is shared.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
import numpy as np
import cv2
from scipy.interpolate import interp1d


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

@dataclass
class Camera:
    """A camera with fully resolved pose and intrinsics.

    R is WORLD -> CAMERA, matching the Phase 2 convention:
        Xc = R @ (X_world - center)
    so a camera-frame ray is taken to world frame by  R.T @ ray_cam.
    """
    cam_id: int
    model: str
    center: np.ndarray          # (3,) world
    R: np.ndarray               # (3,3) world -> camera
    K: np.ndarray               # (3,3) at NATIVE resolution
    dist: np.ndarray            # (5,) or (4,)
    fps: float
    resolution: Optional[Tuple[int, int]] = None

    def pixels_to_world_rays(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) native pixel coords -> (N,3) unit direction vectors in WORLD frame.

        Native resolution is required: K is defined at the sensor's own
        resolution, so any resize applied for detection must be undone before
        this call. See fod/sources/README_native_coords.md.
        """
        pts = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)
        rays = np.concatenate([und, np.ones((len(und), 1))], axis=1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays @ self.R          # (R.T @ r).T  ==  r @ R


@dataclass
class Ray:
    cam_id: int
    origin: np.ndarray
    direction: np.ndarray


# ----------------------------------------------------------------------------
# Base scene
# ----------------------------------------------------------------------------

class Scene:
    """Base class. Subclasses populate .cameras and implement ._detection_at()."""

    cameras: List[Camera]

    def time_range(self) -> Tuple[float, float]:
        raise NotImplementedError

    def time_unit(self) -> str:
        """Human-readable unit for the time axis ('cam0 frames' or 'seconds')."""
        raise NotImplementedError

    def _detection_at(self, cam_id: int, t: float) -> Optional[np.ndarray]:
        """Native-resolution (x, y) for this camera at scene-time t, or None."""
        raise NotImplementedError

    def rays(self, t: float) -> List[Ray]:
        """All available world-frame rays at scene-time t."""
        out = []
        for cam in self.cameras:
            xy = self._detection_at(cam.cam_id, t)
            if xy is None:
                continue
            d = cam.pixels_to_world_rays(xy[None, :])[0]
            out.append(Ray(cam.cam_id, cam.center, d))
        return out

    def rays_in_window(self, t0: float, t1: float, n_samples: int) -> List[Ray]:
        """All rays from all cameras across [t0, t1). This is what the voxel
        accumulator consumes -- a time window, not a single instant."""
        out = []
        for t in np.linspace(t0, t1, n_samples, endpoint=False):
            out.extend(self.rays(t))
        return out


# ----------------------------------------------------------------------------
# Dataset 3
# ----------------------------------------------------------------------------

# Ground-truth sync table, dataset 3 README.
# frame_in_cam_j = ALPHA[i][j] * frame_in_cam_i + BETA[i][j]
_DS3_ALPHA = np.array([
    [1.0000, 0.5005, 0.4960, 0.4171, 0.5000, 0.8341],
    [1.9982, 1.0000, 0.9910, 0.8333, 0.9990, 1.6667],
    [2.0163, 1.0091, 1.0000, 0.8409, 1.0081, 1.6819],
    [2.3978, 1.2000, 1.1892, 1.0000, 1.1988, 2.0000],
    [2.0001, 1.0010, 0.9919, 0.8342, 1.0000, 1.6683],
    [1.1989, 0.6000, 0.5946, 0.5000, 0.5994, 1.0000]])
_DS3_BETA = np.array([
    [0.00, 1013.95, 546.98, 251.16, 961.02, 137.51],
    [-2026.04, 0.00, -457.83, -593.82, -51.96, -1552.47],
    [-1102.90, 461.99, 0.00, -208.81, 409.59, -782.45],
    [-602.21, 712.57, 248.32, 0.00, 659.93, -364.81],
    [-1922.12, 52.01, -406.29, -551.00, 0.00, -1465.78],
    [-164.85, 931.45, 465.22, 182.40, 878.60, 0.00]])

_DS3_MODELS = ["gopro3", "mate7", "mate10", "sony5n_1440x1080", "sony5100", "sonyG"]


def _read_detection_txt(path: str):
    """Shared reader for the '<frame> <x> <y>' label format used by datasets 3-5.

    Two documented quirks, both of which silently corrupt geometry if ignored:
      - a non-numeric header row
      - '0.0 0.0' meaning 'not visible this frame', NOT a pixel position
    """
    fr, xs, ys = [], [], []
    n_zero = n_bad = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 3:
                n_bad += 1
                continue
            try:
                a, b, c = float(p[0]), float(p[1]), float(p[2])
            except ValueError:
                n_bad += 1
                continue
            if b == 0.0 and c == 0.0:
                n_zero += 1
                continue
            fr.append(a); xs.append(b); ys.append(c)
    return (np.array(fr), np.array(xs), np.array(ys), n_zero, n_bad)


class _InterpolatedTrack:
    """Per-camera 2D detection track with gap-aware interpolation.

    Interpolating across a long gap (drone left the frame) would fabricate
    detections along a straight line that the drone never flew, so gaps wider
    than `gap_thresh` return None instead.
    """

    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, gap_thresh: float):
        o = np.argsort(t)
        self.t, self.x, self.y = t[o], x[o], y[o]
        self.gap_thresh = gap_thresh
        self._ix = interp1d(self.t, self.x, bounds_error=False)
        self._iy = interp1d(self.t, self.y, bounds_error=False)

    def at(self, t: float) -> Optional[np.ndarray]:
        if t < self.t[0] or t > self.t[-1]:
            return None
        j = np.searchsorted(self.t, t)
        if j == 0 or j >= len(self.t):
            return None
        if self.t[j] - self.t[j - 1] > self.gap_thresh:
            return None
        vx, vy = float(self._ix(t)), float(self._iy(t))
        if np.isnan(vx) or np.isnan(vy):
            return None
        return np.array([vx, vy])


class Dataset3Scene(Scene):
    """Dataset 3. Time axis = cam0 frame index (59.94 fps).

    Camera positions are survey-grade ground truth (<5 cm). Rotations come from
    the Phase 2 cross-camera solve and carry ~7 deg of error, which is the
    dominant error source in anything built on this scene.
    """

    FPS_REF = 59.94006

    def __init__(self, det_dir: str, poses_json: str, gap_thresh: float = 30.0):
        d = json.load(open(poses_json))
        self.cameras, self._tracks = [], {}
        for c in d["cameras"]:
            i = c["cam_id"]
            self.cameras.append(Camera(
                cam_id=i, model=c["model"],
                center=np.array(c["center"]), R=np.array(c["R"]),
                K=np.array(c["K"]), dist=np.array(c["distCoeff"]), fps=c["fps"]))
            fr, x, y, _, _ = _read_detection_txt(f"{det_dir}/cam{i}_dataset3.txt")
            tg = fr.astype(float) if i == 0 else _DS3_ALPHA[i, 0] * fr + _DS3_BETA[i, 0]
            self._tracks[i] = _InterpolatedTrack(tg, x, y, gap_thresh)

    def time_range(self):
        lo = max(tr.t[0] for tr in self._tracks.values())
        hi = min(tr.t[-1] for tr in self._tracks.values())
        return (lo, hi)

    def time_unit(self):
        return "cam0 frames @ 59.94 fps"

    def seconds_per_time_unit(self):
        return 1.0 / self.FPS_REF

    def _detection_at(self, cam_id, t):
        return self._tracks[cam_id].at(t)


# ----------------------------------------------------------------------------
# Dataset 5
# ----------------------------------------------------------------------------

_DS5_MODELS = ["sonya5100", "huaweiP40pro", "gopro7", "sonyG", "samsungS10", "sonynex5n"]


class Dataset5Scene(Scene):
    """Dataset 5. Time axis = PROJECT SECONDS.

    Simpler than dataset 3 in two ways worth stating explicitly:

      1. cam*_frame_ts.txt is ALREADY in project time. The coefficients in
         sync_coefficients_cam2pc.txt have been applied by the dataset authors.
         Applying them again double-counts the shift. Verified: cam0 frame 1
         has ts 10.5846 while cam0's shift is 10.4863.

      2. campos.txt and fused_pose.txt share the project local frame, so drone
         ground truth is directly comparable to reconstructions. This is what
         dataset 3 lacks and it is the reason dataset 5 is the primary dataset.

    BLOCKED until two inputs arrive:
      - Calibration/ : gopro7, huaweiP40pro and samsungS10 have no calibration
        in the global set (gopro3 != gopro7, p20pro != P40pro, no S10 at all).
      - Detections/  : Drone0..2 / cam0..5.txt 2D labels.

    Everything else (positions, timestamps, ground truth) is present and parsed
    below, so this class becomes live the moment those two folders land.
    """

    def __init__(self, root: str, calib_dir: str, drone: str = "Drone0",
                 gap_thresh: float = 0.5, rotations_json: Optional[str] = None):
        self.root = root
        self.centers = self._load_campos(f"{root}/campos_dataset5.txt")
        self.frame_ts = {i: self._load_frame_ts(f"{root}/cam{i}_frame_ts.txt")
                         for i in range(6)}
        self.gt = load_fused_pose(f"{root}/fused_pose.txt")

        rots = json.load(open(rotations_json)) if rotations_json else {}
        self.cameras, self._tracks = [], {}
        for i, model in enumerate(_DS5_MODELS):
            K, dist, fps, res = _load_calib(f"{calib_dir}/{model}.json")
            R = np.array(rots[str(i)]) if str(i) in rots else np.eye(3)
            self.cameras.append(Camera(i, model, self.centers[i], R, K, dist, fps, res))
            fr, x, y, _, _ = _read_detection_txt(f"{root}/{drone}/cam{i}.txt")
            ts = self._frames_to_seconds(i, fr)
            self._tracks[i] = _InterpolatedTrack(ts, x, y, gap_thresh)

    @staticmethod
    def _load_campos(path):
        out = []
        for line in open(path):
            p = line.split()
            if len(p) == 4 and p[0].lower().startswith("cam"):
                out.append([float(v) for v in p[1:4]])
        return np.array(out)

    @staticmethod
    def _load_frame_ts(path):
        """-> dict frame_id -> project-time seconds (already synced)."""
        d = np.loadtxt(path, skiprows=1)
        return dict(zip(d[:, 0].astype(int), d[:, 1]))

    def _frames_to_seconds(self, cam_id, frames):
        m = self.frame_ts[cam_id]
        return np.array([m.get(int(f), np.nan) for f in frames])

    def time_range(self):
        return (float(self.gt["t"].min()), float(self.gt["t"].max()))

    def time_unit(self):
        return "project seconds"

    def seconds_per_time_unit(self):
        return 1.0

    def _detection_at(self, cam_id, t):
        return self._tracks[cam_id].at(t)

    def ground_truth_at(self, t: float) -> Optional[np.ndarray]:
        """Interpolated GT drone position at project time t. Dataset 5 only --
        this is what makes quantitative error reporting possible."""
        g = self.gt
        if t < g["t"][0] or t > g["t"][-1]:
            return None
        return np.array([np.interp(t, g["t"], g["X"]),
                         np.interp(t, g["t"], g["Y"]),
                         np.interp(t, g["t"], g["Z"])])


def _load_calib(path):
    d = json.load(open(path))
    return (np.array(d["K-matrix"]), np.array(d["distCoeff"]),
            d.get("fps"), tuple(d.get("resolution", ())) or None)


def load_fused_pose(path: str) -> Dict[str, np.ndarray]:
    """Dataset 5 ground truth. ~9 Hz, mm-level std, 187.4 s of the ~10 min flight.

    Note: the shipped fused_pose.csv is an XLSX file with a .csv extension.
    This reads the .txt, which has identical content.
    """
    rows = []
    with open(path, errors="replace") as f:
        next(f)
        for line in f:
            p = line.replace("\r", "").split()
            if len(p) >= 11:
                rows.append([float(v) for v in p[:11]])
    a = np.array(rows)
    return {"t": a[:, 0], "X": a[:, 1], "Y": a[:, 2], "Z": a[:, 3],
            "roll": a[:, 4], "pitch": a[:, 5], "yaw": a[:, 6],
            "std": a[:, 7:10], "status": a[:, 10]}
