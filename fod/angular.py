"""
Angular velocity estimation from a single camera.

WHY THIS MODULE EXISTS
----------------------
A single camera cannot measure metric speed. An object at range 2d moving at
speed 2v produces exactly the same image motion as an object at range d moving
at speed v -- the observation is rank-deficient in range, and no amount of
image processing recovers it. This is the limitation that motivates the whole
multi-camera stage of the project.

What a single calibrated camera CAN measure is the *angular* velocity: the rate
at which the bearing to the object sweeps across the observer, in degrees per
second. That is a real, physically meaningful quantity, it is defensible under
questioning, and combined with a range estimate from the multi-camera stage it
yields metric speed:

    v_tangential = omega_rad_per_s * range_m

So this module is not a weaker substitute for speed. It is the half of the
problem a single camera is actually able to solve, stated honestly.

CALIBRATION IS REQUIRED FOR DEGREES
-----------------------------------
Converting pixel displacement to an angle requires the focal length in pixels.
Without it, this module reports px/s and says so, rather than silently emitting
a number in degrees that is off by whatever the true focal length happens to be.

Calibrate a phone in ~20 minutes: print a checkerboard, take 15-20 photos of it
at varied angles filling different parts of the frame, run cv2.calibrateCamera.
Save the result with save_calibration() and pass it via --calib.

IMPORTANT: the focal length is in PIXELS and therefore depends on the capture
resolution. A calibration taken at 1920x1080 does not apply to a 1280x720
stream. scale_calibration() handles the rescale; use it rather than reusing a
mismatched K, which is a silent error that produces plausible wrong numbers.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Pinhole intrinsics for one camera at one specific resolution."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: np.ndarray | None = None

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view. Useful as a sanity check on a calibration:
        a phone main camera is typically 60-70 deg, an action cam 90-120 deg.
        A number far outside that range means the calibration is wrong."""
        return 2.0 * math.degrees(math.atan(self.width / (2.0 * self.fx)))

    def scaled_to(self, width: int, height: int) -> "Calibration":
        """Rescale intrinsics to a different capture resolution.

        Valid only for a pure resize. If the camera changes aspect ratio by
        cropping or anamorphic squeeze rather than scaling, this is wrong --
        dataset 3's sony5n is a real example of an anamorphic sensor where
        fx and fy differ by 33% for exactly that reason.
        """
        sx = width / self.width
        sy = height / self.height
        return Calibration(self.fx * sx, self.fy * sy,
                           self.cx * sx, self.cy * sy,
                           width, height, self.dist)

    def to_dict(self) -> dict:
        d = {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
             "width": self.width, "height": self.height}
        if self.dist is not None:
            d["dist"] = np.asarray(self.dist).ravel().tolist()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        dist = np.array(d["dist"], dtype=np.float64) if "dist" in d else None
        return Calibration(float(d["fx"]), float(d["fy"]),
                           float(d["cx"]), float(d["cy"]),
                           int(d["width"]), int(d["height"]), dist)

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        with open(path, "r", encoding="utf-8") as fh:
            return Calibration.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_hfov(width: int, height: int, hfov_deg: float) -> "Calibration":
        """Approximate intrinsics from a nominal field of view.

        A FALLBACK, NOT A CALIBRATION. Manufacturer FOV figures are marketing
        numbers, often quoted diagonally, and ignore the crop the video mode
        applies. Expect 5-15% focal length error, which propagates directly
        into the reported angular rate. Anything presented as a measurement
        should use a real checkerboard calibration instead.
        """
        f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return Calibration(f, f, width / 2.0, height / 2.0, width, height)


# ---------------------------------------------------------------------------
# angular velocity
# ---------------------------------------------------------------------------

def bearing_vector(px: float, py: float, calib: Calibration) -> np.ndarray:
    """Unit vector in camera coordinates pointing at pixel (px, py)."""
    x = (px - calib.cx) / calib.fx
    y = (py - calib.cy) / calib.fy
    v = np.array([x, y, 1.0], dtype=np.float64)
    return v / np.linalg.norm(v)


def angular_separation_deg(p0, p1, calib: Calibration) -> float:
    """Angle between the bearings to two pixels, in degrees.

    Uses the full bearing-vector angle rather than the small-angle shortcut
    (d_px / fx). Off-axis, near a frame edge, or on a wide-angle lens the
    shortcut overestimates -- at the corner of a 90 deg lens by roughly 30%.
    The exact form costs nothing here.
    """
    v0 = bearing_vector(p0[0], p0[1], calib)
    v1 = bearing_vector(p1[0], p1[1], calib)
    return math.degrees(math.acos(float(np.clip(v0 @ v1, -1.0, 1.0))))


@dataclass
class AngularTrack:
    """Per-track history and smoothed angular rate."""

    track_id: int
    smoothing: int = 5
    history: deque = field(default_factory=lambda: deque(maxlen=64))
    _rates: deque = field(default_factory=lambda: deque(maxlen=16))

    def update(self, t: float, centroid, calib: Calibration | None):
        """Add an observation. Returns (rate, unit) or (None, unit) if the
        track is too short to have a rate yet."""
        self.history.append((float(t), float(centroid[0]), float(centroid[1])))
        unit = "deg/s" if calib is not None else "px/s"
        if len(self.history) < 2:
            return None, unit

        # Compare against a sample `smoothing` steps back rather than the
        # immediately previous frame. Frame-to-frame centroid jitter on a small
        # blob is a pixel or two, which at 30fps is tens of deg/s of pure noise;
        # a longer baseline averages it down without adding real lag at the
        # timescales these objects move on.
        k = min(self.smoothing, len(self.history) - 1)
        t0, x0, y0 = self.history[-1 - k]
        t1, x1, y1 = self.history[-1]
        dt = t1 - t0
        if dt <= 0:
            return None, unit

        if calib is not None:
            delta = angular_separation_deg((x0, y0), (x1, y1), calib)
        else:
            delta = math.hypot(x1 - x0, y1 - y0)

        self._rates.append(delta / dt)
        return float(np.median(self._rates)), unit

    @property
    def bearing_deg(self) -> float | None:
        """Direction of travel in the image plane, degrees, 0 = right,
        increasing counter-clockwise."""
        if len(self.history) < 2:
            return None
        k = min(self.smoothing, len(self.history) - 1)
        _, x0, y0 = self.history[-1 - k]
        _, x1, y1 = self.history[-1]
        if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9:
            return None
        # negate dy because image y grows downward
        return math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 360.0

    @property
    def compass(self) -> str:
        b = self.bearing_deg
        if b is None:
            return "--"
        names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        return names[int((b + 22.5) % 360.0 // 45.0)]


class AngularVelocityEstimator:
    """Maintains an AngularTrack per 2D track id.

    Deliberately holds no opinion about how tracks are produced -- it consumes
    (id, centroid) pairs, so it works with the Phase 1 tracker as it stands and
    with whatever replaces it.
    """

    def __init__(self, calib: Calibration | None = None, smoothing: int = 5):
        self.calib = calib
        self.smoothing = smoothing
        self.tracks: dict[int, AngularTrack] = {}

    @property
    def unit(self) -> str:
        return "deg/s" if self.calib is not None else "px/s"

    def update(self, t: float, tracks: dict) -> dict:
        """tracks: {track_id: (cx, cy)}. Returns {track_id: dict of readings}."""
        out = {}
        for tid, centroid in tracks.items():
            if tid not in self.tracks:
                self.tracks[tid] = AngularTrack(tid, self.smoothing)
            at = self.tracks[tid]
            rate, unit = at.update(t, centroid, self.calib)
            out[tid] = {
                "id": tid,
                "centroid": (float(centroid[0]), float(centroid[1])),
                "rate": rate,
                "unit": unit,
                "bearing_deg": at.bearing_deg,
                "compass": at.compass,
                "n_obs": len(at.history),
            }
        # drop tracks that disappeared, so ids don't accumulate over a long run
        for tid in list(self.tracks):
            if tid not in tracks:
                del self.tracks[tid]
        return out

    def format_line(self, reading: dict) -> str:
        r = reading
        rate = "  --  " if r["rate"] is None else f"{r['rate']:6.1f}"
        brg = "---" if r["bearing_deg"] is None else f"{r['bearing_deg']:5.1f}"
        return (f"id={r['id']:<3} ({r['centroid'][0]:7.1f},{r['centroid'][1]:7.1f})  "
                f"{rate} {r['unit']:<6}  bearing {brg} {r['compass']:<2}  n={r['n_obs']}")


def tangential_speed_mps(rate_deg_s: float, range_m: float) -> float:
    """Convert angular rate to metric tangential speed given a range.

    This is the bridge between the single-camera and multi-camera stages: the
    camera supplies the angle, the multi-camera reconstruction supplies the
    range, and only together do they give a speed. Note this is the TANGENTIAL
    component only -- motion directly toward or away from the camera produces
    no angular rate at all and is invisible to this measurement.
    """
    return math.radians(rate_deg_s) * range_m
