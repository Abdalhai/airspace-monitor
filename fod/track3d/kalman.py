"""
kalman.py -- link per-window clusters into 3D tracks and estimate speed.

Constant-velocity Kalman filter, 6-state [x y z vx vy vz]. Speed comes from the
filtered velocity rather than from differencing raw positions, because raw
per-window positions carry metres of calibration error and differencing them
amplifies that into nonsense velocities.

Association is nearest-neighbour with a gate. That is adequate here because
objects are metres apart while position error is metres, not centimetres -- the
voxel grid has already done the hard part of separating simultaneous objects
into distinct clusters, which is what an EKF-only approach could not do.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Track3D:
    track_id: int
    times: List[float] = field(default_factory=list)
    raw: List[np.ndarray] = field(default_factory=list)
    filtered: List[np.ndarray] = field(default_factory=list)
    velocity: List[np.ndarray] = field(default_factory=list)
    misses: int = 0

    @property
    def speeds(self) -> np.ndarray:
        if not self.velocity:
            return np.array([])
        return np.linalg.norm(np.array(self.velocity), axis=1)

    def summary(self) -> str:
        s = self.speeds
        if s.size == 0:
            return f"track {self.track_id}: empty"
        p = np.array(self.filtered)
        dist = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        return (f"track {self.track_id}: {len(self.times)} pts, "
                f"t {self.times[0]:.1f}-{self.times[-1]:.1f}, "
                f"path {dist:.1f} m, speed mean {s.mean():.1f} max {s.max():.1f} m/s")


class ConstantVelocityKF:
    """6-state constant-velocity filter.

    q -- process noise. Raise it when the object manoeuvres hard; the drone in
         these datasets changes direction sharply, so an over-smoothed filter
         will lag through turns.
    r -- measurement noise. Set this from the MEASURED reconstruction error, not
         optimistically: ~4-7 m for dataset 3's ~7 deg orientation solve, and
         well under 1 m once dataset 5's PnP poses are available.
    """

    def __init__(self, dt: float, q: float = 1.0, r: float = 3.0):
        self.dt = dt
        self.F = np.eye(6)
        self.F[:3, 3:] = np.eye(3) * dt
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)
        self.Q = np.eye(6) * q
        self.Q[:3, :3] *= dt ** 2
        self.R = np.eye(3) * r ** 2
        self.x = None
        self.P = None

    def init(self, z: np.ndarray):
        self.x = np.concatenate([z, np.zeros(3)])
        self.P = np.eye(6) * 100.0

    def predict(self, dt: Optional[float] = None):
        if dt is not None and abs(dt - self.dt) > 1e-9:
            self.F[:3, 3:] = np.eye(3) * dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class Tracker3D:
    """Nearest-neighbour association + one Kalman filter per track."""

    def __init__(self, gate: float = 15.0, max_misses: int = 3,
                 dt: float = 1.0, q: float = 1.0, r: float = 3.0):
        self.gate, self.max_misses = gate, max_misses
        self.dt, self.q, self.r = dt, q, r
        self.tracks: List[Track3D] = []
        self._kf = {}
        self._next = 0
        self._last_t = None

    def update(self, clusters, t: float) -> List[Track3D]:
        dt = self.dt if self._last_t is None else max(t - self._last_t, 1e-3)
        self._last_t = t

        live = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        for tr in live:
            self._kf[tr.track_id].predict(dt)

        unmatched = set(range(len(clusters)))
        for tr in live:
            kf = self._kf[tr.track_id]
            pred = kf.x[:3]
            best, bd = None, self.gate
            for i in unmatched:
                d = float(np.linalg.norm(clusters[i].position - pred))
                if d < bd:
                    best, bd = i, d
            if best is None:
                tr.misses += 1
                continue
            c = clusters[best]
            kf.update(c.position)
            tr.times.append(t)
            tr.raw.append(c.position)
            tr.filtered.append(kf.x[:3].copy())
            tr.velocity.append(kf.x[3:].copy())
            tr.misses = 0
            unmatched.discard(best)

        for i in unmatched:
            c = clusters[i]
            tr = Track3D(self._next)
            kf = ConstantVelocityKF(self.dt, self.q, self.r)
            kf.init(c.position)
            tr.times.append(t)
            tr.raw.append(c.position)
            tr.filtered.append(kf.x[:3].copy())
            tr.velocity.append(kf.x[3:].copy())
            self._kf[self._next] = kf
            self.tracks.append(tr)
            self._next += 1

        return [tr for tr in self.tracks if tr.misses == 0]

    def finished(self, min_points: int = 3) -> List[Track3D]:
        return [tr for tr in self.tracks if len(tr.times) >= min_points]
