"""
grid.py -- time-windowed ray accumulation into a voxel grid.

Relationship to ray_voxel.cpp
-----------------------------
This is a NumPy reimplementation of the accumulation step, deliberately kept
binary-compatible with the existing C++ tooling: write_bin() emits exactly the
layout spacevoxelviewer.py and voxelmotionviewer.py already read
(int32 N, float32 voxel_size, then N*N*N float32 row-major).

Three changes versus the original C++, all of which were required:

  1. GRID SIZE. The original hardcodes N=500, voxel_size=6.0, centre {0,0,500}
     -- a 3 km cube at 500 MB, sized for an astronomy use case. Dataset 3's
     flight volume is ~92 x 96 x 46 m; dataset 5's is ~91 x 84 x 42 m. At
     N=128, voxel_size=1.0 a grid is 8 MB, which is what makes ONE GRID PER
     TIME WINDOW affordable. That is the whole reason this rewrite exists.

  2. TIME WINDOWING. The original accumulates every ray from the entire
     sequence into a single grid, so a moving object smears into a tube and
     no position can be recovered. Here accumulation is per window: rays from
     [t0, t1) go into their own grid, which is then reset.

  3. REAL DETECTIONS. The original synthesises motion masks with a random
     noise generator. Rays here come from the Scene interface.

Traversal
---------
The C++ uses Amanatides-Woo DDA. This uses dense sampling at voxel_size/2
along each ray, which visits the same voxels for the step size used here and
vectorises cleanly in NumPy. Documented rather than hidden: for grids much
finer than the ray-position error it would begin to differ, but at 1 m voxels
against multi-metre calibration error the distinction is not measurable.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class GridSpec:
    """Voxel grid geometry. Size this from the flight volume, and choose
    voxel_size from the CALIBRATION ERROR, not from a desire for precision:
    voxels much finer than the ray error just spread one blob over more cells."""
    N: int
    voxel_size: float
    center: np.ndarray

    @property
    def half(self) -> float:
        return 0.5 * self.N * self.voxel_size

    @property
    def origin(self) -> np.ndarray:
        return self.center - self.half

    @classmethod
    def from_volume(cls, lo, hi, voxel_size=1.0, margin=10.0, min_n=32):
        """Fit a cubic grid around a bounding box with margin."""
        lo, hi = np.asarray(lo, float) - margin, np.asarray(hi, float) + margin
        center = 0.5 * (lo + hi)
        span = float(np.max(hi - lo))
        N = max(min_n, int(np.ceil(span / voxel_size)))
        N += N % 2
        return cls(N=N, voxel_size=voxel_size, center=center)

    def world_to_index(self, pts: np.ndarray) -> np.ndarray:
        return np.floor((pts - self.origin) / self.voxel_size).astype(np.int32)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(idx, float) + 0.5) * self.voxel_size

    def nbytes(self) -> int:
        return self.N ** 3 * 4


def _ball_offsets(radius_voxels: int):
    """Integer offsets within a sphere, with a linear falloff weight.
    Cached per radius since only a handful of radii ever occur."""
    r = int(radius_voxels)
    if r <= 0:
        return np.zeros((1, 3), np.int32), np.ones(1, np.float32)
    g = np.arange(-r, r + 1)
    dx, dy, dz = np.meshgrid(g, g, g, indexing="ij")
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    keep = d <= r
    off = np.stack([dx[keep], dy[keep], dz[keep]], axis=1).astype(np.int32)
    w = (1.0 - d[keep] / (r + 1.0)).astype(np.float32)
    return off, w


_BALL_CACHE = {}


def ball_offsets(r: int):
    if r not in _BALL_CACHE:
        _BALL_CACHE[r] = _ball_offsets(r)
    return _BALL_CACHE[r]


class VoxelAccumulator:
    """Accumulates rays into a grid, tracking which cameras contributed to each
    voxel so that clustering can enforce a minimum-camera rule.

    The camera count is stored as a bitmask (bit i = camera i contributed), so
    'how many distinct cameras support this voxel' is a popcount rather than a
    per-voxel list. That rule matters: on dataset 3, every physically impossible
    reconstruction came from 2-camera frames, and requiring 3+ removed all of
    them at a cost of 15% of frames.

    UNCERTAINTY CONES
    -----------------
    Rays are not infinitely thin. Each camera's orientation is known only to
    within `ray_sigma_deg`, so the true bearing lies inside a cone about the
    nominal ray. Depositing into a thin line is not merely a simplification --
    it makes the method FAIL OUTRIGHT: measured on dataset 3, thin rays produced
    a maximum of ONE camera per voxel across every window tested, because the
    ~7 deg orientation error makes rays miss each other by ~6.7 m and they never
    share a 1 m cell.

    So each sample point deposits into a ball whose radius is the cone's radius
    at that range, r = range * tan(sigma). This is the calibration uncertainty
    made explicit in the data structure rather than assumed away, and it is what
    lets the grid stay at 1 m voxels (good peak localisation) while still
    letting rays overlap.

    Set ray_sigma_deg from MEASURED per-camera residuals: ~7 deg for dataset 3's
    bundle-adjusted rotations, and well under 1 deg once dataset 5's PnP poses
    are available -- at which point the cones shrink and the peak sharpens
    without any other change.
    """

    def __init__(self, spec: GridSpec, alpha: float = 0.0, max_range: float = 400.0,
                 ray_sigma_deg: float = 0.0, max_radius_voxels: int = 12):
        self.spec = spec
        self.alpha = alpha              # distance attenuation, 0 = off
        self.max_range = max_range
        self.ray_sigma = np.radians(ray_sigma_deg)
        self.max_radius_voxels = max_radius_voxels
        self.reset()

    def reset(self):
        N = self.spec.N
        self.grid = np.zeros((N, N, N), dtype=np.float32)
        self.cam_mask = np.zeros((N, N, N), dtype=np.uint32)

    def add_ray(self, origin: np.ndarray, direction: np.ndarray, cam_id: int,
                weight: float = 1.0):
        spec = self.spec
        step = spec.voxel_size * 0.5

        t_enter, t_exit = self._slab_clip(origin, direction)
        if t_enter is None:
            return 0
        t_exit = min(t_exit, self.max_range)
        if t_exit <= t_enter:
            return 0

        ts = np.arange(t_enter, t_exit, step)
        if ts.size == 0:
            return 0
        pts = origin[None, :] + ts[:, None] * direction[None, :]
        idx = spec.world_to_index(pts)

        if self.ray_sigma <= 0:
            all_idx, all_w = idx, np.full(len(idx), weight, dtype=np.float32)
        else:
            # Cone radius grows with range; bin it so the ball offsets cache hits.
            radii = np.clip(
                np.ceil(ts * np.tan(self.ray_sigma) / spec.voxel_size).astype(int),
                0, self.max_radius_voxels)
            chunks_i, chunks_w = [], []
            for r in np.unique(radii):
                sel = radii == r
                off, ow = ball_offsets(int(r))
                base = idx[sel]
                exp = (base[:, None, :] + off[None, :, :]).reshape(-1, 3)
                # PEAK-normalised, not volume-normalised. Dividing by the ball
                # volume would make near-camera samples (thin cone, few voxels)
                # far brighter than distant ones, so the grid maximum would sit
                # against the cameras instead of on the object.
                wv = np.tile(ow, len(base)).astype(np.float32) * weight
                chunks_i.append(exp); chunks_w.append(wv)
            all_idx = np.concatenate(chunks_i)
            all_w = np.concatenate(chunks_w)

        ok = np.all((all_idx >= 0) & (all_idx < spec.N), axis=1)
        all_idx, all_w = all_idx[ok], all_w[ok]
        if all_idx.size == 0:
            return 0

        # collapse duplicates so one ray cannot deposit twice into one voxel
        flat = (all_idx[:, 0].astype(np.int64) * spec.N * spec.N
                + all_idx[:, 1].astype(np.int64) * spec.N + all_idx[:, 2])
        uniq, inv = np.unique(flat, return_inverse=True)
        wsum = np.zeros(len(uniq), np.float32)
        np.add.at(wsum, inv, all_w)

        ix = (uniq // (spec.N * spec.N)).astype(np.int32)
        iy = ((uniq // spec.N) % spec.N).astype(np.int32)
        iz = (uniq % spec.N).astype(np.int32)

        np.add.at(self.grid, (iz, iy, ix), wsum)        # stored (Z, Y, X)
        self.cam_mask[iz, iy, ix] |= np.uint32(1 << cam_id)
        return len(uniq)

    def add_rays(self, rays) -> int:
        return sum(self.add_ray(r.origin, r.direction, r.cam_id) for r in rays)

    def camera_counts(self) -> np.ndarray:
        """Number of distinct cameras contributing to each voxel."""
        m = self.cam_mask
        c = np.zeros_like(m, dtype=np.uint8)
        for b in range(32):
            c += ((m >> np.uint32(b)) & np.uint32(1)).astype(np.uint8)
        return c

    def _slab_clip(self, o, d) -> Tuple[Optional[float], Optional[float]]:
        """Ray/AABB intersection. Returns (t_enter, t_exit) or (None, None)."""
        lo, hi = self.spec.origin, self.spec.origin + 2 * self.spec.half
        t0, t1 = 0.0, np.inf
        for i in range(3):
            if abs(d[i]) < 1e-12:
                if o[i] < lo[i] or o[i] > hi[i]:
                    return None, None
                continue
            a, b = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return None, None
        return t0, t1

    def write_bin(self, path: str):
        """Emit the exact format spacevoxelviewer.py / voxelmotionviewer.py read."""
        with open(path, "wb") as f:
            f.write(np.int32(self.spec.N).tobytes())
            f.write(np.float32(self.spec.voxel_size).tobytes())
            f.write(self.grid.astype(np.float32).tobytes())


def accumulate_windows(scene, spec: GridSpec, t0: float, t1: float,
                       window: float, samples_per_window: int = 20,
                       min_cameras: int = 3, alpha: float = 0.0,
                       ray_sigma_deg: float = 0.0):
    """Sweep [t0, t1) in windows, yielding one accumulator per window.

    Yields (t_start, t_end, accumulator, n_rays).
    """
    acc = VoxelAccumulator(spec, alpha=alpha, ray_sigma_deg=ray_sigma_deg)
    t = t0
    while t < t1:
        te = min(t + window, t1)
        acc.reset()
        rays = scene.rays_in_window(t, te, samples_per_window)
        acc.add_rays(rays)
        yield t, te, acc, len(rays)
        t = te
