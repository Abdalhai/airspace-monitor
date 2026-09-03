"""
cluster.py -- reduce an accumulated voxel grid to discrete 3D detections.

One grid covers one time window and contains a bright region wherever rays from
several cameras agreed. This turns those regions into points.

The minimum-camera rule is the important part. Measured on dataset 3: 13.4% of
2-camera reconstructions placed the drone underground (physically impossible --
the cameras sit at Z = -1.8 to 4.1 m and the drone never goes below 0), while
3-, 4-, 5- and 6-camera reconstructions produced zero impossible results. So a
cluster supported by two cameras is noise and a cluster supported by three is a
detection. This costs ~15% of frames and removes the entire population of gross
errors, which is a trade worth making.

Multi-object support is why the voxel grid was chosen over an EKF in the first
place: N simultaneous objects produce N bright regions in one grid with no
data-association step. Dataset 5 has three drones, which is where this matters.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage


@dataclass
class Cluster:
    """A single 3D detection extracted from one time window."""
    position: np.ndarray        # (3,) world, intensity-weighted centroid
    intensity: float            # summed voxel value
    n_voxels: int
    n_cameras: int              # distinct cameras supporting the peak voxel
    extent: np.ndarray          # (3,) bounding-box size in metres
    t_start: float
    t_end: float

    @property
    def t_mid(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


def extract_clusters(acc, t_start: float, t_end: float,
                     percentile: float = 99.9,
                     min_cameras: int = 3,
                     min_voxels: int = 2,
                     max_clusters: int = 5) -> List[Cluster]:
    """Threshold, label connected components, and return them brightest-first.

    percentile   -- how bright a voxel must be to count. 99.9 keeps the top 0.1%.
    min_cameras  -- reject clusters supported by fewer distinct cameras.
    min_voxels   -- reject single-voxel specks (usually one ray pair crossing).
    """
    grid = acc.grid
    nz = grid[grid > 0]
    if nz.size == 0:
        return []

    thresh = np.percentile(nz, percentile)
    if thresh <= 0:
        thresh = np.finfo(np.float32).eps
    mask = grid > thresh
    if not mask.any():
        return []

    lbl, n = ndimage.label(mask)          # 6-connectivity
    if n == 0:
        return []

    cams = acc.camera_counts()
    spec = acc.spec
    out = []
    for i in range(1, n + 1):
        sel = lbl == i
        cnt = int(sel.sum())
        if cnt < min_voxels:
            continue
        zz, yy, xx = np.nonzero(sel)
        vals = grid[zz, yy, xx].astype(np.float64)

        peak = np.argmax(vals)
        n_cam = int(cams[zz[peak], yy[peak], xx[peak]])
        if n_cam < min_cameras:
            continue

        w = vals / vals.sum()
        idx = np.stack([xx, yy, zz], axis=1)
        centroid_idx = (idx * w[:, None]).sum(axis=0)
        pos = spec.index_to_world(centroid_idx)
        extent = (idx.max(axis=0) - idx.min(axis=0) + 1) * spec.voxel_size

        out.append(Cluster(position=pos, intensity=float(vals.sum()),
                           n_voxels=cnt, n_cameras=n_cam, extent=extent,
                           t_start=t_start, t_end=t_end))

    out.sort(key=lambda c: -c.intensity)
    return out[:max_clusters]
