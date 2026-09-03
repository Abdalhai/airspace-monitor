"""
tracking.py — blob extraction from a foreground mask + frame-to-frame centroid
tracking, used to compute a simple direction vector per object.

Scope note (explicit limitation, not an oversight): this is the simplest possible
tracker — nearest-centroid matching within a max-distance gate. It is NOT designed
to resolve ambiguous pairing when multiple objects cross paths or move in similar
directions close together. That's exactly one of the failure cases the top-level
prompt calls out as belonging to the later voxel-based extension. For Phase 1
(single or a few well-separated objects) this is sufficient and easy to verify
by eye against the overlay.

FIXES APPLIED
-------------
defect 1  `direction_lookback` was configurable but never reached this module --
          every direction used the hardcoded default of 5. CentroidTracker now
          carries it and Track.direction() defaults to the configured value.

defect 4  No track confirmation: a single noise blob immediately became a track
          with an id. Tracks now report `confirmed` only after `min_hits`
          observations. Unconfirmed tracks still update, they are just not
          reported -- which keeps ids stable while suppressing one-frame noise.

defect 5  Greedy nearest-centroid matching resolved in arbitrary dict order, so
          whichever track happened to be iterated first claimed the nearest blob
          even when that assignment was globally worse. With three drones in
          dataset 5 this is exactly the documented failure regime. Replaced with
          optimal assignment (Hungarian, scipy.optimize.linear_sum_assignment),
          which minimises TOTAL matching cost. Falls back to the original greedy
          behaviour if scipy is unavailable, so the module keeps working with
          only opencv and numpy installed.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except ImportError:      # keep working on a numpy+opencv-only install
    _HAVE_SCIPY = False


@dataclass
class Blob:
    centroid: tuple    # (x, y) in pixels
    bbox: tuple         # (x, y, w, h)
    area: float


@dataclass
class Track:
    track_id: int
    history: List[tuple] = field(default_factory=list)  # (x, y) centroids, oldest first
    last_seen_frame: int = 0
    missed_frames: int = 0
    hits: int = 1                 # total observations matched to this track
    min_hits: int = 3             # observations required before it is reported
    default_lookback: int = 5     # set by CentroidTracker from config

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def direction(self, lookback: Optional[int] = None) -> Optional[tuple]:
        """(angle_degrees, compass_label) from displacement over the last `lookback`
        centroids, or None if there isn't enough history yet to be meaningful."""
        if lookback is None:
            lookback = self.default_lookback
        pts = self.history[-lookback:]
        if len(pts) < 2:
            return None
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        # Screen y grows downward; flip sign so "up" reads as up.
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        return angle, _angle_to_compass(angle)


def _angle_to_compass(angle: float) -> str:
    dirs = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    idx = int(((angle + 22.5) % 360) // 45)
    return dirs[idx]


def extract_blobs(mask: np.ndarray, min_area: int = 25) -> List[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        blobs.append(Blob(centroid=(cx, cy), bbox=(x, y, w, h), area=area))
    return blobs


class CentroidTracker:
    def __init__(self, max_match_distance: float = 60.0, max_missed_frames: int = 10,
                 history_len: int = 15, direction_lookback: int = 5,
                 min_hits: int = 3):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.history_len = history_len
        self.direction_lookback = direction_lookback   # defect 1: now actually used
        self.min_hits = min_hits                       # defect 4
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

    def update(self, blobs: List[Blob], frame_index: int) -> List[Track]:
        matched_blobs = self._assign(blobs)

        for tid, track in self._tracks.items():
            bi = matched_blobs.get(tid)
            if bi is None:
                track.missed_frames += 1
                continue
            track.history.append(blobs[bi].centroid)
            track.history = track.history[-self.history_len:]
            track.last_seen_frame = frame_index
            track.missed_frames = 0
            track.hits += 1

        used = set(matched_blobs.values())
        for i in range(len(blobs)):
            if i in used:
                continue
            self._tracks[self._next_id] = Track(
                track_id=self._next_id, history=[blobs[i].centroid],
                last_seen_frame=frame_index, hits=1, min_hits=self.min_hits,
                default_lookback=self.direction_lookback)
            self._next_id += 1

        dead = [tid for tid, t in self._tracks.items() if t.missed_frames > self.max_missed_frames]
        for tid in dead:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _assign(self, blobs: List[Blob]) -> Dict[int, int]:
        """Return {track_id: blob_index} for matched pairs only.

        Optimal assignment rather than greedy: greedy resolves in iteration order,
        so an early track can claim a blob that a later track needed more, and the
        total cost is worse. With well-separated single objects the two agree; with
        several objects near each other they do not, and that is the case that
        matters for multi-drone footage."""
        tids = [tid for tid, t in self._tracks.items() if t.history]
        if not tids or not blobs:
            return {}

        cost = np.full((len(tids), len(blobs)), np.inf)
        for r, tid in enumerate(tids):
            lx, ly = self._tracks[tid].history[-1]
            for c, b in enumerate(blobs):
                d = math.hypot(b.centroid[0] - lx, b.centroid[1] - ly)
                if d <= self.max_match_distance:
                    cost[r, c] = d

        out: Dict[int, int] = {}
        if _HAVE_SCIPY:
            # linear_sum_assignment cannot handle inf, so substitute a value that
            # exceeds any admissible cost; pairs landing on it are rejected after.
            big = self.max_match_distance * 10.0
            finite = np.where(np.isinf(cost), big, cost)
            rows, cols = linear_sum_assignment(finite)
            for r, c in zip(rows, cols):
                if np.isfinite(cost[r, c]):
                    out[tids[r]] = int(c)
        else:
            taken = set()
            order = sorted(((cost[r, c], r, c)
                            for r in range(len(tids)) for c in range(len(blobs))
                            if np.isfinite(cost[r, c])))
            for _, r, c in order:
                if tids[r] in out or c in taken:
                    continue
                out[tids[r]] = c
                taken.add(c)
        return out
