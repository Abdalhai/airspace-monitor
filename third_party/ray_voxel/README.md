# Inherited C++ voxel accumulator — reference only

Pristine copy of the open-source `ray_voxel.cpp` and companions the project
started from. **Not a runtime dependency.** The shipped pipeline is pure Python;
nothing here is compiled or called. It is kept for provenance and for diffing
against the reimplementation.

Place the original sources here unmodified: `ray_voxel.cpp`, `process_image.cpp`,
`setup.py`, `spacevoxelviewer.py`, `voxelmotionviewer.py`.

## Three changes the Python version makes, and why each was forced

**1. Grid sizing.** The original allocates N=500 at 6.0 m voxels over a 3 km
cube: 500 MB. The reimplementation uses N=68 at 1.5 m — 1.3 MB per window, 385×
smaller. That reduction is what makes one grid per time window affordable at all.

**2. Time windowing.** The original accumulates an entire sequence into a single
grid, which smears a moving object into a tube along its own path. Windowing at
0.5 s keeps each grid a snapshot.

**3. Real detections.** The original reads from a synthetic noise image source.

## The failure that shaped the design

With infinitely thin rays the maximum number of cameras contributing to any
single voxel was **1**, across every window and every window length tested. Zero
clusters, always.

The cause is geometric, not a bug: 7° of orientation error puts rays roughly
6.7 m apart at working range, so they never share a 1.5 m cell. The fix is
**uncertainty-cone splatting**, with the cone half-angle set to the *measured*
calibration error rather than a tuned constant.

| σ | max cameras/voxel | clusters |
|---|---|---|
| 0 (thin) | 1 | 0 |
| 3° | 5 | 0–1 |
| 5° | 6 | 1 |
| 7° (measured) | 6 | 1 |

Because σ is the measured error, cones shrink automatically when better
calibration arrives. See `fod/voxel/grid.py` for the peak-normalisation bug this
introduced and how it was caught.
