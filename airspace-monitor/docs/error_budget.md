# Error budget

Every measured number in the project, with how it was obtained. This is the
document that distinguishes measured work from work that merely runs. Nothing
here is estimated; each figure traces to a script that produced it.

---

## 1. Camera orientation error (Tier 2)

Dataset 3 supplies camera positions at true metric scale in `campos.txt`, so only
the 18 rotational degrees of freedom needed solving — no scale or gauge ambiguity.

**Validated out of sample** on 553 frames the optimiser never saw, using raw
detections, the published synchronisation table, and the solved rotations.

| Statistic | Value |
|---|---|
| Mean | **7.04°** |
| Median | 6.53° |
| 90th percentile | 11.6° |
| 95th percentile | 14.3° |
| Mean ray miss distance | 6.72 m |

| Camera | Error |
|---|---|
| cam5 | 4.94° |
| cam2 | 5.36° |
| cam0 | 6.39° |
| cam4 | 6.58° |
| cam1 | 7.91° |
| cam3 | **11.27°** (worst) |

### The correction that matters

An earlier figure of **4.5°** appeared in `solved_camera_poses.json`. It was the
in-sample joint-refinement residual — measured on the same 143 points the
optimiser was free to move — and it understated the true error by 36%.

This is the standard trap of evaluating on training data. The 7.04° figure is
measured on held-out frames and is what every downstream accuracy claim uses.
Reporting the smaller number would have been indefensible under questioning.

### Diverged points

Indices 11 and 50 in `sample_3d_points` diverged to **1.65 × 10⁶ m** and
**2.60 × 10⁶ m**. The joint refinement has no depth bound, so a point observed at
a shallow intersection angle can be pushed toward infinity without the cost
function objecting.

Both are culled in `results/solved_camera_poses.json` (141 of 143 retained). Any
RMS computed including them is meaningless.

---

## 2. Angular error propagates linearly to position

6-camera dataset 3 geometry, 3000 Monte Carlo trials per row.

| Orientation error | Position error |
|---|---|
| 0.5° | 0.27 m |
| 1.0° | 0.53 m |
| 2.0° | 1.06 m |
| 3.0° | 1.58 m |
| 4.5° | 2.43 m |
| **7.0° (measured)** | **3.68 m** |

Strictly linear, no threshold or cliff. Two consequences:

**Voxel size follows calibration error, not available memory.** No accuracy is
gained by making voxels finer than the error feeding them. 1.5 m at 7° is the
matched choice; 0.5 m would cost 27× the memory for no improvement.

**Stated system accuracy is 4–7 m**, derived from this table rather than asserted.

---

## 3. The 3-camera minimum

Reconstructions placing the object below ground level, by camera count
(553 dataset 3 frames):

| Cameras | Underground |
|---|---|
| 2 | **13.4%** |
| 3 | 0.0% |
| 4 | 0.0% |
| 5 | 0.0% |
| 6 | 0.0% |

Overall 11 of 553 frames (2.0%) reconstruct underground — all of them 2-camera.

**Rule: require 3+ cameras per cluster.** Costs about 15% of frames and removes
every gross error. Enforced by `min_cameras` in `fod/voxel/cluster.py`.

---

## 4. Thin rays do not work — the failure that shaped the design

With infinitely thin rays, the maximum number of cameras contributing to any
single voxel was **1**. Across every window and every window length tested. Zero
clusters, always.

The cause is geometric, not a coding error: 7° of orientation error separates rays
by roughly 6.7 m at working range, so they never share a 1.5 m cell.

Fix: **uncertainty-cone splatting**, cone half-angle `ray_sigma_deg`, radius
= range × tan(σ).

| σ | Max cameras/voxel | Clusters |
|---|---|---|
| 0° (thin) | 1 | 0 |
| 3° | 5 | 0–1 |
| 5° | 6 | 1 |
| **7° (measured)** | 6 | 1, agrees with least-squares to ~2.5 m |

**σ is the measured calibration error, not a tuning knob.** Cones shrink
automatically as calibration improves — at dataset 5's projected sub-degree
accuracy they nearly vanish.

### The bug this introduced

Cone weight was initially volume-normalised (`/ ow.sum()`). Because near-camera
cones are small, their voxels received the largest weight, so the brightest cells
in the grid sat *on the cameras* rather than on the drone. The system ran, produced
output, and was completely wrong.

Corrected to peak-normalisation. Documented in `fod/voxel/grid.py` because it is
exactly the kind of error that looks like a working system.

---

## 5. Detection throughput

Single core, measured: MOG2 + morphology + contours.

| Resolution | Downscale | CLAHE | ms/frame | fps |
|---|---|---|---|---|
| 1080p | none | yes | 58.2 | 17.2 |
| 1080p | none | **no** | 33.0 | **30.3** |
| 1080p | 0.5× | yes | 13.1 | 76.3 |
| 1080p | 0.5× | no | 9.0 | 110.7 |
| 4K | none | yes | 234.3 | 4.3 |
| 4K | 0.5× | no | 38.1 | 26.2 |

**CLAHE costs 43% of the frame budget.** It was chosen in Phase 1 for sky-lighting
stability, but MOG2 already adapts to lighting, so on a fixed tripod with locked
exposure the benefit may be zero. Default is now `none`; enable it only if an A/B
test against measured recall justifies the cost.

### Dataset 5 budget

151,937 frames across six cameras. Detection is embarrassingly parallel across
cameras, so wall-clock is set by the longest camera.

| Decimation | Frames | Effective rate | 6 cores |
|---|---|---|---|
| 1/1 | 151,937 | 37.5 fps | ~32 min |
| 1/3 | 50,646 | 12.5 fps | ~11 min |
| 1/6 | 25,323 | 6.3 fps | ~5 min |

**Temporal decimation is free.** Because `cam*_frame_ts.txt` gives exact per-frame
timestamps, skipping frames costs nothing in synchronisation accuracy. At 1/3
decimation there are still ~6 detections per camera per 0.5 s window.

### Hardware

Any modern 6–8 core laptop. No GPU. 16 GB RAM comfortable, 8 GB workable. **SSD
matters more than CPU** — six parallel decoders bottleneck on a spinning disk.
Budget 100 GB free for source video plus intermediates.

---

## 6. Self-calibration feasibility (dataset 5)

Dataset 5 has no 2D labels, and cam1, cam2 and cam4 have no calibration file
anywhere. But 1,512 ground-truth 3D positions at millimetre accuracy span a
91 × 84 × 42 m volume — richly non-planar, and far better conditioned than a flat
checkerboard.

Simulated with 1.5 px detection noise, seeded from a deliberately crude generic
guess (fx = 1500, cx = 960, no per-camera information):

| Camera | fx err | fy err | cx err | cy err | Rotation err |
|---|---|---|---|---|---|
| huaweiP40pro | 0.6 px | 0.0 px | 5.9 px | 4.3 px | 0.215° |
| gopro7 | 1.2 px | 1.1 px | 1.2 px | 0.8 px | 0.081° |
| samsungS10 | 0.0 px | 0.5 px | 6.7 px | 9.6 px | 0.349° |

With intrinsics known, plain PnP gives 0.0147°–0.0744°.

### Minimum labels required

| Labels/camera | Rotation error |
|---|---|
| 20 | 0.92° |
| 50 | 0.58° |
| 150 | 0.41° |
| 300 | 0.24° |
| 900 | 0.10° |

**~50–100 hand-clicked points per camera, spread across the flight volume,
unblocks dataset 5 entirely.** Even the weakest row beats dataset 3's 7.04° by an
order of magnitude.

Real data will be worse than simulation — label noise, drone-centre offset, 9 Hz
ground truth interpolated onto 30 Hz frames. Even 10× worse is under 1°.

Note that cam0 (sonya5100), cam3 (sonyG) and cam5 (sonynex5n) already have
calibration files. That is three cameras, exactly the pipeline minimum, so
cam1/2/4 can be dropped without argument.

---

## 7. Working result (dataset 3)

53-second segment, `--window 0.5 --voxel-size 1.5 --samples 3 --percentile 99.8
--ray-sigma 7`:

| Quantity | Value |
|---|---|
| Windows processed | 107 |
| Clusters extracted | 79 (0.74/window) |
| Tracks formed | 2 |
| Primary track | 68 points over 39.0 s |
| Path length | 216.3 m |
| Speed | mean 1.23 m/s, max 6.21 m/s |
| Cameras per cluster | min 3, mean 4.4, max 6 |

Top-down view shows a coherent closed circuit; the altitude profile is smooth.

---

## 8. Calibration ambiguities resolved

Three questions that consumed time and are now closed. Recorded so they are not
reopened.

**`mate10_1` vs `mate10_2`** — byte-identical K and distCoeff. Only fps differs
(29.727612 native vs 30.0 remapped). Never a calibration question at all.

**`sony5n_1440x1080` fx=1176.9 vs fy=1572.9** — a genuine anamorphic squeeze, not
a bug. Verified: the 1920 calibration scaled by 0.75 gives fx 1184.0 (vs 1176.9)
and cx 701.2 (vs 702.7). This retracts an earlier claim that it was the likely
source of error.

**`sonyG_1` vs `sonyG_2`, and the sony5n variants** — triangulating from other
cameras and refitting the target rotation gave differences of 0.24° and 0.94°
against a 6–13° noise floor. **Not discriminable.** Calibration choice is not the
dominant error source; further time on it has no return.

---

## 9. Known defects

Fixed in this repository:

| # | Defect | Fix |
|---|---|---|
| 1 | `direction_lookback` configurable but never plumbed through | passed to `CentroidTracker` and `draw_tracks` |
| 2 | `frame_index` a local counter, not the decoder's frame number | `native_frame_id` from `CAP_PROP_POS_FRAMES` |
| 3 | `VideoWriter` hardcoded 30 fps | uses source fps, warns when unknown |
| 4 | No track confirmation; every noise blob spawned a track | `min_hits`, `Track.confirmed` |
| 5 | Greedy nearest-centroid matching | Hungarian (`linear_sum_assignment`), greedy fallback |
| 6 | Fixed 960×540 resize invalidated every K | `preserve_aspect` + `to_native()` |
| 7 | `design_notes_voxel_integration.md` stale | replaced by `third_party/ray_voxel/README.md` |

Defect 2 deserves emphasis: joining detections to dataset 5's `cam*_frame_ts.txt`
is keyed on the decoder's frame number. A `frame_index`-based join looks correct
and produces a wrong time base that propagates into every 3D position with no
visible symptom.

Defect 6 likewise: a standardized centroid against a native-resolution K yields a
bearing wrong by the scale ratio, and the output stays plausible throughout.

Outstanding, disclosed rather than hidden:

- Tripod drift over a 10-minute outdoor recording is unverified. MOG2 assumes a
  static camera; snow and wind make this worth checking before trusting dataset 5
  detections.
- CLAHE has never been A/B tested against measured recall.
- The 2D tracker has not been exercised on genuinely crossing trajectories.
