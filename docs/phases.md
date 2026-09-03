# How the system is organised, and how it got here

## The design premise

The system assumes a **deliberately surveyed camera cluster**: known positions,
known intrinsics, a shared time base. This is not a convenience assumption — it
is what makes metric 3D reconstruction possible at all from consumer cameras
with no depth sensing.

Every public dataset violates a different part of that premise, which is what
makes them useful as staged tests rather than a single benchmark:

| Dataset | Lacks | Consequence here |
|---|---|---|
| 1, 2 | synchronisation | cannot form time windows; not used |
| 3 | camera orientations | solved in Tier 2; primary result |
| 4 | camera positions | no `campos.txt`; dropped |
| 5 | 2D labels | detector must supply them |

Stated up front, each limitation is a consequence of a declared assumption rather
than a gap in the work.

## The three tiers

### Tier 1 — single camera

Detects moving objects (MOG2 background subtraction, morphological cleanup,
contour extraction), tracks them frame to frame, and reports **angular velocity**
in degrees per second.

A single camera **cannot** measure metric speed. An object at range 2d moving at
speed 2v is pixel-for-pixel identical to one at range d moving at v — the
observation is rank-deficient in range, and no amount of image processing
recovers it.

Angular rate is the component a single camera genuinely observes, and it
composes:

    v_tangential = ω_rad/s × range_m

Tier 3 supplies the range. So Tier 1 is not a weaker attempt at the goal; it is
the front half of it, and Tier 3 exists precisely because of the limitation
Tier 1 identifies. Note the tangential qualifier: motion directly toward or away
from a camera produces no angular rate and is invisible to this measurement.

Reporting degrees requires the focal length in pixels, so an uncalibrated camera
reports px/s and says so. `fod calibrate-camera` closes that gap in about twenty
minutes with a printed checkerboard.

### Tier 2 — camera geometry

Dataset 3 gives camera positions at true metric scale, so only the 18 rotational
degrees of freedom needed solving — no scale or gauge ambiguity, an unusually
well-posed version of the problem.

Solved by cross-camera ray agreement, seeded from pairwise essential-matrix
decomposition, then jointly refined. **Validated out of sample** on 553 held-out
frames: 7.04° mean.

Code lives in `tools/`. It ran once to produce
`results/solved_camera_poses.json`, which the pipeline consumes; nothing in
`fod/` imports from it.

### Tier 3 — 3D reconstruction

Rays from each camera's detections are accumulated into a voxel grid over 0.5 s
windows. Peaks are extracted by percentile threshold and connected components,
filtered by a minimum camera count, and linked into 3D tracks by a
constant-velocity Kalman filter.

Three properties of this stage came from measurement rather than choice, and are
documented in `error_budget.md`: voxel size is matched to calibration error
rather than made as fine as memory allows; rays are splatted as uncertainty cones
because thin rays never intersect at 7° of error; and clusters require three
cameras because two-camera reconstructions land underground 13.4% of the time.

## Why the repository is not organised by phase

The work happened in three phases and the phases were real, but they are a story
about chronology, not a useful way to arrange code. A reviewer opening `phase1/`
then `phase2/` learns nothing about what the system does.

So the layout is functional — `fod/detect/`, `tools/`, `fod/voxel/` — and the
chronology lives here.

## Scope, honestly

The project was built by one person over a summer semester. What that bought:
a validated error budget, a working reconstruction on real multi-camera data, and
a documented failure that was hit and fixed rather than hidden.

What it did not buy: object classification (bird vs aircraft vs drone), sub-metre
accuracy on dataset 3, dataset 4, or a completed dataset 5 chain. Each of those is
scoped in `error_budget.md` with the specific work required, not left as a vague
"future work" gesture.
