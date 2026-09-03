# What to submit, and how to run it

## Project framing

> A multi-camera airspace monitoring system using consumer smartphone cameras:
> real-time per-camera motion detection feeding windowed 3D localisation over a
> fixed monitored volume.

Never state that bare. Always attach the numbers: 3+ cameras per cluster, 4–7 m
reconstruction accuracy on dataset 3, derived from a measured 7.04° orientation
error.

**Say "airspace", not "aerospace".** Aerospace is the aircraft-and-space
industry; airspace is the monitored volume.

### The design premise, and why it does the work of an excuse without being one

The system assumes a **deliberately surveyed camera cluster**: known positions,
known intrinsics, shared time base. Every public dataset violates a different
part of that premise:

| Dataset | What it lacks | Consequence |
|---|---|---|
| 1, 2 | synchronisation | cannot form time windows |
| 3 | camera orientations | solved here, Tier 2 |
| 4 | camera positions | dropped |
| 5 | 2D labels | detector must supply them |

Stated up front, each dataset limitation becomes a consequence of a declared
assumption rather than a gap in the work. This is the strongest framing
available and it happens to be true.

### The three tiers

**Tier 1 — single camera.** Detects moving objects and measures **angular
velocity** in degrees per second.

A single camera *cannot* measure metric speed: an object at range 2d moving at
speed 2v is pixel-for-pixel identical to one at range d moving at v. The
observation is rank-deficient in range. Do not claim single-camera speed — it is
the first thing an examiner will attack, and correctly.

Angular velocity is the half of the problem one camera genuinely solves, and it
composes: `v_tangential = ω_rad/s × range_m`. Tier 3 supplies the range. So
Tier 1 is not a weaker version of the goal, it is the front half of it, and
Tier 3 exists precisely because of the limitation Tier 1 identifies.

**Tier 2 — geometry.** Solves 18 rotational DOF for dataset 3's six cameras.
Validated out-of-sample on 553 held-out frames: 7.04° mean.

**Tier 3 — 3D reconstruction.** Voxel-grid ray accumulation over 0.5 s windows,
clustering, 3D tracking.

---

## Running it

```
pip install -r requirements.txt
python fod.py demo
```

No install step, no compiler, no C++ toolchain. `fod.py` inserts its own
directory on the path so it runs from wherever it sits — a committee laptop may
not permit `pip install -e .`, and configuring PYTHONPATH mid-demo is not a plan.

| Command | Purpose |
|---|---|
| `python fod.py demo` | full walkthrough, cached, instant |
| `python fod.py calibration` | Tier 2 error budget |
| `python fod.py reconstruct --cached --plot` | Tier 3 result + ASCII trajectory |
| `python fod.py angular --source 0 --show` | Tier 1 live on webcam |
| `python fod.py angular --source http://IP:8080/video --calib phone.json --show` | Tier 1 on phone stream |
| `python fod.py calibrate-camera --images ./board --output phone.json` | calibrate the demo phone |

The ASCII plots exist so Tier 3 renders on any terminal with no display, no GUI
toolkit, and no dependency on matplotlib being importable at demo time.

---

## Live demo checklist

Run through this the night before, on the actual demo laptop, on the actual
network.

**Lock exposure, focus and white balance in the IP Camera app.** This is the
single highest risk item. MOG2 models per-pixel background statistics; if
auto-exposure hunts even slightly, every pixel changes at once, the whole frame
is marked foreground, and the demo displays a screen of white noise with no
recoverable explanation. Everything else on this list is an inconvenience — this
one is fatal.

- [ ] Exposure / focus / white balance locked in the app
- [ ] Anti-flicker set to 50 Hz if the room has LED or fluorescent lighting
      (50 Hz lighting against a 30 fps sensor beats, and MOG2 reads the beat as
      motion)
- [ ] Phone rigidly mounted — MOG2 assumes a static camera; handheld is unusable
- [ ] Windows Firewall prompt already accepted for Python, once, in advance
- [ ] Stream URL passed as an argument, not baked into a config — the MiFi will
      hand out a different IP than it did at home
- [ ] Moving subject decided and rehearsed. A tossed ball is repeatable and
      re-runnable; pointing at the sky and hoping something flies past is a coin
      flip during a fifteen-minute defence
- [ ] `python fod.py demo` run once on the demo machine
- [ ] **Screen recording of the live stream working**, as fallback. If the
      network dies, play it: "same pipeline on the live feed, recorded
      yesterday, because I didn't want to bet the demo on the venue's network."
      That reads as preparation, not failure.

### Calibrate the demo phone — 20 minutes, three separate payoffs

Print a checkerboard, take 15–20 photos at varied angles filling different parts
of the frame, run `fod calibrate-camera`.

Without it you can only honestly report px/s. Claiming °/s from an uncalibrated
phone is the same category of overclaim as single-camera speed. With it you get:
physically meaningful units; a live demonstration of the same calibration
machinery Tier 2 is built on; and the ability to say "this phone was calibrated
with the procedure used on the dataset cameras" — which ties the live demo into
the system instead of leaving it a detached appendix.

Note the focal length is in **pixels** and depends on capture resolution. A
calibration at 1920×1080 does not apply to a 1280×720 stream; `--calib` detects
the mismatch and rescales, but check the printed note.

---

## What to submit

**Include:**

```
fod.py                  launcher
requirements.txt        pinned, wheels only
README.md
SUBMISSION.md           this file
fod/
  cli.py                unified interface
  angular.py            Tier 1 angular velocity
  sources/scene.py      dataset loaders, camera model, ray generation
  sources/live.py       live-feed interface (stubs, documented)
  voxel/grid.py         accumulator with cone splatting
  voxel/cluster.py      connected components + min-camera rule
  track3d/kalman.py     constant-velocity 3D tracker
  run_pipeline.py       full reconstruction entry point
  viz/plot_result.py
tools/                  Tier 2 orientation solve — see below
results/                cached outputs the demo reads
docs/
  decisions/            why voxel over EKF, why dataset 5, why cones
  dead_ends.md
  error_budget.md
```

**On hiding Tier 2:** presenting it through documentation is fine, but **do not
remove the code from the repo.** Put it in `tools/` with a README pointing at
it. A submission that references work not present invites "so where is it?" —
and Tier 2 holds your most rigorous material: the out-of-sample validation, the
diverged-point catch, the 3+ camera rule. Leaving it in a labelled folder costs
nothing and is insurance if anyone probes.

**Exclude:** `data/` and video (gitignore them), the inherited C++ (reference it
in docs as heritage — it is not a runtime dependency, and shipping it invites
questions about code you did not write), scratch analysis scripts.

**Mention working alone in a summer semester once, in the written report**, as a
scope note. Not in the oral defence, and not twice. Stated once it is context;
repeated it reads as pre-emptive excuse, and the results do not need one.

---

## Known defects, disclosed

Listing these is a strength. An examiner who finds an undisclosed bug has caught
you; one who reads your own list of known issues sees engineering judgement.

- `tracking.direction_lookback` is read from config but never plumbed through
- `VideoWriter` hardcodes 30 fps regardless of source rate
- Greedy nearest-centroid association; should be Hungarian
  (`scipy.optimize.linear_sum_assignment`) before the 3-drone case
- Fixed 960×540 resize in the Phase 1 path invalidates K — the CLI corrects
  centroids back to native pixels, the older path does not
- Two diverged points (indices 11, 50) in `solved_camera_poses.json` at 1.6M and
  2.6M metres; joint refine has no depth bound. Cull beyond ±200 m before
  reporting RMS
- CLAHE costs 43% of the frame budget and has never been A/B tested against
  MOG2's own light adaptation
- Tripod drift over a 10-minute outdoor recording is unverified

---

## Dataset 5: the honest status

Confirmed from the upstream repository: dataset 5 has **no 2D labels**, and the
"3D orientation: Yes" column refers to the **drone's** roll/pitch/yaw from IMU
fusion, not camera extrinsics. `fused_pose.txt` columns are
`Timestamp, X, Y, Z, Roll, Pitch, Yaw, Std.X, Std.Y, Std.Z, TrackingStatus`.
Camera orientations are not provided.

So the path is: run the detector on video → 2D detections → PnP against the 1512
mm-accurate GT positions → camera orientations → reconstruction.

Only cam0 (sonya5100), cam3 (sonyG) and cam5 (sonynex5n) have calibration files.
That is three cameras — exactly the pipeline minimum. cam1, cam2 and cam4 can be
dropped without argument.

Measured feasibility: 50 correspondences per camera give 0.58° orientation
error; 150 give 0.41°; 300 give 0.24°. Against dataset 3's 7.04°, even the
weakest of those is an order-of-magnitude improvement.

**This is the better story.** On dataset 3 the detections were given and you
solved the geometry. On dataset 5 the geometry is derived *from your own
detector's output* — which puts the image processing on the critical path of the
headline result rather than beside it.

**Set a hard cutoff date.** Dataset 3 already produces a complete result: 68
track points over 39 s, 216.3 m path, coherent closed circuit. Dataset 5 is
upside, not a dependency. If it is not working by your chosen date, ship
dataset 3 and present dataset 5 as characterised future work with the numbers
above. Decide that date now, in writing, while it is cheap.
