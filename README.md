# Multi-camera airspace monitoring system

Real-time per-camera motion detection feeding windowed 3D localisation over a
fixed monitored volume, using consumer smartphone cameras.

**Measured accuracy:** 4–7 m reconstruction error on dataset 3, derived from a
validated 7.04° camera orientation error. Requires 3+ cameras per cluster.

## Quick start

```
pip install -r requirements.txt
python fod.py demo
```

No install step, no compiler. Runs from wherever the folder sits.

| Command | Purpose |
|---|---|
| `python fod.py demo` | full walkthrough, cached, instant |
| `python fod.py calibration` | Tier 2 error budget |
| `python fod.py reconstruct --cached --plot` | Tier 3 result + trajectory plots |
| `python fod.py angular --source 0 --show` | Tier 1 live on webcam |
| `python fod.py calibrate-camera --images ./board --output phone.json` | calibrate a camera |

## Design premise

The system assumes a **deliberately surveyed camera cluster**: known positions,
known intrinsics, shared time base. Each public dataset violates a different part
of that premise, which is what makes them useful as staged tests.

| Dataset | Lacks | Consequence |
|---|---|---|
| 1, 2 | synchronisation | cannot form time windows |
| 3 | camera orientations | solved here (Tier 2) |
| 4 | camera positions | dropped |
| 5 | 2D labels | detector must supply them |

## The three tiers

**Tier 1 — single camera.** Motion detection and **angular velocity** in
degrees per second. A single camera cannot recover metric speed: range and speed
are not separable from image motion. Angular rate is the observable component,
and composes with range from Tier 3 as `v_tangential = ω × range`.

**Tier 2 — geometry.** 18 rotational DOF for dataset 3's six cameras, validated
out-of-sample on 553 held-out frames.

**Tier 3 — 3D reconstruction.** Voxel-grid ray accumulation over 0.5 s windows
with uncertainty-cone splatting, clustering, and 3D tracking.

## Documentation

- `SUBMISSION.md` — framing, live-demo checklist, known defects
- `docs/REPO_LAYOUT.md` — layout and build instructions
- `docs/error_budget.md` — every measured number
- `docs/dead_ends.md` — what was tried and abandoned
- `tools/README.md` — the orientation solve
- `third_party/ray_voxel/README.md` — inherited code and what changed

## Data

Datasets are not committed. Source:
https://github.com/CenekAlbl/drone-tracking-datasets
Place extracted datasets under `data/` (gitignored).
