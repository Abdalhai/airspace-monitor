# Repository layout and submission build

## The tree

Organised by **function**, not by phase. Phases are a story about how the work
happened; they are not a sensible way to arrange code, and a reviewer opening
`phase1/` then `phase2/` learns nothing about what the system does. The phase
narrative lives in `docs/phases.md` where it belongs.

```
airspace-monitor/
├── README.md                    ← project front door
├── SUBMISSION.md                ← framing, demo checklist, known defects
├── requirements.txt             ← pinned, wheels only, no compiler
├── fod.py                       ← launcher: python fod.py demo
├── .gitignore
│
├── config/
│   └── default.yaml             ← runtime config, CLI overrides it
│
├── fod/                         ← THE PACKAGE (everything that runs)
│   ├── cli.py                   ← unified interface, all subcommands
│   ├── angular.py               ← Tier 1 angular velocity + calibration
│   ├── detect/                  ← ⚠ DROP PHASE 1 HERE
│   │   ├── capture.py
│   │   ├── standardization.py
│   │   ├── motion_detection.py
│   │   ├── tracking.py
│   │   ├── visualization.py
│   │   └── pipeline.py          ← main.py becomes this (see below)
│   ├── sources/
│   │   ├── scene.py             ← dataset loaders, camera model, rays
│   │   └── live.py              ← live-feed interface (documented stubs)
│   ├── voxel/
│   │   ├── grid.py              ← accumulator, cone splatting
│   │   └── cluster.py           ← connected components, min-camera rule
│   ├── track3d/kalman.py
│   ├── viz/plot_result.py
│   └── run_pipeline.py
│
├── tools/                       ← ⚠ DROP PHASE 2 HERE (see tools/README.md)
│   ├── README.md                ← already written
│   ├── dataset3_scene.py
│   ├── detection_converter.py
│   ├── orientation_solver.py
│   ├── rtk_and_sync.py
│   └── run_joint_refine.py
│
├── results/                     ← COMMITTED, small, what `fod demo` reads
│   ├── solved_camera_poses.json ← ⚠ DROP THIS HERE TOO
│   ├── track_3d.csv
│   ├── detections_3d.json
│   └── reconstruction.png
│
├── docs/
│   ├── phases.md                ← the three-tier narrative
│   ├── error_budget.md          ← every measured number in one place
│   ├── dead_ends.md             ← what was tried and abandoned
│   ├── REPO_LAYOUT.md           ← this file
│   └── decisions/               ← one file per decision, ADR style
│
├── third_party/ray_voxel/       ← ⚠ DROP INHERITED C++ HERE, unmodified
│   └── README.md                ← already written
│
├── scripts/                     ← numbered analysis scripts
│
└── data/                        ← GITIGNORED, never committed
```

The four ⚠ markers are the only places your existing code goes.

## Where each existing file lands

| Your file | Destination | Note |
|---|---|---|
| `capture.py` | `fod/detect/` | add `native_frame_id` — see defects |
| `standardization.py` | `fod/detect/` | make CLAHE a flag, default off |
| `motion_detection.py` | `fod/detect/` | as is |
| `tracking.py` | `fod/detect/` | swap greedy for Hungarian |
| `visualization.py` | `fod/detect/` | as is |
| `main.py` | `fod/detect/pipeline.py` | becomes `run_single_camera()`, called by `cli.py` |
| `config.yaml` | `config/default.yaml` | already merged with Tier 3 settings |
| Phase 2 `.py` files | `tools/` | unchanged |
| `solved_camera_poses.json` | `results/` | cull diverged indices 11, 50 first |
| `ray_voxel.cpp` and friends | `third_party/ray_voxel/` | unmodified, reference only |
| `design_notes_voxel_integration.md` | delete | stale, superseded by `third_party/ray_voxel/README.md` |

### The one refactor that matters

`main.py` becomes `fod/detect/pipeline.py` exposing a function, not a script:

```python
def run_single_camera(source, config, on_frame=None):
    """Yields (native_frame_id, timestamp, tracks) per frame."""
```

Then `cli.py` calls it instead of carrying its own detection loop. Right now the
CLI reimplements detection inline, which means the submission would contain two
detectors doing the same job — the first thing a reviewer would flag. Wiring
`cli.py` to your modules removes the duplicate and makes Phase 1 genuinely part
of the running system rather than a folder someone has to take on trust.

Upload Phase 1 and I'll do this wiring and fix the seven known defects in place.

## Building the submission directory

The submission is the repo minus datasets, git metadata, and scratch work.

**Windows PowerShell:**
```powershell
$src = "airspace-monitor"; $dst = "SUBMISSION_<yourname>"
Remove-Item -Recurse -Force $dst -ErrorAction SilentlyContinue
Copy-Item -Recurse $src $dst
Remove-Item -Recurse -Force "$dst\.git","$dst\data","$dst\scripts" -ErrorAction SilentlyContinue
Get-ChildItem -Path $dst -Include __pycache__,*.pyc,*.bin -Recurse -Force |
    Remove-Item -Recurse -Force
Compress-Archive -Path $dst -DestinationPath "$dst.zip" -Force
```

**Linux / macOS:**
```bash
src=airspace-monitor; dst=SUBMISSION_yourname
rm -rf "$dst" && cp -r "$src" "$dst"
rm -rf "$dst"/.git "$dst"/data "$dst"/scripts
find "$dst" \( -name __pycache__ -o -name '*.pyc' -o -name '*.bin' \) \
     -exec rm -rf {} + 2>/dev/null
zip -r "$dst.zip" "$dst"
```

Then verify the archive actually works from a clean extraction — the failure
mode is a submission that only runs on the machine it was built on:

```bash
cd /tmp && unzip -q SUBMISSION_yourname.zip && cd SUBMISSION_yourname
python -m venv .v && . .v/bin/activate      # Windows: .v\Scripts\activate
pip install -r requirements.txt
python fod.py demo
```

If `demo` prints the trajectory plots, the submission is sound. Do this at least
once before the deadline, not on the day.

**Keep `results/` in the submission.** It is a few hundred kilobytes and it is
the reason `fod demo` works without any dataset present. A submission that
requires a 15 GB download before it does anything is a submission nobody runs.

## Git: how to commit this

Do not push it as one commit. A single dump of a finished project reads as
either imported from elsewhere or written the night before. A sequence of
commits that mirrors how the work actually happened is a record of the work.

```bash
cd airspace-monitor
git init && git branch -M main

# 1 -- skeleton first, so history starts with structure
git add .gitignore README.md requirements.txt fod.py config/
git commit -m "Project skeleton, runtime config, pinned dependencies"

# 2 -- single-camera detection
git add fod/detect/ fod/__init__.py
git commit -m "Tier 1: single-camera motion detection and 2D tracking"

# 3 -- the angular velocity argument
git add fod/angular.py
git commit -m "Tier 1: angular velocity with checkerboard calibration

Single camera cannot recover metric speed -- range and speed are not
separable from image motion. Angular rate is the component that is
observable, and composes with multi-camera range to give speed."

# 4 -- geometry, with the number that matters in the message
git add tools/ results/solved_camera_poses.json
git commit -m "Tier 2: camera orientation solve for dataset 3

18 rotational DOF; positions known at metric scale so no gauge freedom.
Validated out-of-sample on 553 held-out frames: 7.04 deg mean. An earlier
in-sample figure of 4.5 deg understated the error and is not used."

# 5 -- inherited code, isolated and attributed before it is superseded
git add third_party/
git commit -m "Inherited C++ voxel accumulator, unmodified, reference only"

# 6 -- reconstruction, and the failure that shaped it
git add fod/voxel/ fod/sources/
git commit -m "Tier 3: voxel accumulation with uncertainty-cone splatting

Thin rays yield max 1 camera per voxel and zero clusters -- 7 deg of
orientation error separates rays by ~6.7 m at working range. Cone
half-angle is set from measured calibration error, not tuned."

# 7 -- tracking and pipeline
git add fod/track3d/ fod/viz/ fod/run_pipeline.py fod/sources/live.py
git commit -m "Tier 3: 3D tracking, live-feed interface stubs, pipeline entry"

# 8 -- the interface
git add fod/cli.py
git commit -m "Unified CLI with cached demo path for presentation"

# 9 -- results and documentation last
git add results/ docs/ SUBMISSION.md
git commit -m "Cached results, error budget, decision records"

git remote add origin https://github.com/<you>/airspace-monitor.git
git push -u origin main
```

Write the commit messages as above — a message that carries the finding
(`7.04 deg`, `max 1 camera per voxel`) makes the history itself evidence.

**Check before the first push:**

```bash
git count-objects -vH        # under a few MB
git ls-files | grep -Ei '\.(mp4|mov|zip|7z|bin)$'   # must be empty
```

GitHub rejects files over 100 MB, and a rejected push after committing a 4.5 GB
video is genuinely painful to unwind.

## The docs still to write

Four files, all short. Most of the content exists in our analysis already.

- **`docs/phases.md`** — the three-tier narrative and the surveyed-cluster
  premise, with the table of what each dataset lacks.
- **`docs/error_budget.md`** — every measured number in one place: 7.04° with
  percentiles and per-camera breakdown; angular-to-position propagation
  (1° → 0.53 m, 7° → 3.68 m); the 2-camera 13.4% underground result; the
  detection benchmark table; the cone-sigma table. This is the single most
  valuable document in the submission — it is what distinguishes measured work
  from work that merely runs.
- **`docs/dead_ends.md`** — `mvus` on datasets 1 and 2; the `rtk.txt` frame
  mismatch; `process_image.cpp` astronomy orientation; dataset 4 dropped for
  lack of `campos.txt`; the sony5n anamorphic scare that turned out to be
  correct; the calibration-variant comparison that showed the choice was not
  discriminable against the noise floor.
- **`docs/decisions/`** — one short file each: voxel over EKF; monitoring rather
  than real-time tracking; cone splatting; the 3-camera minimum; dataset
  selection.

## What not to do

**Do not rename things to sound impressive.** `fod/voxel/grid.py` is clearer
than `core/spatial/accumulation_engine.py`, and inflated naming invites a
reviewer to check whether the substance matches.

**Do not delete the dead ends.** They are evidence of process. A project with no
recorded failures either hid them or never probed hard enough to find any.

**Do not commit the datasets.** Link the source repository in the README instead.
