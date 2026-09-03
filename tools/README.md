# Camera geometry solve (Tier 2)

Standalone scripts, not part of the runtime package. They were run once to
produce `results/solved_camera_poses.json`, which the pipeline consumes. Nothing
in `fod/` imports from here.

They live in the repository rather than only in the write-up because a
submission that references work not present invites the question of where it is,
and this is the most rigorous component of the project.

## Files

| File | Role |
|---|---|
| `dataset3_scene.py` | dataset 3 loader, camera table, alpha/beta sync |
| `detection_converter.py` | raw detection files to the solver's format |
| `orientation_solver.py` | pairwise seed then joint refinement, 18 rotational DOF |
| `rtk_and_sync.py` | RTK parsing and synchronisation experiments |
| `run_joint_refine.py` | entry point producing `solved_camera_poses.json` |

## What was solved, and why it was tractable

Dataset 3 provides camera positions at true metric scale in `campos.txt`, so
only the 18 rotational degrees of freedom needed solving. No scale or gauge
ambiguity — an unusually well-posed version of the problem.

## Result

Validated **out of sample** on 553 frames the optimiser never saw: mean 7.04°,
median 6.53°, 90th percentile 11.6°.

An earlier in-sample figure of 4.5° was measured on the same 143 points the
optimiser was free to move and understated the error. Reconstruction accuracy is
quoted from the 7.04° figure.

Per camera: cam5 4.94°, cam2 5.36°, cam0 6.39°, cam4 6.58°, cam1 7.91°,
cam3 11.27°.

## Known defect

Two points in `solved_camera_poses.json` (indices 11 and 50) diverged to 1.65M
and 2.6M metres — the joint refinement has no depth bound. Cull points beyond
±200 m before reporting any RMS figure.
