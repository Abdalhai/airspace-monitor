# Dead ends

Approaches tried and abandoned. Recorded so they are not re-attempted, and
because a project with no documented failures either hid them or never probed
hard enough to find any.

---

## `mvus` on datasets 1 and 2

The dataset authors' own pipeline, applied to the two easiest datasets.

Neither has synchronisation data. `mvus` solves for temporal offsets jointly with
geometry, which is elegant but leaves both under-constrained without a
ground-truth time base to check against — there is no way to tell a good solve
from a plausible wrong one.

**Abandoned in favour of dataset 3**, which publishes a synchronisation table and
camera positions, making every intermediate result checkable.

---

## `rtk.txt` on dataset 3

The obvious route to ground truth: RTK positions to validate reconstruction
directly.

`rtk_dataset3.txt` is in a different coordinate frame from `campos.txt`, with no
published transform between them. Recovering the transform means solving for
7 parameters using the very reconstruction it was meant to validate — circular.

**Abandoned.** Tier 2 validation instead uses cross-camera ray agreement on
held-out frames, which needs no external frame. See `tools/rtk_and_sync.py` for
the parsing work, retained because the file is still the right starting point if
the transform is ever published.

---

## `process_image.cpp` orientation

The inherited C++ includes an orientation routine. On reading it, it solves an
**astronomical** orientation problem — plate-solving against a star field — not a
camera-extrinsics problem.

Wrong tool entirely. Not a defect in the inherited code; it was written for a
different purpose.

---

## Dataset 4

Seven cameras, moving clouds, a genuinely harder detection case — attractive as a
stress test.

It has **no `campos.txt`**. Without camera positions the scale and gauge freedoms
return, so it would require a full structure-from-motion solve before any of the
existing pipeline applies.

**Dropped.** It adds a solving stage without adding a result. The effort is better
spent on dataset 5, which does publish positions.

---

## The sony5n anamorphic scare

`sony5n_1440x1080.json` has fx = 1176.9 against fy = 1572.9 — a 33% discrepancy
that looks exactly like a calibration bug, and was initially reported as the
likely dominant error source.

It is not a bug. The sensor applies a genuine anamorphic squeeze in that mode.
Verified numerically: the 1920-width calibration scaled by 0.75 gives fx 1184.0
against the file's 1176.9, and cx 701.2 against 702.7 — agreement to under 1%.

**Claim retracted.** Recorded here because the retraction matters more than the
original observation.

---

## Choosing between calibration variants

Several cameras ship two calibration files (`sonyG_1` / `sonyG_2`, sony5n
variants). Significant time went into deciding which was correct.

Tested empirically: triangulate a point from the other cameras, refit the target
camera's rotation under each candidate calibration, compare. The differences were
**0.24° and 0.94°** — against a 6–13° noise floor.

**Not discriminable, and not the dominant error source.** The choice does not
matter at the accuracy this system operates at. Time spent here has no return
until orientation error drops below about 1°.

---

## Thin-ray voxel accumulation

Not abandoned so much as *diagnosed* — but it belongs here because it consumed
real time and looked like a coding error.

With infinitely thin rays the voxel grid produced zero clusters, always, in every
configuration. Maximum cameras contributing to any single voxel: 1.

The cause is geometric. 7° of orientation error separates rays by ~6.7 m at
working range; they never share a 1.5 m cell. No parameter tuning fixes this,
because nothing is wrong with the code.

Resolved by uncertainty-cone splatting with the cone half-angle set to the
measured calibration error. Full before/after table in `error_budget.md` §4.
