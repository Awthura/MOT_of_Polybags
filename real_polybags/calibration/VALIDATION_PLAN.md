# Validating the calibration — plan

How to establish that a finished calibration is *correct*, not merely that it
completed. Written as a plan: most of it is not built yet, and §7 says what to
build in what order.

---

## 1. The problem with the checks that already exist

The tool reports several numbers that look like validation and are not.

| what it reports | why it is not validation |
|---|---|
| **Reprojection error** (intrinsic and extrinsic) | Measured on the *same* corners the parameters were fitted to. It says the model reproduces its own training data. A degenerate pose set — all views frontal, all at one distance — yields an RMS of 0.2 px alongside a focal length 57% wrong. This was measured, not hypothesised. |
| **`verify_synthetic.py` / `verify_beltmap.py`** | These verify the *mathematics* against a known virtual camera. They prove the code recovers a `K` it was given. They say nothing about whether the number now sitting in `results/basler_1.json` describes the lens on the rig. |
| **Coverage and tilt warnings** | Necessary conditions, not sufficient ones. They catch a bad *sample*; they cannot catch a good sample of the wrong thing — a scaled board, say. |
| **Height vs anticipated** (the rig page) | Genuine, and the cheapest guard available, but it is one scalar compared against a tape measure. It catches gross scale error and nothing subtler. |

**The rule everything below follows: a validation must use information that was
not used in the fit.** Anything else is a consistency check, which is worth
having and is not the same claim.

---

## 2. What we are actually validating

The calibration exists to make one sentence true:

> A bag at belt coordinates (X, Y) mm is the same bag whichever camera saw it.

So the quantity that matters is **cross-camera positional agreement in
millimetres**, not focal length, not reprojection error. Everything else is
instrumental. A calibration with a beautiful RMS and 80 mm of cross-camera
disagreement has failed at its only job.

Two secondary claims also need support:

- positions are **metrically true**, not merely self-consistent (a uniform scale
  error keeps every camera agreeing with every other while all of them are
  wrong);
- the mapping is **stable over the session** — a camera that is bumped halfway
  through invalidates everything recorded after it, silently.

---

## 3. The accuracy floor — decide this before setting any threshold

`beltmap.parallax_error_mm()` already quantifies something that bounds the whole
exercise. The belt map assumes everything lies on Z = 0. A bag has height, so its
visible top projects outward from the camera's nadir: **10–45 mm of displacement
for a 60 mm bag**, in the verification rig, growing toward the frame edges.

That is a systematic bias, not noise. It does not average away.

The consequence is a design decision, not a detail: **there is no point driving
cross-camera agreement below the parallax bias** unless bag height is going to be
modelled. Chasing 2 mm agreement while an unmodelled 30 mm bias sits underneath
it is effort spent on the wrong term.

So the first task is not a measurement, it is a decision:

- **Option A — accept the bias.** Set thresholds at roughly the parallax scale
  (order 20–40 mm). Cheap, and probably sufficient if bags are well separated
  relative to that.
- **Option B — compensate.** Estimate bag height (the RealSense measures it
  directly) and project to the bag's *centroid* plane rather than Z = 0. Then
  tighter thresholds become meaningful.

Everything in §5 is written with thresholds as *placeholders to be set after the
first real run*, because a threshold invented before any real measurement exists
is a guess wearing a number.

---

## 4. Tier 1 — at the rig, while the board is still there

These cost minutes and must happen before the board is put away, because every
one of them is impossible to redo later without another rig session.

### V1 · Height against tape — **already automated**
Recorded in the plan as the anticipated height; the page compares and flags.
Catches: wrong board preset, scaled print, wrong square size. Cost: nothing.

### V2 · Known-length check
Click two points on the preview a known distance apart (a ruler laid on the
belt, or two marked points); compare the reported belt millimetres against the
tape.
Catches: scale error, gross homography error. Cost: ~1 min per camera.
*Currently manual — the click already reports millimetres, but nothing records
the comparison.*

### V3 · Belt-width check
Click both belt edges in each camera; the width should match the tape and should
**agree between cameras**.
Catches: scale error, and it is the one check that uses a feature every camera
can see without moving anything.

### V4 · Held-out board placement — **the strongest check available**
After every camera is solved, **move the board to a new position** in view of two
or more cameras, and click the *same physical corner* in each. Every camera
should report the same belt coordinates.

This is genuine held-out validation: the extrinsics came from a different
placement, so nothing about this measurement was used in the fit. It directly
measures the sentence in §2.

Repeat at 3–4 positions spread along and across the belt — error is not uniform,
and a single central test hides edge behaviour, which is exactly where the
homography and the residual distortion are worst.

Cost: ~10 min total. **This is the highest-value thing in the whole plan and it
is currently entirely manual.**

---

## 5. Tier 2 — offline, from recorded footage

The reason this tier matters: it needs **no rig access**, it is **repeatable**,
it can run over the footage already recorded, and it can be automated so it runs
again every time a calibration changes.

All of these use the belt's own physics as the reference. Nothing here consumes
the board, the tape, or anything used in the fit.

### V5 · Straight-line and constant-speed check
A conveyor moves objects in a straight line at constant speed. Therefore, in
belt coordinates, any tracked object must trace a straight line parallel to the
+Y axis, at constant mm/s.

Three residuals fall out of one trajectory fit:

| residual | what a failure means |
|---|---|
| **lateral deviation** from a straight line | homography error, or uncorrected lens distortion — and it will be worst at the frame edges, which is diagnostic |
| **heading** vs the +Y axis | the belt frame's rotation is wrong; usually a bad board alignment when defining the origin |
| **speed variation** along the track | scale is not uniform across the field — the classic signature of a pose solved from a board that was not flat |

This is strong because the reference (straightness, constancy) is *known a
priori* and needed no measurement at all. It is also the only check here that
validates the mapping across the **whole field of view** rather than at a few
clicked points.

Requires: object tracks. The existing detector output can supply them; a single
high-contrast marker taped to the belt would be cleaner and is worth one short
dedicated recording.

### V6 · Cross-camera hand-off continuity
Follow one object as it leaves camera A and enters camera B. In the overlap,
both should report the same belt position at the same instant; with no overlap,
the trajectory should be *continuous across the seam* once belt travel is
accounted for.

This measures cross-camera agreement **on the actual objects of interest**,
across the full session, rather than at four clicked corners — the field version
of V4.

One caveat that must be handled or the result is meaningless: `measure_sync.py`
established the cameras are **not frame-synced**, with skew at the frame-rate
floor. At 15 fps and a moving belt, a 1-frame offset is a real displacement. The
comparison must therefore be made against **time-interpolated** positions using
the recorded device timestamps, not against nearest frames. Otherwise this check
measures the sync error and reports it as calibration error.

### V7 · Rigid-size invariance
A rigid object's measured length must not change as it travels. Measure the same
object's extent in belt millimetres at several points along the belt; the
variation is a direct read-out of scale non-uniformity.

Needs no known ground-truth size — the *constancy* is the reference. If a known
size is available, it becomes an absolute scale check too.

### V8 · Static landmark agreement
Pick fixed features visible to more than one camera — a bolt, a frame edge, a
mark on the belt housing. Each camera maps it to belt coordinates; they must
agree. Costs one frame, no rig access, and can run on footage recorded months
ago.

Limitation: only valid for landmarks actually **on the belt plane**. A feature on
the machinery above the belt will disagree for correct reasons (parallax), and
mistaking that for calibration error would be an own goal.

---

## 6. Tier 3 — independent metric reference

Everything above is either self-referential or referenced to a tape measure. Two
sources of genuinely independent truth exist on this rig:

### V9 · RealSense depth
The D435 measures metric depth directly, by a completely different mechanism from
anything in this pipeline. Its depth to the belt surface can be compared against
the calibrated `height_above_belt_mm`, and its depth-derived point cloud gives an
independent estimate of the belt plane itself — including **how planar the belt
actually is**, which §3 assumes and nothing else checks.

This is the single most valuable check in the plan after V4, and it is the only
one that can catch a *uniform* scale error affecting every camera at once.
Requires `pyrealsense2`, which is not currently installed.

### V10 · RealSense factory intrinsics
The D435 ships factory-calibrated. Calibrating it with our board and comparing
`fx, fy, ppx, ppy` against `get_intrinsics()` is an acceptance test for the
*method itself* on real hardware. The Basler and Lucid cameras ship uncalibrated
and have nothing to compare against — which is precisely why this one has to be
done first. If our number disagrees with the factory's here, no result on the
other cameras can be trusted.

---

## 7. What to build, in order

Ordered by value per unit of effort. Estimates assume the existing core modules
are reused rather than reimplemented.

| # | Deliverable | Covers | Effort | Why this rank |
|---|---|---|---|---|
| 1 | **`validate_calibration.py`** — offline runner over `results/*.json` + footage | V5, V7, V8 | ~1 day | No rig needed, runs on footage that already exists, and re-runnable forever. Turns validation into something that happens automatically instead of something someone remembers to do. |
| 2 | **Point-pair capture in the rig page** — click the same feature in two cameras, record the disagreement into the calibration record | V2, V3, V4 | ~half day | V4 is the strongest check available and is currently entirely manual, meaning in practice it will be skipped or its result lost. Recording it into `results/<camera>.json` makes it part of the artefact. |
| 3 | **Sync-aware trajectory comparison** | V6 | ~half day on top of #1 | Needs the device timestamps already recorded by `record_all_5_cameras_macos.py`. Without the interpolation it reports sync error as calibration error, so it is worth doing properly or not at all. |
| 4 | **RealSense cross-check** | V9, V10 | ~half day + `pip install pyrealsense2` | Blocked on the SDK and on hardware access, but it is the only independent metric reference. Schedule it for whenever the rig is next available. |
| 5 | **Validation record + staleness** | all | ~2 h | A `validation` block in the per-camera JSON: when it was last validated, against what, with what residuals. The rig page's status board then shows `solved` vs `solved + validated`, and flags a calibration whose validation predates its last save. |

### Sketch of `validate_calibration.py`

```
python3 validate_calibration.py \
    --results results/ \
    --footage /path/to/experiments/Bulk_2_moving \
    --belt-speed-mm-s 250          # optional; enables absolute speed check
    --report validation/2026-08-03.md
```

Design constraints, carried over from what already works in this tool:

- **Reads `results/*.json` only** — never re-solves anything. Validation must not
  be able to quietly fix what it is checking.
- **Refuses to pass what it could not check.** Same principle as the anticipated
  measurements: a check with no reference reports `no ref`, never `ok`. A report
  of "3 passed, 5 unmeasurable" is useful; "8 passed" when 5 were skipped is
  worse than no report.
- **Reports per-camera and per-region**, not one aggregate. Error grows toward
  frame edges; a single mean is exactly the statistic that hides it.
- **Writes overlay images**, not only numbers — a trajectory drawn on the belt
  map is worth more than its RMS for diagnosing *why* something failed.
- **Exit code non-zero on failure**, so it can gate a pipeline run.

### Proposed thresholds — provisional, to be set from the first real run

| check | provisional threshold | basis |
|---|---|---|
| V1 height vs tape | ±5% | already implemented; tape accuracy on a ~1.4 m measurement |
| V2/V3 known length | ±1% of the measured distance | tape resolution over a ~500 mm span |
| V4 cross-camera agreement | ≤ parallax bias at that point (§3) | tighter is not meaningful without height compensation |
| V5 lateral deviation | RMS ≤ 5 mm | tighten once real numbers exist |
| V5 speed variation | CV ≤ 2% | belt drive should be far steadier than this |
| V7 size variation | ≤ 3% across the field | |
| V9 depth vs calibrated height | ±2% | D435 depth accuracy at ~1.4 m |

**These are starting points, not requirements.** The honest way to set them is to
run the first real calibration, measure what it actually achieves, and then set
the threshold where a *regression* would be caught — rather than picking a number
now and either failing everything or passing everything.

---

## 8. What none of this can catch

Stated because a validation suite that is trusted beyond its reach is worse than
none.

- **A uniform scale error shared by the board and the tape measure.** If the
  printed board is 3% small *and* the tape is misread the same way, every check
  in Tiers 1 and 2 agrees. Only V9 (RealSense depth) is independent enough to
  break the tie.
- **A non-planar belt.** Everything assumes Z = 0. A belt sagging between rollers
  breaks that assumption in a way that looks like position error and will be
  attributed to calibration. V9's point cloud is the only proposed check that
  measures planarity directly.
- **Drift after validation.** These are point-in-time measurements. A camera
  nudged the next day invalidates all of it with no signal. Mitigation is
  deliberate re-validation (item 5 above), not detection.
- **Systematic bag-height bias**, unless Option B in §3 is taken. It is
  quantified and predictable, but it is not removed by any check here — only
  measured.

---

## 9. Minimum viable validation

If only one afternoon is ever available for this:

1. **V4 at three board positions** — 15 min at the rig, and it directly measures
   the claim the whole calibration exists to support.
2. **V5 on one existing recording** — no rig needed, and it validates the mapping
   across the entire field of view rather than at a few points.
3. **Write both results into the calibration record**, so the next person can
   tell a validated calibration from an unvalidated one.

That combination gives held-out cross-camera agreement plus full-field geometric
consistency, which is most of what matters. The rest is refinement.
