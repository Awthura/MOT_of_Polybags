# OVGU AMS Calibration Tool — User Manual

A local web tool for calibrating the conveyor rig's cameras: lens **intrinsics**,
camera **extrinsics**, and a shared metric **belt map** that all cameras project
into.

---

## 1. Quick start

```bash
conda activate ams
cd /Users/awthura/OVGU/AMS/real_polybags/calibration
python3 app.py
```

You will see:

```
======================================================================
  OVGU AMS — camera calibration
======================================================================
  open   http://127.0.0.1:5000
  results -> .../calibration/results
  no hardware? choose the synthetic source to exercise the whole flow
======================================================================
```

Open **http://127.0.0.1:5000** in a browser. Stop the server with `Ctrl+C`.

Options:

```bash
python3 app.py --port 5001              # if 5000 is taken (macOS AirPlay uses it)
python3 app.py --host 0.0.0.0           # reachable from another machine on the LAN
python3 app.py --results-dir /some/dir  # write calibrations elsewhere
```

**Try it with no hardware first.** Choose **Synthetic camera** as the source and
walk the whole workflow. This is not a demo mode: the synthetic camera has a
*known* focal length and distortion, so the tool shows your recovered values
against the truth. Ten minutes doing this means you arrive at the lab knowing
the software works, instead of debugging a web app next to a running conveyor.

### Requirements

| | |
|---|---|
| Python | the `ams` conda env |
| Packages | `flask`, `opencv-python`, `numpy`, `Pillow` (all in `requirements.txt`) |
| Camera SDKs | only for live capture: `pypylon` (Basler), `pygobject`+Aravis (Lucid), `pyrealsense2` (RealSense) |
| Printed board | see §3 |

The tool runs entirely on your machine. Nothing is uploaded anywhere.

---

## 2. What the tool actually does

Three quantities, obtained in order, because each depends on the last.

**Intrinsics — what the lens does.**
A camera turns 3-D directions into pixels. That mapping is described by the
matrix `K` (focal lengths `fx`, `fy` and principal point `cx`, `cy`) and
distortion coefficients `D` (how the lens bends straight lines). These are
properties of *this camera with this lens at this resolution*, and they are
measured by photographing a board of known geometry from many angles.

**Extrinsics — where the camera is.**
Given `K` and `D`, one view of the board lying flat on the belt yields the
camera's position and orientation relative to the belt: `R` and `t`.

**The belt map — the shared frame.**
Treating the belt as the plane Z = 0, a homography maps pixels to belt
millimetres and back. Every camera solved against that same plane lands in one
coordinate system, so a bag at (X, Y) mm is the same bag no matter which camera
saw it.

### Why this is needed at all

The tracking code inherited from the synthetic track associates cameras without
calibration, using two cues: a **colour-class gate** (the synthetic set has 7
classes) and **left-to-right ordering** with an assumption that cameras are
synchronized.

Neither survives on the real rig. Real polybags are a **single class**, so the
colour gate does not exist; and measurement showed the cameras are **not
frame-synchronized**. Calibration replaces both with geometry.

---

## 3. Before the lab: print the boards

Two PDFs in [`boards/`](boards/):

| file | paper | board | square | dictionary |
|---|---|---|---|---|
| `charuco_AMS-small_A4.pdf` | A4 | 175 × 200 mm | 25 mm | `DICT_4X4_50` |
| `charuco_AMS-large_A3.pdf` | A3 | 252 × 324 mm | 36 mm | `DICT_5X5_100` |

**Two sizes are needed.** `basler_1` is an extreme close-up — the belt fills its
frame — while the other cameras see the whole belt width. A board large enough
for the wide cameras will not fit in `basler_1`'s view; one small enough for
`basler_1` is too coarse for the others. They use different ArUco dictionaries
so they can never be confused with one another.

**Printing rules — these matter more than they sound:**

1. **Print at 100% / "actual size".** Never "fit to page". Scaling shrinks the
   board a few percent, and every millimetre the tool later reports is wrong by
   that factor — with nothing able to detect it. The calibration will look
   perfectly healthy and the numbers will be wrong.
2. **Measure the printed 100 mm bar with a ruler.** Each page has one. If it is
   not exactly 100 mm, reprint.
3. **Matte paper.** Gloss blows out under the rig's lighting and corners are lost.
4. **Mount flat and rigid** — foamboard or stiff card. A bowed board is a
   systematic error no amount of averaging removes.
5. **Keep the footer.** It prints the full spec. A board whose parameters are
   unknown afterwards is scrap.

---

## 4. Walkthrough

### Step 1 — Setup

| Field | Meaning |
|---|---|
| **Camera name** | Identifies the calibration. Use the recording names: `basler_1`, `basler_2`, `lucid`, `rgbd_1`. Saved as `results/<name>.json`. |
| **Frame source** | `Synthetic` (no hardware), `Folder` (existing images), or a live camera. |
| **Board** | Must match the board you are physically holding. |

Press **Start session**. The live preview appears with a green overlay on
detected corners and a running corner count.

> If a camera source says "SDK present — camera must be connected" but starting
> fails, the SDK is installed and the camera is not reachable. See §7.

### Step 2 — Capture (intrinsics)

Hold the board in front of the camera and take 15–25 shots. **Space** captures,
so both hands stay on the board.

**This step decides the quality of everything downstream.** The single most
common way calibration fails is silently: a set of similar shots produces a
low error number and a badly wrong focal length. In testing, 14 flat-on shots
at one distance gave a reprojection error of 0.37 px — which looks excellent —
alongside a focal length **57% wrong**.

So:

- **Tilt the board, 20–45°**, in different directions. Tilt is what separates
  focal length from distance. Without it the two are mathematically ambiguous.
- **Reach the frame edges and corners.** Distortion is strongest at the
  periphery; a board only ever seen in the middle leaves it unconstrained.
- **Vary the distance** — near, middle, far.
- **Keep it sharp.** Motion blur moves corners. A blurred shot is worse than no
  shot.

Do **not** align the board square to the camera. Squareness is the failure mode
here. (It matters in step 4, not this one.)

The **coverage grid** on the right fills in as regions of the frame are
visited, and tells you what is still missing. Aim to fill it. Thumbnails of
captured shots appear below the preview; click **x** to drop a bad one.

### Step 3 — Intrinsics result

**Calibrate intrinsics** runs the solve and reports:

| Field | What it means |
|---|---|
| `fx`, `fy` | Focal length in pixels |
| `cx`, `cy` | Principal point — usually near the image centre |
| `D` | Distortion coefficients |
| **RMS** | Mean reprojection error. Below ~0.5 px is good — **but see the warning below** |
| **Frame coverage** | Regions of the image the board reached |
| **Tilt range** | Spread of board orientations |

> **A low RMS does not mean a good calibration.** It only says the solution is
> self-consistent with the shots you gave it. If those shots were all similar,
> it can be self-consistent and wrong. Read the coverage and tilt figures, and
> read the warnings.

Warnings are listed explicitly — too few views, insufficient coverage, no
tilted views, or one view far worse than the rest (usually a blurred frame,
worth deleting and recalibrating).

**vs known reference** appears when the truth is available: the synthetic
camera's true parameters, or the RealSense's factory intrinsics. This is the
only way to distinguish a *correct* calibration from a merely *plausible* one.
Green means agreement.

### Step 4 — Belt plane (extrinsics)

Now alignment matters. **Lay the board flat on the belt**, visible in the
preview, and press **Solve extrinsics**.

**Origin offset** places this board position within the belt coordinate system.
Leave it at 0 for the first camera — that defines the origin. For a camera
that sees a *different* stretch of belt, place the board there, measure how far
it is from the origin, and enter that offset. This is what ties cameras into one
frame **without requiring them to see each other**.

You get the camera's position in belt coordinates, its height above the belt,
and a reprojection error (under ~1 px is good). Sanity-check the height against
reality — if the camera is 1.4 m above the belt and the tool says 300 mm, the
board's square size does not match what was printed.

Then **click anywhere on the preview** to read that point in belt millimetres.
Clicking two points a known distance apart and comparing against a tape measure
is the fastest honest test of the whole chain.

---

## 4b. The extrinsics workflow — getting four cameras into one frame

Intrinsics and extrinsics are different kinds of job and are best done as two
separate phases.

**Intrinsics need no rig.** They describe the lens, not the mounting. You can
calibrate a camera on a bench, at your desk, at any time — as long as the lens,
focus and **resolution** are the ones you will record with, and are not touched
afterwards.

**Extrinsics need the rig, in its final state.** They describe where the camera
is. Any bump to the mount invalidates them.

So the natural order is: **all four cameras' intrinsics first, then all four
cameras' extrinsics in one sitting.** The tool supports this — when you start a
session for a camera that has been calibrated before, its saved intrinsics are
reloaded automatically and you can go straight to step 4.

### The actual problem: one shared origin

Each camera solves its own pose. What makes them a *system* is that they solve
against the **same belt origin**. There are two ways to achieve that.

#### Method A — one board placement, every camera (preferred)

If several cameras can see the same patch of belt:

```
1. STOP THE CONVEYOR.
2. Lay the board flat on the belt, inside the shared view.
3. DO NOT MOVE IT until every camera has been solved.
4. For each camera in turn:
     Setup -> enter that camera's name -> Start session
       (its intrinsics reload automatically)
     Step 4 -> leave both offsets at 0 -> Solve extrinsics
     Step 6 -> Save
```

Every camera is solved against one physical board placement, so they share an
origin **exactly** — no measurement, no error. Use this wherever it is possible.

#### Method B — measured offsets (for cameras that see different stretches)

If a camera cannot see the board where the first camera saw it:

```
1. Solve the first camera with offsets 0, 0. This DEFINES the belt origin.
2. Move the board to where the next camera can see it.
3. Measure the displacement from the original position:
     X = across the belt, Y = along the belt (direction of travel).
4. Enter those numbers as the origin offset, then solve.
```

Accuracy here is your tape measure's accuracy — a few millimetres, which
propagates directly into cross-camera agreement. Prefer Method A when you can,
and keep the moves square to the belt so X and Y stay meaningful.

### Five things that will bite you

1. **Stop the belt.** The board rests on the belt surface; a running conveyor
   carries it away. Less obviously, Method A depends on every camera seeing the
   *same* placement, which a moving board makes impossible.
2. **Do not nudge the board between cameras.** In Method A that is the entire
   basis of the shared frame.
3. **The belt is not a perfect plane.** It can sag between rollers. Place the
   board where the bags actually travel, not at an unsupported span, so Z = 0
   means the surface bags really sit on.
4. **Mount everything first.** Extrinsics describe the camera's position; if a
   camera is re-aimed afterwards, its extrinsics are void. Intrinsics survive
   re-aiming — only extrinsics need redoing.
5. **Resolution must match.** If reloaded intrinsics were measured at a
   different resolution than the camera is currently delivering, the tool
   refuses to solve rather than producing a plausible pose that is wrong by the
   scale ratio.

### Checking it worked

Per camera, before moving on:

- **Height above belt** should match a tape measure. If the camera is 1.4 m up
  and the tool says 300 mm, the board's square size is not what the tool thinks
  — wrong board selected, or a scaled print.
- **Reprojection error** under about 1 px.
- **Click two points** on the preview a known distance apart and compare the
  belt millimetres against a tape.

Across cameras, once two or more are done:

- **Build the belt map** (step 5). Footprints should sit where those cameras
  actually look. A footprint in the wrong place means a wrong origin offset.
- **The strongest check:** move the board to a *new* position on the belt, in
  view of two solved cameras, and click the same board corner in each. Both
  should report the same belt coordinates. This is independent — the extrinsics
  came from the earlier placement — and it is the real test of whether the
  cameras share a frame.

### Step 5 — Belt map

Enter the belt's **real width and length**, then **Build belt map**. The map is
assembled from every saved calibration, so it grows as you calibrate more
cameras.

You get a top-down metric view: each camera's rectified image, its **footprint**
outlined, a millimetre grid, and camera positions marked.

The statistics panel reports **measured pairwise overlap** — which finally
answers whether these cameras share a view. Nothing in the design assumes they
do. If they do not overlap, the cameras are still in one frame via the belt, but
a bag is never seen by two at once, so hand-off between them depends on belt
travel rather than a shared view.

> Enter real belt dimensions rather than guessing. The automatic extent sizes
> the canvas to everything the cameras see — including floor and machinery — and
> measured 3055 × 4199 mm for a 700 × 1400 mm belt in testing.

### Step 6 — Save

Write notes that would let you reproduce this: lens, focal length, aperture,
whether focus was locked. **Save calibration** writes
`results/<camera>.json` containing `K`, `D`, `R`, `t`, both homographies, all
error figures, the board used, and the image size.

Repeat steps 1–6 for each camera. The summary table lists everything calibrated
so far.

---

## 5. A full lab session

```
Beforehand
  1. Print both boards, check the 100 mm bar, mount them flat.
  2. Run the tool with the Synthetic source once, end to end.

Phase 1 - intrinsics, per camera (~15 min each, no rig needed)
  3. Set the camera to the SAME resolution you record at.        <- see §6
  4. Start session with that camera's name.
  5. Capture 15-25 varied, TILTED shots; fill the coverage grid.
  6. Calibrate. Read the warnings, not just the RMS.
  7. Save.

Phase 2 - extrinsics, all cameras in one sitting (rig required)
  8. Mount and aim every camera in its final position.
  9. STOP THE CONVEYOR. Lay the board flat on the belt.
 10. For each camera: start session (intrinsics reload automatically),
     solve extrinsics, save. DO NOT MOVE THE BOARD between cameras.
     If a camera cannot see it, use a measured origin offset - see §4b.
 11. Per camera: check height against a tape measure, click a couple of
     points and compare distances.

At the end
 12. Build the belt map with real belt dimensions.
 13. Check the overlap figures and that footprints look sensible.
 14. Move the board somewhere new and confirm two cameras agree on where
     it is - the real test of a shared frame.
```

**Calibrate the RealSense first.** It is the only camera with factory
intrinsics, so it is the one place you can check the tool against an
independent reference. If it agrees there, you can trust it on the Basler and
Lucid cameras, which have nothing to compare against.

---

## 6. Two things that will silently ruin a calibration

**Resolution.** Intrinsics are resolution-specific. The Basler `a2A1920` has a
1920×1200 sensor while the rig records 1280×720, so the camera is cropping or
scaling — and a `K` measured at one resolution does not transfer to the other.
**Calibrate at exactly the resolution you record at.** The saved file records
`image_size` so a mismatch is at least detectable later.

**Print scaling.** Covered in §3, and worth repeating because it is undetectable
downstream: a board printed at 97% produces a calibration that passes every
internal check while every millimetre it reports is 3% wrong.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "no board detected" | Wrong board selected in Setup; too far or too oblique; motion blur; glare. Check the board matches the dropdown. |
| Port 5000 refuses to bind | macOS AirPlay Receiver uses it. `python3 app.py --port 5001`, or disable it in System Settings → General → AirDrop & Handoff. |
| Live camera fails to start | RealSense on macOS needs `sudo` (USB permission). Basler/Lucid need the camera on the same subnet. If a previous run crashed, run `../real_data/utils/reset_realsense.py`. |
| Preview frozen or black | The source stopped returning frames. Restart the session; check the terminal for `[source] read failed`. |
| Calibration refuses to run | Fewer than 5 usable shots. |
| Warning: "board only reached N/16 of the frame" | Genuine. Capture more shots at the edges and corners and recalibrate — do not ignore it. |
| Warning: "no strongly tilted views" | Also genuine, and the more dangerous of the two: `fx` may be badly wrong despite a low RMS. Recapture with tilt. |
| Extrinsics: implausible height | Board square size does not match the printed board, or the wrong board is selected. |
| Belt map: "no calibrations with extrinsics" | Step 4 was not completed for any camera — intrinsics alone are not enough. |
| Belt map mostly empty | Belt dimensions too large, or a camera pointing away from the belt. |

---

## 8. Checking the tool itself

Both run without hardware and check against known ground truth:

```bash
python3 verify_synthetic.py --views 24   # intrinsics + extrinsics + plane
python3 verify_beltmap.py                # 3 cameras agreeing on a shared frame
```

Expect `PIPELINE VERIFIED` and `BELT MAP VERIFIED`. Reference figures from a
good run: focal length within 0.5%, principal point within a few pixels,
end-to-end belt accuracy ~0.2 mm, cross-camera agreement ~0.005 mm.

Regenerate the boards:

```bash
python3 core/board.py                    # both, 300 dpi
python3 core/board.py --preset large --dpi 600
```

---

## 9. Using the results

```python
import json, numpy as np, sys
sys.path.insert(0, "core")
import store, extrinsics as extr

rec = store.load("results/basler_1.json")
a = store.load_arrays(rec)

# A detection at pixel (u, v) -> belt millimetres
belt_mm = extr.image_to_belt(np.array([[u, v]], np.float32),
                             a["K"], a["D"], a["H_image_to_belt"])
```

`image_to_belt` undistorts before applying the homography. Do not apply the
homography to raw pixels — a homography cannot represent lens distortion, and
skipping the undistortion leaves an error that grows toward the frame edges.

**One limitation to carry forward.** The belt map assumes everything lies on the
belt surface. A bag has height, so its visible top projects outward from the
camera: for a 60 mm bag, 10–45 mm of displacement depending on distance from the
camera. This is a *systematic bias*, not noise — it does not average away, and
it grows toward the frame edges. `beltmap.parallax_error_mm()` quantifies it for
a given camera, point and bag height.

---

## 10. Files

```
calibration/
  app.py                 the web server — run this
  USER_MANUAL.md         this document
  README.md              design rationale and verification method
  verify_synthetic.py    ground-truth check, no hardware
  verify_beltmap.py      multi-camera map check, no hardware
  boards/                print-ready PDFs + specs
  core/
    board.py             board definition and printing
    intrinsics.py        detection -> K, D, coverage analysis
    extrinsics.py        pose solve, belt-plane transforms
    beltmap.py           top-down map, footprints, overlap, parallax
    sources.py           synthetic / folder / live cameras
    store.py             results schema
  static/                the web page
  results/               per-camera calibration JSON
```
