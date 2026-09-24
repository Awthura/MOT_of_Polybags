# OVGU AMS Calibration Tool — User Manual

A local web tool for calibrating the conveyor rig's cameras: lens **intrinsics**,
camera **extrinsics**, and a shared metric **belt map** that all cameras project
into.

> This document is the **reference** — what each control does, and what to do
> when something misbehaves. If you are about to actually calibrate, follow
> **[PROCEDURE.md](PROCEDURE.md)** instead: it is the same material in the order
> the work happens, with screenshots, and it covers the routes around the camera
> SDKs that are not installed on this machine. Checking the result afterwards is
> [VALIDATION_PLAN.md](VALIDATION_PLAN.md).

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
| **Device** | Only shown when more than one physical unit of that kind is connected — see below. |
| **Board** | Must match the board you are physically holding. |

Press **Start session**. The live preview appears with a green overlay on
detected corners and a running corner count.

> If a camera source says "SDK present — camera must be connected" but starting
> fails, the SDK is installed and the camera is not reachable. See §7.

**A camera name is a label you type; it is not the same thing as a physical
camera.** This rig has two Basler units, and `record_all_5_cameras_macos.py`
tells them apart from `basler_2` by nothing more than which one
`EnumerateDevices()` happens to return first — there is no serial permanently
pinned to either name. If two Baslers (or two RealSense units) are connected,
the **Device** field appears and lists each by serial number; you must pick
one rather than the tool guessing. With exactly one device of a kind
connected, there is nothing to choose and the field stays hidden.

Once a session starts, the page shows which physical unit it actually
connected to — **"Connected: S/N 24681012 (acA1920-40gc)"** — and that serial
is written into the saved calibration for traceability. If two saved
calibrations ever end up carrying the *same* serial under two different
camera names, the rig status board (§4b) flags it: that is either the same
camera calibrated twice under two names, or two results cross-attributed, and
either way something downstream would use the wrong pose for one of them.

**Route B has no live device to ask** (§1's recommended path when
`pypylon`/`pyrealsense2` are not installed): a folder of extracted frames
carries no hardware identity at all. Typing a serial into the Device field
before starting a folder session still works — it is recorded as
**operator-asserted**, not hardware-confirmed, and shown that way everywhere
it appears. Read it off the recorder's own console output, which already
prints each camera's serial at record time
(`[Basler_1] Pre-initialized: ... (S/N: ...)`).

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

### Shortcut — using the RealSense's own factory intrinsics

The D435 reports its own `fx, fy, ppx, ppy` and distortion model directly from
the sensor — the only camera on this rig with independent ground truth. When a
RealSense session starts, a green banner offers **Use factory intrinsics**:
click it and those numbers are adopted immediately, with no board capture at
all.

The results panel looks the same as a board fit, except there is no RMS or
frame coverage to show — a factory number was never fitted here, so the panel
says so rather than displaying a zero that would read as suspiciously
perfect. Save it exactly like a board result; the saved record marks
`"method": "factory"` under `intrinsics`, and the rig status board shows a
`factory` tag instead of an RMS figure for that camera.

**Do the board-fit comparison at least once first.** Calibrating the
RealSense from a board and checking it *against* the factory numbers (the "vs
known reference" panel above) is the acceptance test for this tool's method —
it is the only camera on the rig where an independent answer exists to check
against. Once that agreement is established, adopting the factory numbers
directly is the fast path for every session after.

Refused if the sensor reports a distortion model this tool cannot represent
(anything other than Brown-Conrady) — a fisheye model pushed through the same
undistortion code would produce a plausible-looking but wrong result, with
nothing downstream able to tell.

### Clearing a saved calibration

Both pages have a way to delete a camera's saved result outright — intrinsics
and any pose together, since `results/<camera>.json` holds both halves and
there is no smaller unit to remove.

- **Intrinsics page (Setup):** type the camera name and press **Clear saved
  calibration**. Works even before starting a session — useful for wiping a
  stale result ahead of time.
- **Rig page (status board):** each row has its own **Clear** button, greyed
  out when there is nothing saved for that camera.

Both ask for confirmation first and cannot be undone. This is the in-app form
of `rm results/<camera>.json` — most useful for clearing the synthetic
rehearsal result before real work (§3.4) or for discarding a bad calibration
to force a clean redo.

### Step 4 — Belt plane (extrinsics) → the rig page

Extrinsics have **their own page**, at `/extrinsics` — the *Extrinsics* link in
the header. They are a different job from intrinsics: intrinsics describe the
lens and can be measured at a desk, extrinsics describe where the camera is
bolted and need the rig in its final state. The two are separate sittings, so
they are separate pages. See §4b for how that page works.

The intrinsics page keeps only a link across to it.

---

## 4b. The rig page — getting every camera into one frame

`http://127.0.0.1:5000/extrinsics`

Intrinsics and extrinsics are different kinds of job and are best done as two
separate phases.

**Intrinsics need no rig.** They describe the lens, not the mounting. You can
calibrate a camera on a bench, at your desk, at any time — as long as the lens,
focus and **resolution** are the ones you will record with, and are not touched
afterwards.

**Extrinsics need the rig, in its final state.** They describe where the camera
is. Any bump to the mount invalidates them.

So the natural order is: **every camera's intrinsics first, then every camera's
extrinsics in one sitting.** The tool supports this — start a session for a
camera that has been calibrated before and its saved intrinsics reload
automatically, so you can solve its pose without recapturing anything.

### A · The status board — what is still outstanding

The top of the page lists every camera in one table, rebuilt from
`results/*.json` each time you open or refresh it. Each is in one of four
states:

| state | meaning | what to do |
|---|---|---|
| `blocked` | no intrinsics saved | calibrate the lens first — bench work, no rig needed |
| `ready` | intrinsics in hand, no pose | **this is the rig work** |
| `provisional` | a tape-measured homography only | re-solve against the board when you can; this route does not correct lens distortion |
| `solved` | full board extrinsics | done |

Under each row, anything still outstanding for that camera is spelled out. The
headline counts it up: *"1 of 5 cameras solved · 1 ready to solve now."* Read it
before walking to the rig — it is the difference between one trip and two.

Cameras with a saved calibration that are not in the plan are listed anyway,
marked `unlisted`, so a result can never be invisible here. A camera whose
intrinsics were adopted from a sensor's factory calibration rather than fitted
here (§4, "Shortcut") shows a `factory` tag in the Intrinsics column instead
of an RMS figure. A rehearsal result from the synthetic camera is flagged
`SYNTHETIC` in red — clear it (below) before real work.

> **Camera names are load-bearing.** A calibration is joined to detections by
> name, so `results/<name>.json` must use the same name the recorder writes its
> video under — `basler_1`, `rgbd_2_color`, and so on. A calibration filed under
> a name nothing else uses is invisible to everything downstream, and nothing
> will report that: it simply never matches.

> **But a name is not a physical camera.** Under the camera name, the board
> also shows the **device serial** the calibration was actually measured
> from — the one piece of identity that comes from the hardware itself rather
> than from what an operator typed. If the same serial turns up under two
> different camera names, both rows say so: that pairing needs checking before
> either result is trusted (§6).

### The anticipated measurements

The shaded columns are inputs, not results. Fill them in **before** the session:

- **Expected height (mm)** — roughly how far above the belt the camera is
  mounted. A tape measure to the lens is plenty accurate for this.
- **Planned offset X, Y (mm)** — where you intend to place the board for this
  camera, relative to the origin placement (X across the belt, Y along travel).
- **Measured** — tick it once the offset is an actual tape measurement rather
  than an intention.

They persist to `results/_plan/rig_plan.json` and survive restarts, so the plan
can be built at your desk days ahead. **Save plan** writes it; the button grows
a dot when there are unsaved edits.

This is what turns each solve into a check. A pose is easy to look at and hard
to judge on its own: 298 mm above the belt reads as a perfectly ordinary number
until you remember the camera is mounted 1.4 m up. After each solve the page
shows **measured against anticipated**:

```
check                measured      anticipated    Δ
height above belt    1380 mm       1400 mm      -1.4%   [ok]
reprojection error   0.412 px      < 1 px               [ok]
origin offset        0, 0 mm       0, 0 mm              [ok]
```

Height within 5% passes, within 15% warns, and beyond that fails with the likely
cause named — nearly always a board mismatch: the wrong preset selected, or a
print scaled by "fit to page". That failure is otherwise **undetectable**, since
it leaves the reprojection error looking perfectly healthy while every
millimetre the camera reports is wrong by the scale factor.

A camera with no expected height recorded reports `no ref` rather than passing.
An unmade comparison should not look like a successful one.

### B2 · The workspace map — using this tool on any rig

*Optional, and purely additive: without it the belt width and length in panel
B define the map exactly as before.*

Nothing about the geometry here is specific to a conveyor. Every camera is
solved against **one shared plane**; that the plane happens to be a belt is a
naming convention, not a constraint. Upload a top-down map of whatever plane
your rig watches and the footprints, coverage and overlap render against it
instead of a blank rectangle.

**Two upload formats, and the difference matters:**

| | georeferencing | when to use |
|---|---|---|
| `.png` / `.jpg` | **by hand** — click the origin, set the scale | a floor plan, a CAD screenshot, a stitched overhead photo |
| `.glb` | **automatic** — from the model itself | a 3-D scan or CAD export of the cell |

A **GLB carries its own units**: glTF fixes lengths at metres and defines an
origin, so the scale is exact and the world origin is the model's origin.
Nothing to click, nothing to measure, no chance of a mis-clicked datum. That
is the one real advantage of the format and the reason it is worth supporting
— if you can export a scan as GLB, prefer it.

A **PNG is a picture until you georeference it.** An image file carries no
units, so two things have to be supplied:

1. **The origin** — click the point that is world (0, 0). Choose something
   you can also identify physically on the rig: a bolt, a frame corner, a
   marked datum. This is the point every camera's extrinsics will be
   expressed relative to.
2. **The scale** — either type mm-per-pixel, or click two points and enter
   the distance between them measured with a tape. The second is preferred
   for the same reason the rest of this tool prefers it: it is a real
   measurement rather than an assumption about what a pixel means.
3. **The +Y direction** *(optional)* — world +X runs right and +Y runs down
   the map. If the plane's natural axis is not aligned that way, click a
   point along it. The rotation is **baked into the stored image once**,
   rather than carried as a term through every later transform, where it
   would be a standing invitation for sign errors.

Until both an origin and a scale exist, the map shows as *not georeferenced*
and **the belt map ignores it** and keeps using the typed belt dimensions.
That is deliberate: rendering a metric overlay against unknown units would
produce confident nonsense, which is worse than an obviously plain rectangle.

The grid spacing adapts to the map's size — 200 mm on a 1.4 m belt, 2 m on a
20 m hall — so the same code is readable at a workbench and across a factory
floor. The world origin is marked with a red cross whenever it falls on the
canvas; seeing it sit where you expect is the fastest check that the whole
georeferencing is right.

> Uploaded maps live in `results/_workspace/` and are **not committed** —
> they are rig-specific and can be large. The directory and `*.glb` are
> gitignored.

### B3 · Solving from map points — the board-free route

*Requires a georeferenced map (B2). An alternative to panel C, not an
addition: either route produces the same kind of result and both are stored
identically.*

Instead of laying a board on the plane for every camera, this needs **one map
and one frame per camera**. Click a feature you can identify in the camera
frame, click the same feature on the map, repeat a handful of times, solve.

That difference matters more than it sounds:

- **No rig visit.** The frame can come from footage recorded months ago.
- **No board.** Nothing has to be printed, placed, or kept still.
- **Nothing to coordinate.** Cameras are solved independently, whenever.

```
1. B2 -> upload and georeference the map (once for the whole rig)
2. Start a session for the camera (any source, including Folder)
3. B3 -> Freeze current frame
4. Click a feature on the frame, then the same feature on the map
   -> repeat, 5-8 times, spread across the area you care about
5. Solve -> read the per-point errors
6. Validate -> click anywhere on the frame, check the answer against the map
7. Save as extrinsics
```

**Four pairs is the minimum and is not enough.** A plane homography has eight
degrees of freedom, so four point pairs determine it exactly — the residuals
come back as zero by construction and tell you nothing. The fifth point is
the first one that measures anything. The panel says so rather than letting a
zero read as perfect.

**Spread matters more than count.** Points clustered in one corner produce a
fit that is excellent there and arbitrarily wrong everywhere else. Five
well-spread points beat fifteen bunched ones.

**A mis-click is caught, named, and excluded.** Points are fitted with RANSAC,
so one bad correspondence is rejected rather than bending the whole solution,
and the table shows each point's error in millimetres with outliers marked.
Measured on synthetic ground truth: with one badly mis-clicked point, this
route lands 0.04 mm RMS where a plain least-squares fit lands 261 mm. You are
told *which* point disagrees instead of deleting points until the number
improves.

**With intrinsics you get a full pose.** The homography is decomposed into a
real `R`/`t`, so the camera gets a position and height, distortion is
corrected, and the result is `solved` — indistinguishable downstream from a
board calibration. Verified against a known camera: position recovered to
0.24 mm, held-out points to 0.05 mm.

**Without intrinsics it degrades honestly.** You still get a usable plane
mapping, but distortion is uncorrected and there is no camera position. It is
recorded as `provisional`, and — because the clicked points are stored in the
calibration file — measuring the lens later and re-solving needs **no
re-clicking**.

### B, C, D · Session parameters, solve, save

**B** sets the origin method (below) and the belt dimensions, which the belt map
on the intrinsics page then picks up.

**C** is the solve: pick a camera — the picker shows each one's state — start
the session, and its saved intrinsics reload. If it has none, the page says so
and does not offer to solve. The origin offset is prefilled from the plan.
After solving, **click anywhere on the preview** to read that point in belt
millimetres; two points a known distance apart, against a tape, tests the whole
chain.

**D** saves. Then pick the next camera — under Method A, without touching the
board.

### The actual problem: one shared origin

Each camera solves its own pose. What makes them a *system* is that they solve
against the **same belt origin**. There are two ways to achieve that.

#### Method A — one board placement, every camera (preferred)

If several cameras can see the same patch of belt:

```
1. STOP THE CONVEYOR.
2. Lay the board flat on the belt, inside the shared view.
3. DO NOT MOVE IT until every camera has been solved.
4. On the rig page, for each camera the board reports as `ready`:
     C -> pick that camera -> Start session
            (its intrinsics reload automatically)
       -> leave both offsets at 0 -> Solve extrinsics
       -> read the measured-vs-anticipated panel
     D -> Save
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
4. Enter those numbers as the origin offset, tick Measured, then solve.
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
  — wrong board selected, or a scaled print. Record the expected height in the
  plan and the page makes this comparison for you; without one it can only show
  you the number.
- **Reprojection error** under about 1 px.
- **Click two points** on the preview a known distance apart and compare the
  belt millimetres against a tape. This one is yours — no recorded expectation
  can substitute for it, because it tests the mapping rather than the pose.

Across cameras, once two or more are done:

- **Build the belt map** (step 5). Footprints should sit where those cameras
  actually look. A footprint in the wrong place means a wrong origin offset.
- **The strongest check:** move the board to a *new* position on the belt, in
  view of two solved cameras, and click the same board corner in each. Both
  should report the same belt coordinates. This is independent — the extrinsics
  came from the earlier placement — and it is the real test of whether the
  cameras share a frame.

---

## 4c. Back on the intrinsics page

### Step 5 — Belt map

Enter the belt's **real width and length**, then **Build belt map**. They are
prefilled from the rig plan, so the figure you enter once in panel B is the one
the map is built with. The map is
The map is assembled from every saved calibration, so it grows as you calibrate
more cameras.

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
whether focus was locked. **Save calibration** writes `results/<camera>.json` —
`K`, `D`, error figures, the board used, and the image size, plus `R`, `t` and
both homographies once the pose has been solved on the rig page.

Both pages write the same file. **A save never discards the other half:** saving
intrinsics at the bench keeps a pose solved earlier at the rig rather than
deleting it as a side effect. If the intrinsics changed, the pose is kept but
flagged stale — it was solved against a different `K` — and the rig page lists
it as outstanding until you re-solve it. Deciding which is right is yours; only
you know whether the camera has been touched.

Repeat for each camera. The summary table lists everything calibrated so far.

---

## 5. A full lab session

```
Beforehand
  1. Print both boards, check the 100 mm bar, mount them flat.
  2. Run the tool with the Synthetic source once, end to end.
  3. Rig page -> fill in the ANTICIPATED measurements and Save plan:
       camera names matching the recorder's stream names,
       expected height per camera, belt width and length,
       method A or B, and any offsets you intend to use.

Phase 1 - intrinsics, per camera (~15 min each, no rig needed)
  4. Set the camera to the SAME resolution you record at.        <- see §6
  5. Start session with that camera's name.
  6. Capture 15-25 varied, TILTED shots; fill the coverage grid.
  7. Calibrate. Read the warnings, not just the RMS.
  8. Save.

Phase 2 - extrinsics, all cameras in one sitting (rig required)
  9. Rig page -> read the status board. Every camera should say `ready`.
     Anything still `blocked` needs Phase 1 first - find that out now,
     not at the conveyor.
 10. Mount and aim every camera in its final position.
 11. STOP THE CONVEYOR. Lay the board flat on the belt.
 12. For each `ready` camera: start session (intrinsics reload
     automatically), solve, read the measured-vs-anticipated panel,
     save. DO NOT MOVE THE BOARD between cameras.
     If a camera cannot see it, use a measured origin offset - see §4b.
 13. Per camera: click a couple of points and compare distances against
     a tape. The height check the page has already made for you.

At the end
 14. Status board should read "N of N cameras solved".
 15. Intrinsics page -> build the belt map with real belt dimensions.
 16. Check the overlap figures and that footprints look sensible.
 17. Move the board somewhere new and confirm two cameras agree on where
     it is - the real test of a shared frame.
```

**Calibrate the RealSense first.** It is the only camera with factory
intrinsics, so it is the one place you can check the tool against an
independent reference. If it agrees there, you can trust it on the Basler and
Lucid cameras, which have nothing to compare against.

---

## 6. Three things that will silently ruin a calibration

**Resolution.** Intrinsics are resolution-specific. The Basler `a2A1920` has a
1920×1200 sensor while the rig records 1280×720, so the camera is cropping or
scaling — and a `K` measured at one resolution does not transfer to the other.
**Calibrate at exactly the resolution you record at.** The saved file records
`image_size` so a mismatch is at least detectable later.

**Print scaling.** Covered in §3, and worth repeating because it is undetectable
downstream: a board printed at 97% produces a calibration that passes every
internal check while every millimetre it reports is 3% wrong.

**The wrong physical camera.** With two Baslers on the rig and nothing but
enumeration order telling `basler_1` from `basler_2` (§4, Step 1), it is
possible to calibrate the *right name* against the *wrong hardware* — pick the
device that answers first without checking, and everything downstream about
that camera is wrong while looking completely normal. The **Device** picker
(shown whenever more than one unit is connected) and the serial recorded into
each saved calibration exist specifically to catch this: if two camera names
end up sharing one serial, the rig status board says so.

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
| Rig page: a camera says `blocked` | It has no saved intrinsics. Calibrate its lens on the intrinsics page — no rig access needed. |
| Rig page: a camera says `provisional` | Its calibration came from the tape-measure route (`metric_map.py`), which has no intrinsics and so does not correct lens distortion. Re-solve it against the board. |
| Height check says `no ref` | No expected height recorded for that camera. Enter one in the status board and solve again. |
| Height check fails by tens of percent | Almost always the board: wrong preset selected, or a print scaled by "fit to page". Check the printed 100 mm bar with a ruler. |
| Pose listed as outstanding after it was solved | It was carried over from an earlier save and solved against different intrinsics, so it is stale. Re-solve it. |
| Edits to the plan vanish | They were not saved — the **Save plan** button shows a dot while edits are pending. |
| Belt map: "no calibrations with extrinsics" | No camera has a solved pose yet — intrinsics alone are not enough. Check the rig page's status board. |
| Belt map mostly empty | Belt dimensions too large, or a camera pointing away from the belt. |
| Lucid preview/recording is grayscale (colour camera) | Two causes, both fixed 2026-07-31: the Lucid source used to treat the raw Bayer stream as grayscale instead of debayering it; and if another process holds GigE Vision controller access, format writes fail read-only. Restart `app.py` to load the fix, and close any other viewer/recorder (ArenaView, a stale session, another machine on the switch). |
| Lucid: "could not start streaming … another process probably holds control" | GigE Vision allows exactly one controlling process per camera. Find and close the other session, then retry. |
| RealSense: source available but no frames ever arrive | macOS needs root for the USB power state: `sudo /opt/anaconda3/envs/ams/bin/python app.py`. If it previously exited uncleanly, run `reset_realsense.py` first. |
| "N Basler cameras connected — specify which one" | Two or more Baslers (or RealSense units) are on the network and the tool refuses to guess between them. Pick one from the **Device** field. |
| "no Basler camera with serial ... found" | The device picked isn't connected right now — reconnect it, or refresh the page to re-enumerate. |
| Rig status board: two cameras flag "also recorded under ..." | The same physical serial is saved under two camera names — either the same camera was calibrated twice under different names, or two results were cross-attributed. Check both before trusting either; **Clear** the wrong one and redo it. |
| Setup shows a device serial marked "(asserted)" | It was typed in by hand rather than read from live hardware — normal for the Folder-source route (§1), which has no camera to ask. Verify it against the recorder's console output from when the footage was captured. |
| "Use factory intrinsics" refuses with a model name | The sensor reported a distortion model this tool cannot represent (not Brown-Conrady — e.g. a fisheye stream). Board-calibrate that camera instead; adopting the factory numbers would silently misapply the wrong undistortion model. |
| Clear button greyed out | Nothing is saved for that camera yet — there is nothing to clear. |
| Workspace map uploaded but the belt map ignores it | It is not georeferenced yet: a PNG needs both an origin **and** a scale. The panel says which is missing. |
| "this GLB uses Draco compression" | Re-export the model without Draco — the decoder is a large dependency this tool deliberately does not carry. |
| "not a GLB file (bad magic…)" | You exported `.gltf` (JSON + separate files) rather than `.glb` (single binary). Re-export as binary. |
| Belt map grid is too dense or too sparse | It adapts to the map extent automatically. If it looks wrong, the scale is probably wrong — check mm-per-pixel against a known distance. |
| World origin cross is not where you expect | The origin click or the scale is off. Re-click the origin; both can be redone at any time without re-uploading. |
| B3: "upload and georeference a workspace map first" | The board-free route reads world coordinates off the map, so B2 must have both an origin and a scale. |
| B3: a point shows ✕ and a large error | RANSAC rejected it as an outlier — one of its two clicks is on the wrong feature. Delete that row and re-click it; the rest of the fit is unaffected. |
| B3: RMS is large but no point is marked ✕ | The errors are spread rather than isolated: usually the points are clustered in one region, or the map's scale is wrong. Add points further apart and re-check mm-per-pixel. |
| B3: residuals are all exactly 0 | You have exactly 4 pairs — they fit perfectly by construction. Add a fifth to get a real error estimate. |
| B3 solve refuses on resolution | The frozen frame is a different size than the saved intrinsics were measured at. Grab the frame at the calibrated resolution. |

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
    plan.py              rig plan, status board, measured-vs-anticipated
  static/
    index.html/app.js    intrinsics page  (/)
    extrinsics.html/.js  rig page         (/extrinsics)
    common.js            helpers shared by both
    ams.css              theme, shared
  results/               per-camera calibration JSON
    _plan/rig_plan.json  the session plan and anticipated measurements
```

`_plan/` is a subdirectory rather than a file in `results/` on purpose: the belt
map and the summary table both enumerate calibrations with
`results/*.json`, and a planning file sitting beside them would be picked up and
parsed as a camera record.
