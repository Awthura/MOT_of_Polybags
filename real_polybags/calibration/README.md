# OVGU AMS — camera calibration

Calibrates the conveyor rig's cameras (2× Basler GigE, 1× Lucid Triton GigE,
RealSense D435 — five recorded streams) for **intrinsics**, **extrinsics**, and a
shared **metric belt-plane** coordinate frame.

| document | when to read it |
|---|---|
| **[PROCEDURE.md](PROCEDURE.md)** | **doing a calibration** — step by step, with screenshots, in the order the work happens |
| [USER_MANUAL.md](USER_MANUAL.md) | reference for individual controls, and troubleshooting |
| [VALIDATION_PLAN.md](VALIDATION_PLAN.md) | establishing a finished calibration is correct, not merely complete |
| this file | why it is built this way, and how the method itself is verified |

> **No camera on this rig has been calibrated yet.** The only file in `results/`
> is a synthetic rehearsal artefact, and two of the three camera SDKs are not
> installed on this machine — see [PROCEDURE.md §0](PROCEDURE.md) for what that
> means and [§1](PROCEDURE.md) for the route around it.

## Why this exists

Cross-camera association currently has nothing geometric to stand on. The
association code carried over from the synthetic track
(`synthetic_polybags/tracking/associate_cameras.py`) is deliberately
calibration-free and leans on two cues:

1. a **colour-class gate** (7 classes in the synthetic set), and
2. **x-order rank** plus temporal overlap, assuming synchronized cameras.

Neither survives contact with the real rig. Real polybags are a **single merged
class**, so the class gate does not exist at all; and `measure_sync.py` showed
the cameras are not frame-synced — skew sits at the frame-rate floor. Calibration
replaces both cues with one shared metric frame: every camera maps its
detections onto the same belt coordinates, and association becomes a
nearest-neighbour question in millimetres.

---

## 1. Boards — what to print

Two boards, already generated in [`boards/`](boards/):

| file | paper | board | squares | square | dictionary | corners |
|---|---|---|---|---|---|---|
| `charuco_AMS-small_A4.pdf` | A4 | 175 × 200 mm | 7 × 8 | 25 mm | `DICT_4X4_50` | 42 |
| `charuco_AMS-large_A3.pdf` | A3 | 252 × 324 mm | 7 × 9 | 36 mm | `DICT_5X5_100` | 48 |

**Two sizes, because one will not do.** `basler_1` is an extreme close-up — the
belt fills its frame — while `lucid`, `basler_2` and the RealSense see the whole
belt width. A board big enough to be detected reliably by the wide cameras does
not fit inside `basler_1`'s field of view; one small enough for `basler_1` is too
coarse for the others.

The two use **different ArUco dictionaries** so they can never be confused, even
if both appear in one shot. Verified: each board's detector finds 0 corners on
the other.

### Printing instructions (send these with the PDF)

- **Print at 100% / "actual size". Do not use "fit to page" or "shrink to fit".**
  Scaling silently shrinks the board by a few percent, and every distance derived
  from it is then wrong by that factor — with nothing downstream able to detect
  it. The calibration will look perfectly healthy and the millimetres will be
  wrong.
- **Check the printed 100 mm bar with a ruler before use.** Each page prints one.
  If it does not measure exactly 100 mm, the print was scaled — reprint.
- **Matte paper**, not glossy. The rig's lighting already causes specular
  blowout on the bags; a shiny board will lose corners under the same lights.
- **Mount flat and rigid** — foamboard or stiff card. A bowed sheet is a
  systematic error that no amount of averaging removes.
- **Keep the footer.** It carries the full spec (squares, square size, marker
  size, dictionary). A printed board whose parameters are unknown is scrap.

To regenerate, or to produce other sizes:

```bash
python3 core/board.py                        # both, 300 dpi
python3 core/board.py --preset large --dpi 600
```

The generator refuses to emit a page whose board plus footer would overflow the
sheet, rather than quietly running the ruler off the bottom edge.

---

## 2. Capture procedure

### 2a. Intrinsics — one camera at a time, board held in the air

Per camera, ~20–30 shots. **Pose variety is what determines quality**, and it is
the usual reason calibration silently fails: twenty frontal shots at the same
distance give a confident, wrong answer.

- **Tilt the board.** Roughly 20–45° away from square-on, in different
  directions. Tilt is what separates focal length from distance — without it
  they are ambiguous and `fx` is unreliable.
- **Cover the whole frame**, corners included. Lens distortion is strongest at
  the edges, so a board only ever seen in the centre leaves the distortion
  coefficients unconstrained.
- **Vary the distance** — near, mid, far.
- Keep the board **sharp and still**. Motion blur moves corners; a blurred shot
  is worse than no shot.

The board does **not** need to be aligned or square to the camera here. The
opposite: squareness is the failure mode.

### 2b. Extrinsics — board flat on the belt

This is where alignment matters. Lay the board **flat on the belt surface**,
which defines the world Z = 0 plane, and record its position so every camera is
solved against the same origin and axes.

If two cameras can see the board at the same time, capture it — that measures
their overlap directly. It is not required: cameras are tied together through
the shared belt frame, not through seeing each other's views.

---

## 3. The tool

```bash
pip install flask          # only external dependency
python3 app.py             # -> http://127.0.0.1:5000
```

Pick **Synthetic camera** as the source to exercise the whole workflow with no
hardware attached. That is not just a demo: the synthetic camera's true `K` is
known, so the tool reports the recovered values *against the truth* and the
result can be verified rather than merely looked at. Worth doing once before
the lab session, so you arrive knowing the software works.

**Two pages, because there are two sittings.** Intrinsics describe the lens and
can be measured at a desk; extrinsics describe where the camera is bolted and
need the rig in its final state. Putting both on one page invites the wrong
order, so they are separate:

`/` — **intrinsics.** Setup → Capture → Calibrate → Belt map → Save.

- **Capture** shows live corner detection and a frame-coverage grid, and
  `Space` grabs a shot so both hands stay on the board.
- **Calibrate** reports `K`, `D`, per-view error, coverage and tilt — plus a
  direct comparison against known values when the source has them (synthetic
  truth, or RealSense factory intrinsics).
- **Save** writes `results/<camera>.json` — `K`, `D`, error figures, the board
  used, and the image size, joined by `R`, `t` and both homographies once the
  pose is solved.

`/extrinsics` — **the rig page.** A status board over every camera, then the
solve.

- **The status board** is rebuilt from `results/*.json` on every load and
  answers one question: what is still outstanding. Each camera reads `blocked`
  (no intrinsics — bench work first), `ready` (this is the rig work),
  `provisional` (a tape homography, no distortion correction) or `solved`, with
  the specific next action spelled out underneath. Read before walking to the
  rig, it is the difference between one trip and two.
- **The anticipated measurements** — expected height per camera, planned origin
  offsets, belt dimensions — are recorded up front and persisted to
  `results/_plan/rig_plan.json`. This is what makes a solve checkable. A pose is
  easy to look at and hard to judge alone: 298 mm above the belt reads as an
  ordinary number until it is set beside the 1400 mm the camera is mounted at.
  Each solve is reported as measured *versus* anticipated, which catches the one
  failure that is otherwise undetectable — a board printed at "fit to page",
  which leaves the reprojection error healthy while every millimetre is wrong by
  the scale factor. A camera with nothing recorded reports `no ref` rather than
  passing: an unmade comparison must not look like a successful one.
- **The solve** reloads that camera's saved intrinsics, prefills the offset from
  the plan, and then lets you click anywhere on the preview to read that point
  in belt millimetres. Checking a couple of those against a tape measure is the
  fastest honest test of the whole chain.

Both pages write the same per-camera file, and a save never discards the other
half: saving intrinsics at the bench keeps a pose solved earlier at the rig,
flagging it stale rather than deleting it if the intrinsics changed under it.

Verify the maths independently at any time:

```bash
python3 verify_synthetic.py --views 24        # renders a known camera, checks recovery
```

### Layout

```
calibration/
  README.md              this file
  app.py                 Flask server + JSON API
  verify_synthetic.py    intrinsics/extrinsics ground-truth check, no hardware
  verify_beltmap.py      multi-camera belt-map check, no hardware
  extract_board_frames.py  recorded video -> calibration frames, selected for
                         pose variety (the route around the missing SDKs)
  boards/                print-ready PDFs + PNGs + machine-readable specs
  core/
    board.py             ChArUco definition, print-ready output, layout checks
    intrinsics.py        detection -> K, D, coverage analysis, warnings
    extrinsics.py        solvePnP -> R, t; belt-plane homography and transforms
    beltmap.py           top-down conveyor map, footprints, overlap, parallax
    sources.py           synthetic / folder / RealSense / Basler / Lucid
    store.py             results schema (named `store`, not `io` — that would
                         shadow the stdlib module on sys.path)
    plan.py              rig plan, status board, measured-vs-anticipated checks
  static/                AMS-themed pages (no build step)
    index.html/app.js      intrinsics  (/)
    extrinsics.html/.js    rig page    (/extrinsics)
    common.js              shared helpers
    ams.css                shared theme
  results/               per-camera calibration JSON
    _plan/rig_plan.json  session plan and anticipated measurements — one
                         directory down, out of reach of the `results/*.json`
                         glob that enumerates cameras
```

### The belt map

The payoff. Once each camera has intrinsics **and** a belt-plane pose, step 5
builds a top-down metric map of the conveyor from every saved calibration:

- each camera's view rectified to bird's-eye and composited,
- each camera's **footprint** — the belt area it actually covers — drawn as a
  polygon in millimetres,
- **pairwise overlap measured**, which finally answers whether these cameras
  share a view. Nothing in the design assumes they do; a bag at (X, Y) mm is the
  same bag whichever camera saw it, because all of them are solved against one
  belt frame rather than against each other.

Give it the belt's real dimensions. `auto_frame()` exists for when the extent is
unknown, but on a tilted view it sizes the canvas to everything the cameras see
— floor, framing, machinery — which measured several times the belt area in
testing and leaves the region of interest a small patch in an empty canvas.

```bash
python3 verify_beltmap.py --save-dir /tmp/beltmap   # 3 synthetic cameras, checked
```

**One honest limitation.** The map assumes everything lies on Z = 0. A bag has
height, so its top surface projects outward from the camera's nadir — for a
60 mm bag in the verification rig, 10–45 mm of displacement depending on
distance from the camera. That is a *systematic bias*, not noise: it does not
average away, and it grows toward the frame edges. `parallax_error_mm()`
quantifies it for a given camera, point and bag height, so it can be accounted
for rather than discovered later.

### Resolution matters

Intrinsics are **resolution-specific**. The Basler a2A1920 has a 1920x1200
sensor while this rig records 1280x720, so the camera is cropping or scaling —
either way a `K` measured at one resolution does not transfer to the other.
Calibrate at exactly the resolution you record at; the saved record stores
`image_size` so a mismatch is at least detectable later.

---

## 4. How this gets verified

Calibration is easy to get confidently wrong, so the plan checks it against
things that are independently known rather than against itself:

1. **Synthetic ground truth.** Render a board with a known virtual camera and
   confirm the pipeline recovers the `K`, `R`, `t` it was generated with.
2. **RealSense factory intrinsics.** The D435 is factory-calibrated —
   `video_stream_profile.get_intrinsics()` gives `fx, fy, ppx, ppy` and a
   distortion model. It is the only camera on the rig with independent ground
   truth, so calibrating it with our own tool and comparing is the real
   acceptance test. The Basler and Lucid cameras ship **uncalibrated** (their
   intrinsics depend on whatever lens is fitted), so they have nothing to check
   against — which is exactly why the RealSense result has to be trusted first.
3. **Focal-length sanity band.** `fx ≈ f_mm / pixel_size_mm` from the lens
   marking and sensor spec. Catches a degenerate pose set, which can produce a
   low reprojection error alongside a badly wrong `K`.
4. **Reprojection error**, reported per view rather than as one mean, so bad
   poses are identifiable instead of averaged away.
5. **Cross-camera agreement** in millimetres on the same physical point.
6. **Belt width** against a tape measurement, and the RealSense's true metric
   depth as a second, independent scale check.
