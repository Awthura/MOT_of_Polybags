# OVGU AMS — camera calibration

Calibrates the 4-camera conveyor rig (2× Basler GigE, 1× Lucid Triton GigE,
1× RealSense D435) for **intrinsics**, **extrinsics**, and a shared **metric
belt-plane** coordinate frame.

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

Workflow: **Setup → Capture → Calibrate → Belt plane → Save.**

- **Capture** shows live corner detection and a frame-coverage grid, and
  `Space` grabs a shot so both hands stay on the board.
- **Calibrate** reports `K`, `D`, per-view error, coverage and tilt — plus a
  direct comparison against known values when the source has them (synthetic
  truth, or RealSense factory intrinsics).
- **Belt plane** solves the camera pose from the board lying on the belt, then
  lets you click anywhere on the preview to read that point in belt
  millimetres. Checking a couple of those against a tape measure is the fastest
  honest test of the whole chain.
- **Save** writes `results/<camera>.json` — `K`, `D`, `R`, `t`, both
  homographies, error figures, the board used, and the image size.

Verify the maths independently at any time:

```bash
python3 verify_synthetic.py --views 24        # renders a known camera, checks recovery
```

### Layout

```
calibration/
  README.md              this file
  app.py                 Flask server + JSON API
  verify_synthetic.py    ground-truth verification, no hardware needed
  boards/                print-ready PDFs + PNGs + machine-readable specs
  core/
    board.py             ChArUco definition, print-ready output, layout checks
    intrinsics.py        detection -> K, D, coverage analysis, warnings
    extrinsics.py        solvePnP -> R, t; belt-plane homography and transforms
    sources.py           synthetic / folder / RealSense / Basler / Lucid
    store.py             results schema (named `store`, not `io` — that would
                         shadow the stdlib module on sys.path)
  static/                AMS-themed single page (no build step)
  results/               per-camera calibration JSON
```

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
