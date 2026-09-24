# Intrinsics undistortion check — 4 cameras

Before/after undistortion for every camera that currently has intrinsics, rendered
from a frame **out of the exact set each camera was calibrated on** (small A4
ChArUco, 1280×720 session recordings, 2026-07-31). Undistortion uses each camera's
saved `K`/`D` with `getOptimalNewCameraMatrix(alpha=1)`, so the *after* image keeps
every source pixel — the **curved black frame border is the distortion being undone**
and is the thing to read.

Full-resolution pairs: `basler_1_beforeafter.png`, `basler_2_beforeafter.png`,
`lucid_beforeafter.png`, `rgbd_1_color_beforeafter.png` (this folder).

## Parameters used

| Camera | Res | fx | fy | cx, cy (centre = 640, 360) | k1, k2, k3 | RMS px | Verdict |
|---|---|---|---|---|---|---|---|
| basler_1 | 1280×720 | 1354 | 1354 | 659, 382 | −0.35, 0.54, −0.87 | 0.64 | ✅ trustworthy |
| basler_2 | 1280×720 | 2460 | 2461 | 756, 388 | −0.23, 0.77, 0.51 | 0.56 | ✅ trustworthy |
| lucid | 1280×720 | 2903 | 2888 | 640, 360 *(anchored)* | −0.99, 3.54, −39.2 | 0.98 | ⚠️ centre-anchored salvage |
| rgbd_1_color | 1280×720 | 3001 | 3113 | 687, 268 | **2.68, −61.7, 597** | 0.40 | ⚠️ coeffs unreliable |

## What each pair shows

**basler_1 — clean.** The frame border in *after* bows out gently and the board's
lines straighten. A moderate, well-behaved barrel correction (~51° FOV, k1 = −0.35).
At the frame corner the net radial factor is ~0.92, i.e. a real but modest pull.
Use as-is.

**basler_2 — clean, mild.** Correction is subtle (corner factor ~0.99); the lens is
close to rectilinear. Board edges straighten slightly, no warping. Use as-is.

**lucid — centre-anchored salvage.** The *free* fit of this capture was **broken**: the
board never reached the top quarter of the frame (0 detections there across all 123
usable frames), so the solve drifted the principal point to (1046, 567) — 406 px right,
207 px below centre — and undistortion collapsed into a smeared spiral. That drift is the
classic fx↔cx degeneracy of a partial-frame capture, not a real optical centre.

Because a re-record isn't available, this was salvaged by **holding the principal point
at the image centre** (`fix_principal_point=True`). This is justified by the pixel
geometry: the frame has no letterbox/pillarbox, its content bbox is centred at
(639.5, 359.5), and fx≈fy (square pixels) — all consistent with a **centred** crop or
downscale of the sensor, whose optical axis stays at the middle. With cx,cy anchored, the
*after* image is now **sane** (`lucid_beforeafter.png`): a normal, gently-corrected frame.

Caveat, kept honest in the saved file's warning: fx (2903) and the distortion terms are
only **partially** constrained (RMS rose 0.46→0.98 when the crutch of a free principal
point was removed, and the empty top row means top-edge distortion is extrapolated).
**Good for coarse undistortion and rough geometry; not for precision metric work.** The
real fix remains a re-record with the board swept into all four corners.

**Reproduce:** in the calibration app, tick *"Fix principal point at centre
(partial-frame salvage)"* before *Calibrate intrinsics*; or call
`intrinsics.calibrate(dets, fix_principal_point=True)`.

**rgbd_1_color — near-rectilinear lens, unreliable coefficients.** *Before* and *after*
are almost identical: the straight rig rails stay straight both times, and the corner
factor is only ~1.07. So the camera's **true** distortion is small. But the fitted
coefficients (k2 = −62, k3 = +597) are physically absurd — they nearly cancel *inside*
the frame, which is why the picture looks fine, but they do not represent the lens and
will not generalise (and fx swings ~30% with the distortion model). The safe reading:
this stream needs almost no undistortion; prefer the RealSense **factory intrinsics**
over this board fit. Fine for a quick look, not for metric work.

## Bottom line

- **basler_1, basler_2**: intrinsics good, undistortion correct — ready for extrinsics.
- **lucid**: free fit was invalid; **centre-anchored salvage** now gives sane undistortion
  and is usable for coarse work. Re-record for precision.
- **rgbd_1_color**: geometry (fx-aside) usable, distortion model junk but low-impact;
  switch to factory intrinsics when available.
