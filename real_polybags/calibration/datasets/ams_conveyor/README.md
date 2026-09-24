# AMS conveyor — CalibrationHub dataset (iteration_3)

Extrinsics dataset for the OVGU AMS polybag conveyor. Upload these into CalibrationHub
and run the Homography → Validation flow. Everything here is metric (meters).

## Files
| File | What it is |
|---|---|
| `ams_conveyor.glb` | The world map: the conveyor extracted from a room scan, **leveled** (belt flat), **oriented** (+X along belt), **cropped** to the conveyor, origin on the left checkerboard. Textured. |
| `basler_1.png`, `basler_2.png`, `lucid.png`, `rgbd_1_color.png` | One camera frame each, **already undistorted** (lens-rectified). 1280×720. |
| `intrinsics_<cam>.json` | Per-camera intrinsics in CalibrationHub format (`K`, `K_new`, `D`). |

## ⚠️ Read this first — enable "fisheye" for every camera
The camera frames are **already lens-corrected** (we rectified them with each camera's
pinhole calibration, because our Baslers/Lucid are pinhole, not fisheye — CalibrationHub's
built-in undistortion is fisheye-only and would *re-distort* an already-rectified image).

So in the **Homography tab, tick the "fisheye" checkbox for each camera.** That box means
"skip lens-distortion correction, use the image coordinates as-is" — which is exactly right
here. The provided `K`/`K_new` are the rectified camera matrix (`D` is zeros and unused once
fisheye is on). If you leave it off, correspondences will be wrong by tens of pixels at the
frame edges.

## World frame
- **Origin (0,0,0)** = centre of the **8×11 checkerboard**, which sits under **basler_1**
  (camera height ≈ **94 cm** above it — a good live sanity check on basler_1's solve).
- **+X** = along the belt (direction of travel). **+Z** = across the belt (width ≈ **58 cm**).
  Height is +Y.
- Scale is metric, straight from the Scaniverse scan (LiDAR, ~<1%). Cross-checks: belt
  width **58 cm**, camera-beam spacing **91 cm**. Verify in the UI if you want certainty.

### Origin is the left board by assumption — adjust in Map Config if needed
The scan has three targets on the belt. The origin is baked at the **left** plain
checkerboard (assumed to be basler_1's 8×11). If basler_1's board is actually a different
one, set the origin in the **Map Config** tab to the correct board using these offsets from
the current origin:

| Board | Offset from current origin (X along belt, Z across) |
|---|---|
| left plain checker  (current origin) | (0.00, 0.00) m |
| middle ChArUco | (+0.40, −0.07) m |
| right plain checker | (+1.07, −0.24) m |

## Per-camera notes
- **basler_1** — looks straight down (~4° off nadir) at the 8×11 origin board. Cleanest camera. Intrinsics solid.
- **basler_2** — the 8×11 board is **clipped** in its view (only partly visible). You cannot use the full board; click correspondences on **belt edges / rails / frame features** that also appear on the map. Intrinsics solid.
- **lucid** — sees the origin board. Its intrinsics are a **centre-anchored salvage** (the intrinsics-calibration capture never covered the top of the frame), so its rectification — and any pose from it — is usable but less certain than the Baslers.
- **rgbd_1_color** — sees a **different** board (not the origin one). Its lens is near-rectilinear; the board-fit distortion was unreliable but rectification is near-identity. The map is what ties it into the shared frame.
- **rgbd_1_depth** — excluded (no visual board; depth registers to color separately).

## Workflow in CalibrationHub
1. **Sources**: upload `ams_conveyor.glb` as the map; add the 4 `*.png` frames.
2. **Map Config**: origin is on the left board; adjust to basler_1's board if needed (table above).
3. **Homography** (per camera): load `intrinsics_<cam>.json`, **tick "fisheye"**, click ≥4
   map↔image correspondences (board corners where a full board is visible; belt/rail features
   for basler_2), then Calculate.
4. **Validation** (per camera): click test points, check they land where expected on the map;
   review the KPIs (avg error, coverage, confidence). For basler_1, expect the solved camera
   height ≈ 94 cm.

## Notes / deferred
- **Dashboard map** (conveyor only, boards removed) is a follow-up: the boards are baked into
  the scan's texture, so clean removal needs inpainting or a fresh boards-off scan.
- Regenerate the map with `real_polybags/calibration/export_conveyor_glb.py`
  (`--origin-x/--origin-z` override the origin in the leveled frame).
