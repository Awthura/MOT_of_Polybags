"""
Synthetic ground-truth verification — OVGU AMS calibration tool.

Calibration is easy to get confidently wrong: OpenCV will happily return a
low reprojection error alongside a badly incorrect `K`. The only way to know the
pipeline is sound is to run it against a camera whose true parameters are known,
which is what this does — no hardware, no lab time.

Method: take a known `K`, `D` and a set of known board poses; render what that
camera would actually see by warping the printed board pattern through the
projection (distortion included); then run the real detection and calibration
path over those images and compare what comes back against what went in.

This exercises the genuine pipeline — `CharucoDetector.detectBoard` →
`matchImagePoints` → `calibrateCamera` → `solvePnP` — not a shortcut using the
known correspondences, so detection failures are caught too.

Usage:
    python3 verify_synthetic.py
    python3 verify_synthetic.py --views 20 --noise 0.3 --save-dir /tmp/synth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from board import PRESETS, render_board, MM_PER_INCH          # noqa: E402
import intrinsics as intr                                      # noqa: E402
import extrinsics as extr                                      # noqa: E402


def render_view(board_img: np.ndarray, px_per_mm: float,
                K: np.ndarray, D: np.ndarray,
                rvec: np.ndarray, tvec: np.ndarray,
                image_size: tuple[int, int]) -> np.ndarray:
    """Render what a camera with (K, D, rvec, tvec) would see of the board.

    Works backwards from the output image so every destination pixel is filled:
    output pixel -> undistort -> ideal pixel -> H^-1 -> board mm -> board pixel,
    then a single remap. Doing it forwards would leave holes.
    """
    w, h = image_size
    H = extr.homography_from_pose(K, rvec, tvec)
    H_inv = np.linalg.inv(H)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=-1).reshape(-1, 1, 2)

    # Distorted pixel -> ideal (pinhole) pixel.
    ideal = cv2.undistortPoints(pts, K, D, P=K)
    board_mm = cv2.perspectiveTransform(ideal, H_inv).reshape(-1, 2)

    map_x = (board_mm[:, 0] * px_per_mm).reshape(h, w).astype(np.float32)
    map_y = (board_mm[:, 1] * px_per_mm).reshape(h, w).astype(np.float32)

    return cv2.remap(board_img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=128)


def make_poses(n: int, spec, distance_mm: float, rng) -> list:
    """Varied poses: tilted, off-centre, at a range of distances.

    Deliberately includes strong tilt — a face-on-only set is the degenerate
    case this whole exercise is meant to detect.
    """
    poses = []
    cx_mm, cy_mm = spec.width_mm / 2, spec.height_mm / 2
    for i in range(n):
        # Spread tilt across both axes and both signs.
        rx = np.radians(rng.uniform(-35, 35))
        ry = np.radians(rng.uniform(-35, 35))
        rz = np.radians(rng.uniform(-25, 25))
        Rx = cv2.Rodrigues(np.array([rx, 0, 0]))[0]
        Ry = cv2.Rodrigues(np.array([0, ry, 0]))[0]
        Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
        R = Rz @ Ry @ Rx
        rvec = cv2.Rodrigues(R)[0]

        d = distance_mm * rng.uniform(0.75, 1.45)
        # Push the board out to the frame edges, not just around the centre.
        # Distortion is strongest at the periphery and the principal point is
        # only well determined once the target has visited both sides, so a
        # timid spread leaves cx and the outer distortion terms unconstrained.
        # This mirrors the capture practice the README asks for; generating
        # easier poses here would flatter the pipeline rather than test it.
        if i < 4:
            # Deliberately drive the first few into the corners.
            sx, sy = [(-1, -1), (1, -1), (-1, 1), (1, 1)][i]
            tx, ty = sx * 0.46 * d, sy * 0.30 * d
        else:
            tx = rng.uniform(-0.46, 0.46) * d
            ty = rng.uniform(-0.30, 0.30) * d
        t = np.array([[tx - cx_mm], [ty - cy_mm], [d]], np.float64)
        poses.append((rvec, t))
    return poses


def run(preset: str, n_views: int, noise_px: float, seed: int,
        save_dir: Path | None) -> bool:
    rng = np.random.default_rng(seed)
    spec = PRESETS[preset]
    image_size = (1280, 720)
    w, h = image_size

    # ── Ground truth ────────────────────────────────────────────────────────
    K_true = np.array([[1100.0, 0.0, 645.0],
                       [0.0, 1095.0, 355.0],
                       [0.0, 0.0, 1.0]])
    D_true = np.array([-0.28, 0.12, 0.001, -0.0008, 0.0])

    px_per_mm = 8.0
    board_img = np.array(render_board(spec, dpi=int(px_per_mm * MM_PER_INCH)))

    print("=" * 78)
    print(f"SYNTHETIC VERIFICATION — board {spec.name}, {n_views} views, "
          f"{noise_px:g}px noise")
    print("=" * 78)
    print(f"  ground truth  fx={K_true[0,0]:.1f} fy={K_true[1,1]:.1f} "
          f"cx={K_true[0,2]:.1f} cy={K_true[1,2]:.1f}")
    print(f"                D={D_true.round(4).tolist()}")

    # ── Render + detect ─────────────────────────────────────────────────────
    poses = make_poses(n_views, spec, distance_mm=900.0, rng=rng)
    dets, kept_poses = [], []
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, (rvec, tvec) in enumerate(poses):
        img = render_view(board_img, px_per_mm, K_true, D_true, rvec, tvec, image_size)
        if noise_px > 0:
            img = np.clip(img.astype(np.float32)
                          + rng.normal(0, noise_px * 8, img.shape), 0, 255).astype(np.uint8)
        if save_dir:
            cv2.imwrite(str(save_dir / f"view_{i:02d}.png"), img)
        d = intr.detect_charuco(img, spec, source=f"view_{i:02d}")
        if d is not None:
            dets.append(d)
            kept_poses.append((rvec, tvec))

    print(f"  detected in {len(dets)}/{n_views} rendered views")
    if len(dets) < intr.MIN_VIEWS:
        print("  FAIL: too few detections to calibrate")
        return False

    # ── Calibrate ───────────────────────────────────────────────────────────
    res = intr.calibrate(dets, camera=f"synthetic-{preset}")
    print()
    print(intr.format_report(res, expected_fx=K_true[0, 0]))

    # ── Compare against truth ───────────────────────────────────────────────
    print()
    print("=" * 78)
    print("RECOVERED vs TRUE")
    print("=" * 78)
    checks = []
    for label, got, want, tol, unit in (
        ("fx", res.fx, K_true[0, 0], 0.02, "rel"),
        ("fy", res.fy, K_true[1, 1], 0.02, "rel"),
        ("cx", res.cx, K_true[0, 2], 15.0, "abs"),
        ("cy", res.cy, K_true[1, 2], 15.0, "abs"),
    ):
        if unit == "rel":
            dev = abs(got - want) / want
            ok = dev <= tol
            print(f"  {label:3} {got:9.2f}  true {want:9.2f}   "
                  f"{dev*100:5.2f}%  (tol {tol*100:.0f}%)  {'PASS' if ok else 'FAIL'}")
        else:
            dev = abs(got - want)
            ok = dev <= tol
            print(f"  {label:3} {got:9.2f}  true {want:9.2f}   "
                  f"{dev:5.2f}px (tol {tol:.0f}px)  {'PASS' if ok else 'FAIL'}")
        checks.append(ok)

    # Distortion is compared as a FIELD, not coefficient by coefficient.
    # k1/k2/k3 are strongly correlated, so the solver freely trades one against
    # another: a recovered k3 of -0.37 against a true 0.0 can still describe the
    # same optical behaviour. What is physically meaningful is how far the two
    # models disagree about where a pixel actually lands.
    gy, gx = np.mgrid[0:h:20, 0:w:20]
    grid = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32).reshape(-1, 1, 2)
    u_true = cv2.undistortPoints(grid, K_true, D_true, P=K_true).reshape(-1, 2)
    u_got = cv2.undistortPoints(grid, res.K, res.D, P=res.K).reshape(-1, 2)
    disagree = np.linalg.norm(u_true - u_got, axis=1)

    # Split inside/outside the region the board actually visited. A distortion
    # polynomial is only constrained where it saw data; beyond that it
    # extrapolates, and two models agreeing on the observed area can diverge
    # violently in an unobserved corner. Judging the fit on extrapolated
    # territory would condemn a perfectly good calibration — and, worse,
    # trusting it there would silently corrupt real measurements.
    obs = np.vstack([d.image_points.reshape(-1, 2) for d in dets])
    x0, y0 = obs.min(axis=0)
    x1, y1 = obs.max(axis=0)
    g = grid.reshape(-1, 2)
    inside = ((g[:, 0] >= x0) & (g[:, 0] <= x1) &
              (g[:, 1] >= y0) & (g[:, 1] <= y1))

    # Judge on p95 rather than max: the maximum is set by a single pixel at the
    # extreme corner of the observed region, where the polynomial is least
    # constrained, so it says more about that one sample than about the fit.
    d_in = disagree[inside]
    in_p95 = float(np.percentile(d_in, 95)) if d_in.size else float("nan")
    in_max = float(d_in.max()) if d_in.size else float("nan")
    out_max = float(disagree[~inside].max()) if (~inside).any() else 0.0
    field_ok = in_p95 < 2.0
    print(f"  distortion field disagreement:")
    print(f"    inside observed region  p95 {in_p95:6.3f} px  (tol 2.0)  "
          f"{'PASS' if field_ok else 'FAIL'}"
          f"   max {in_max:.3f} px   [x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}]")
    print(f"    outside (extrapolated)  max {out_max:6.3f} px  — not a defect; "
          f"the model is unconstrained there")
    print(f"    raw coeffs differ because k1/k2/k3 are correlated and not "
          f"individually identifiable:")
    print(f"      recovered D = {np.asarray(res.D).ravel().round(4).tolist()}")
    print(f"      true      D = {D_true.round(4).tolist()}")
    checks.append(field_ok)

    # ── Extrinsics + plane round-trip ───────────────────────────────────────
    print()
    print("=" * 78)
    print("EXTRINSICS & BELT-PLANE ROUND-TRIP")
    print("=" * 78)
    rvec_t, tvec_t = kept_poses[0]
    d0 = dets[0]
    ext = extr.calibrate_extrinsics(d0.object_points, d0.image_points,
                                    res.K, res.D, camera="synthetic")

    C_true = (-cv2.Rodrigues(rvec_t)[0].T @ tvec_t).ravel()
    C_got = ext.camera_position_mm
    pos_err = float(np.linalg.norm(C_got - C_true))
    pos_ok = pos_err < 15.0
    print(f"  camera position  got  [{C_got[0]:8.1f} {C_got[1]:8.1f} {C_got[2]:8.1f}] mm")
    print(f"                   true [{C_true[0]:8.1f} {C_true[1]:8.1f} {C_true[2]:8.1f}] mm")
    print(f"                   error {pos_err:.2f} mm (tol 15)  {'PASS' if pos_ok else 'FAIL'}")
    print(f"  extrinsic reprojection error {ext.reproj_error_px:.3f} px")
    checks.append(pos_ok)

    # belt -> image -> belt must return the original millimetres.
    belt_pts = np.array([[0, 0], [spec.width_mm, 0],
                         [spec.width_mm, spec.height_mm], [0, spec.height_mm],
                         [spec.width_mm / 2, spec.height_mm / 2]], np.float32)
    img_pts = extr.belt_to_image(belt_pts, res.K, res.D, ext.rvec, ext.tvec)
    back = extr.image_to_belt(img_pts, res.K, res.D, ext.H_image_to_belt)
    rt_err = float(np.abs(back - belt_pts).max())
    rt_ok = rt_err < 1.0
    print(f"  belt->image->belt max error {rt_err:.4f} mm (tol 1.0)  "
          f"{'PASS' if rt_ok else 'FAIL'}")
    checks.append(rt_ok)

    scale = extr.mm_per_pixel(ext.H_image_to_belt, res.K, res.D, (w / 2, h / 2))
    print(f"  scale at image centre {scale:.3f} mm/px")

    # ── The number that actually matters ────────────────────────────────────
    # Everything above is a component check. This is the end-to-end question:
    # take real belt points, see where the TRUE camera puts them in the image,
    # then map those pixels back to the belt using ONLY the recovered
    # calibration. The residual is the positional error the MOT stage inherits.
    # A round-trip through the recovered calibration alone cannot show this —
    # it is self-consistent by construction and would report ~0 even if the
    # calibration were badly wrong.
    print()
    print("=" * 78)
    print("END-TO-END BELT ACCURACY  (true camera -> recovered calibration)")
    print("=" * 78)
    gxm, gym = np.meshgrid(np.linspace(0, spec.width_mm, 5),
                           np.linspace(0, spec.height_mm, 5))
    truth_mm = np.stack([gxm.ravel(), gym.ravel()], -1).astype(np.float32)
    obj = np.column_stack([truth_mm, np.zeros(len(truth_mm), np.float32)])
    seen_px, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec_t, tvec_t,
                                   K_true, D_true)
    est_mm = extr.image_to_belt(seen_px.reshape(-1, 2), res.K, res.D,
                                ext.H_image_to_belt)
    errs = np.linalg.norm(est_mm - truth_mm, axis=1)
    e2e_ok = float(errs.max()) < 5.0
    print(f"  {len(errs)} points across the board")
    print(f"  mean {errs.mean():.3f} mm   median {np.median(errs):.3f} mm   "
          f"max {errs.max():.3f} mm  (tol 5.0)  {'PASS' if e2e_ok else 'FAIL'}")
    print(f"  -> this is the positional error the MOT stage would inherit")
    checks.append(e2e_ok)

    print()
    print("=" * 78)
    ok = all(checks)
    print(f"RESULT: {sum(checks)}/{len(checks)} checks passed — "
          f"{'PIPELINE VERIFIED' if ok else 'PIPELINE HAS A PROBLEM'}")
    print("=" * 78)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="small")
    ap.add_argument("--views", type=int, default=18)
    ap.add_argument("--noise", type=float, default=0.25,
                    help="Sensor noise level; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None,
                    help="Write the rendered views here for inspection")
    args = ap.parse_args()

    ok = run(args.preset, args.views, args.noise, args.seed,
             Path(args.save_dir) if args.save_dir else None)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
