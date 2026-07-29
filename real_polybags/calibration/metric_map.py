"""
Metric conveyor map from a tape measure — OVGU AMS calibration tool.

Turns the relative conveyor map into a **true top-down map in millimetres**,
using four clicked points and two measured distances. No calibration board, no
intrinsics, no lab visit beyond holding a tape against the belt.

The idea: the belt is a plane, and you mark a rectangle on it whose real
dimensions you have measured. Four image points and their four known
belt coordinates determine a homography exactly — which both removes the
oblique perspective and fixes the scale.

**Measuring once per cluster is enough.** `build_conveyor_map.py` already
computed how the cameras within a cluster relate to each other, so a metric
solve for one camera propagates to the rest by composition. On this rig that
means two measurements total, one per cluster.

What you need to measure:

    P0 ---------------- P1        W = across the belt (its width)
    |                    |        L = along the belt, between two features
    |                    |            you can identify in the image
    P3 ---------------- P2

Pick the corners on things you can actually see and measure between — the belt
edges give you W for free, and roller centres, seams or a taped mark give you L.
The rectangle does **not** need to be large, but bigger is better: an error of a
few millimetres in your tape reading matters proportionally less over a longer
span, and points spread across the frame constrain the homography better than
points bunched in the middle.

Accuracy expectations, stated honestly. This recovers the plane mapping but not
lens distortion, which a homography cannot represent. Near the frame centre the
result is good; toward the edges it carries whatever radial distortion the lens
has — typically a few millimetres on these lenses, growing outward. Calibrating
intrinsics later and re-solving removes that; this is a solid first pass, not a
replacement.

Usage:
    # click the four corners interactively
    python3 metric_map.py --session 20260727_140907 --camera basler_2 \\
        --width-mm 700 --length-mm 900 --out maps/

    # or supply them directly, if you already know the pixels
    python3 metric_map.py --session 20260727_140907 --camera basler_2 \\
        --width-mm 700 --length-mm 900 \\
        --points 310,120 980,140 1010,650 280,630 --out maps/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import analyse_overlap as ov

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
import store as calstore                                        # noqa: E402


def click_four(img, title="Click 4 corners: across-top L, across-top R, "
                          "bottom R, bottom L  (u=undo, Enter=accept)"):
    """Collect four clicked points, in order, with undo."""
    pts: list[tuple[int, int]] = []
    disp = img.copy()
    scale = min(1.0, 1400 / img.shape[1])
    view = cv2.resize(disp, None, fx=scale, fy=scale) if scale < 1 else disp.copy()

    def redraw():
        v = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
        for i, p in enumerate(pts):
            q = (int(p[0] * scale), int(p[1] * scale))
            cv2.circle(v, q, 6, (0, 220, 90), -1)
            cv2.putText(v, f"P{i}", (q[0] + 8, q[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 90), 2)
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(v, (int(pts[i][0] * scale), int(pts[i][1] * scale)),
                         (int(pts[i+1][0] * scale), int(pts[i+1][1] * scale)),
                         (0, 220, 90), 2)
        if len(pts) == 4:
            cv2.line(v, (int(pts[3][0] * scale), int(pts[3][1] * scale)),
                     (int(pts[0][0] * scale), int(pts[0][1] * scale)),
                     (0, 220, 90), 2)
        cv2.imshow(title, v)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10) and len(pts) == 4:
            break
        if k == ord('u') and pts:
            pts.pop()
            redraw()
        if k == 27:
            pts.clear()
            break
    cv2.destroyAllWindows()
    return pts


def solve_metric(image_pts, width_mm, length_mm):
    """Homography from image pixels to belt millimetres.

    Point order is across-top-left, across-top-right, bottom-right,
    bottom-left, so X runs across the belt and Y along it — matching the
    convention the rest of the tool uses for the belt frame.
    """
    src = np.array(image_pts, np.float32).reshape(-1, 1, 2)
    dst = np.array([[0, 0], [width_mm, 0],
                    [width_mm, length_mm], [0, length_mm]], np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise ValueError("could not solve a homography from those four points — "
                         "are any three of them collinear?")
    return H


def check_rectangle(image_pts) -> list[str]:
    """Sanity checks on the clicked quad, before it silently poisons the scale."""
    p = np.array(image_pts, np.float32)
    warns = []

    # Order matters, and a convex-hull test cannot see it: the hull is the same
    # set of points whichever sequence they arrive in, so clicking the corners
    # in a bow-tie order passes it while producing a folded, silently wrong map.
    # Compare the polygon's area IN THE GIVEN ORDER against the hull's. They
    # agree only when the traversal is a simple quadrilateral.
    hull = cv2.convexHull(p)
    hull_area = float(cv2.contourArea(hull))
    poly_area = float(cv2.contourArea(p))
    if hull_area <= 0:
        warns.append("the four points are degenerate — three or more may be collinear")
    elif poly_area < 0.9 * hull_area:
        warns.append("these points traverse a self-intersecting shape, so they "
                     "were clicked out of order — go around the rectangle "
                     "(across-top-left, across-top-right, bottom-right, "
                     "bottom-left), not diagonally")
    if poly_area < 5000:
        warns.append(f"the quad covers only {poly_area:.0f} px^2 — a small "
                     f"reference region amplifies both click and tape error "
                     f"across the whole map")
    return warns


def render_metric_map(img, H_img_to_mm, width_mm, length_mm,
                      mm_per_px=2.0, margin_mm=50.0):
    """Top-down metric render of one camera view."""
    x0, y0 = -margin_mm, -margin_mm
    W = int((width_mm + 2 * margin_mm) / mm_per_px)
    Hgt = int((length_mm + 2 * margin_mm) / mm_per_px)

    # Map destination pixels back to source: mm -> image, via the inverse.
    ys, xs = np.mgrid[0:Hgt, 0:W].astype(np.float32)
    mm = np.stack([xs.ravel() * mm_per_px + x0,
                   ys.ravel() * mm_per_px + y0], -1).reshape(-1, 1, 2)
    src = cv2.perspectiveTransform(mm, np.linalg.inv(H_img_to_mm)).reshape(-1, 2)
    mx = src[:, 0].reshape(Hgt, W).astype(np.float32)
    my = src[:, 1].reshape(Hgt, W).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(20, 20, 20))

    # Metric grid every 100 mm.
    g = (90, 90, 90)
    for x in range(0, int(width_mm) + 1, 100):
        px = int((x - x0) / mm_per_px)
        cv2.line(out, (px, 0), (px, Hgt), g, 1)
        cv2.putText(out, str(x), (px + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, g, 1)
    for y in range(0, int(length_mm) + 1, 100):
        py = int((y - y0) / mm_per_px)
        cv2.line(out, (0, py), (W, py), g, 1)
        cv2.putText(out, str(y), (3, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, g, 1)
    # The measured rectangle itself.
    r = np.array([[0, 0], [width_mm, 0], [width_mm, length_mm], [0, length_mm]])
    rp = ((r - [x0, y0]) / mm_per_px).astype(np.int32)
    cv2.polylines(out, [rp], True, (0, 220, 90), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="../real_data/raw_recordings")
    ap.add_argument("--session", required=True)
    ap.add_argument("--camera", required=True, help="which camera to measure on")
    ap.add_argument("--width-mm", type=float, required=True,
                    help="measured distance across the belt (P0->P1)")
    ap.add_argument("--length-mm", type=float, required=True,
                    help="measured distance along the belt (P1->P2)")
    ap.add_argument("--points", nargs=4, default=None,
                    metavar="X,Y", help="four image points instead of clicking")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--mm-per-px", type=float, default=2.0)
    ap.add_argument("--out", default="maps")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent / "results"),
                    help="also write standard per-camera calibration records here")
    args = ap.parse_args()

    frames = ov.frames_from_session(Path(args.dir), args.session, args.t)
    if args.camera not in frames:
        print(f"ERROR: '{args.camera}' not in this session. Available: {sorted(frames)}")
        return 1
    img = frames[args.camera]

    if args.points:
        pts = [tuple(float(v) for v in p.split(",")) for p in args.points]
    else:
        print("Click the four corners of your measured rectangle, clockwise from")
        print("the top-left (across-top-left, across-top-right, bottom-right,")
        print("bottom-left). 'u' undoes, Enter accepts, Esc cancels.")
        pts = click_four(img)
        if len(pts) != 4:
            print("cancelled")
            return 1
        print(f"  points: {[(round(x), round(y)) for x, y in pts]}")

    for w in check_rectangle(pts):
        print(f"  WARNING: {w}")

    H = solve_metric(pts, args.width_mm, args.length_mm)

    # Residual check: push the clicked points through and see how far they land
    # from the millimetres they are supposed to be. With exactly four points the
    # fit is exact by construction, so this only catches gross input errors -
    # it is NOT an accuracy estimate.
    got = cv2.perspectiveTransform(
        np.array(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    want = np.array([[0, 0], [args.width_mm, 0],
                     [args.width_mm, args.length_mm], [0, args.length_mm]])
    resid = float(np.abs(got - want).max())

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    top = render_metric_map(img, H, args.width_mm, args.length_mm, args.mm_per_px)
    cv2.imwrite(str(outdir / f"metric_{args.camera}.png"), top)

    # Propagate to the rest of this camera's cluster, if a map manifest exists.
    propagated = {}
    manifest_path = outdir / f"conveyor_map_{args.session}.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        for cl in man.get("clusters", []):
            place = cl.get("placement") or {}
            if args.camera not in place:
                continue
            H_this_to_ref = np.array(place[args.camera]["H_to_reference"])
            for other, p in place.items():
                if other == args.camera:
                    continue
                H_other_to_ref = np.array(p["H_to_reference"])
                # other -> reference -> this camera -> millimetres
                H_other_to_mm = H @ np.linalg.inv(H_this_to_ref) @ H_other_to_ref
                propagated[other] = H_other_to_mm.tolist()
                cv2.imwrite(str(outdir / f"metric_{other}.png"),
                            render_metric_map(frames[other], H_other_to_mm,
                                              args.width_mm, args.length_mm,
                                              args.mm_per_px))

    rec = {
        "session": args.session,
        "measured_on": args.camera,
        "width_mm": args.width_mm,
        "length_mm": args.length_mm,
        "image_points": [list(p) for p in pts],
        "H_image_to_mm": H.tolist(),
        "corner_residual_mm": resid,
        "propagated_H_image_to_mm": propagated,
        "caveat": ("homography only — lens distortion is not modelled, so "
                   "accuracy degrades toward the frame edges until intrinsics "
                   "are calibrated and this is re-solved"),
    }
    (outdir / f"metric_{args.session}.json").write_text(json.dumps(rec, indent=2) + "\n")

    # Also write into the standard per-camera results, so the belt map, the web
    # tool and the MOT stage all read ONE format regardless of how the mapping
    # was obtained. `method` records the provenance, because the two routes are
    # not equivalent and a consumer has to be able to tell them apart.
    results_dir = Path(args.results_dir)
    written = []
    h, w = img.shape[:2]
    calstore.save(calstore.build_plane_record(
        args.camera, H, (w, h), args.width_mm, args.length_mm,
        [list(p) for p in pts], session=args.session,
        notes="tape-measured plane homography"), results_dir)
    written.append(args.camera)
    for other, Ho in propagated.items():
        oh, ow = frames[other].shape[:2]
        calstore.save(calstore.build_plane_record(
            other, np.array(Ho), (ow, oh), args.width_mm, args.length_mm,
            image_points=None, session=args.session,
            notes="scale propagated via inter-camera homography",
            propagated_from=args.camera), results_dir)
        written.append(other)

    print()
    print("=" * 74)
    print(f"METRIC MAP — measured on {args.camera}")
    print("=" * 74)
    print(f"  rectangle {args.width_mm:.0f} x {args.length_mm:.0f} mm")
    print(f"  corner residual {resid:.3f} mm (exact by construction with 4 points)")
    for name in [args.camera] + sorted(propagated):
        print(f"  wrote metric_{name}.png")
    if propagated:
        print(f"  scale propagated to {len(propagated)} other camera(s) in this")
        print(f"  cluster via the inter-camera homographies — no second measurement")
    else:
        print("  no cluster manifest found; run build_conveyor_map.py first to")
        print("  propagate this scale to the other camera(s) automatically")
    print(f"  results written for: {', '.join(written)}")
    print(f"    -> {results_dir}")
    print("  These feed the belt map and the MOT stage directly. Re-solving")
    print("  after intrinsics are calibrated upgrades them in place — the")
    print("  clicked points are stored, so no second tape measurement.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
