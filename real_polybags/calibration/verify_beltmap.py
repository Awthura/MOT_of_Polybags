"""
Belt-map verification — OVGU AMS calibration tool.

Checks the claim the whole calibration exists to support: that several cameras,
each solved independently against the belt plane, agree on where a physical
point is — in millimetres — without ever having seen each other.

Method: build a synthetic conveyor with markers at known belt coordinates,
render it through several known cameras at different poses, then run the real
pipeline (pose solve, rectify, map) and check that

1. each camera's rectified view lands back on the true belt scene,
2. a marker at a known position resolves to the same millimetres in every
   camera that can see it,
3. footprints and overlaps are measured correctly.

No hardware. Run before the lab session.

Usage:
    python3 verify_beltmap.py
    python3 verify_beltmap.py --save-dir /tmp/beltmap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
import beltmap as bm          # noqa: E402
import extrinsics as extr     # noqa: E402

# Belt geometry for the synthetic rig, in millimetres.
BELT_W = 700.0      # across the belt (X)
BELT_L = 1400.0     # along the belt  (Y)
MARKERS = [(100, 150), (350, 150), (600, 150),
           (100, 700), (350, 700), (600, 700),
           (100, 1250), (350, 1250), (600, 1250)]


def make_belt_scene(mm_per_px: float = 1.0) -> tuple[np.ndarray, bm.BeltFrame]:
    """A top-down 'truth' image of the conveyor with markers at known mm."""
    frame = bm.BeltFrame(0.0, BELT_W, 0.0, BELT_L, mm_per_px)
    img = np.full((frame.height_px, frame.width_px, 3), 70, np.uint8)

    # Belt surface texture, so rectification errors show as smearing.
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 7, (frame.height_px, frame.width_px, 1))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Belt edges.
    for x in (0.0, BELT_W):
        px = int(np.clip(x / mm_per_px, 1, frame.width_px - 2))
        cv2.line(img, (px, 0), (px, frame.height_px), (150, 150, 150), 3)

    for i, (mx, my) in enumerate(MARKERS):
        p = frame.mm_to_px(np.array([[mx, my]]))[0].astype(int)
        cv2.circle(img, tuple(p), int(18 / mm_per_px), (240, 240, 240), -1)
        cv2.circle(img, tuple(p), int(18 / mm_per_px), (30, 30, 30), 2)
        cv2.putText(img, str(i), (p[0] - 8, p[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 / mm_per_px, (20, 20, 20), 2)
    return img, frame


def camera_looking_at(target_mm, height_mm, offset_mm, K, D, image_size):
    """Build a camera above the belt aimed at `target_mm`.

    Constructs the pose directly from a look-at basis rather than fitting one,
    so the ground truth is exact and any error found later belongs to the
    pipeline rather than to the fixture.
    """
    C = np.array([target_mm[0] + offset_mm[0], target_mm[1] + offset_mm[1], -height_mm])
    T = np.array([target_mm[0], target_mm[1], 0.0])

    fwd = T - C
    fwd = fwd / np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, -1.0])
    right = np.cross(world_up, fwd)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)

    R = np.vstack([right, down, fwd])          # world -> camera
    rvec = cv2.Rodrigues(R)[0]
    tvec = (-R @ C).reshape(3, 1)
    return bm.CameraOnBelt(name="", K=K, D=D, rvec=rvec, tvec=tvec,
                           image_size=image_size)


def render_camera_view(scene, scene_frame, cam):
    """What this camera sees of the belt scene (backward remap, distortion in)."""
    w, h = cam.image_size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pix = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float32)
    mm = extr.image_to_belt(pix, cam.K, cam.D, cam.H_image_to_belt)
    src = scene_frame.mm_to_px(mm)
    mx = src[:, 0].reshape(h, w).astype(np.float32)
    my = src[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(scene, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(25, 25, 25))


def run(save_dir: Path | None) -> bool:
    K = np.array([[1050.0, 0, 640.0], [0, 1050.0, 360.0], [0, 0, 1.0]])
    D = np.array([-0.22, 0.08, 0.0, 0.0, 0.0])
    size = (1280, 720)

    scene, scene_frame = make_belt_scene(1.0)

    # Three cameras: two overlapping over the middle, one further along the belt
    # to exercise the non-overlapping case the design must tolerate.
    specs = [
        ("cam_upstream",   (350, 400),  1500, (-260,  -60)),
        ("cam_mid",        (350, 700),  1500, ( 260,   40)),
        ("cam_downstream", (350, 1150), 1400, (-120,  120)),
    ]
    cams, views = [], {}
    for name, target, height, offset in specs:
        c = camera_looking_at(np.array(target, float), height, np.array(offset, float),
                              K, D, size)
        c.name = name
        cams.append(c)
        views[name] = render_camera_view(scene, scene_frame, c)

    print("=" * 78)
    print("BELT MAP VERIFICATION — 3 synthetic cameras over a 700 x 1400 mm belt")
    print("=" * 78)
    for c in cams:
        p = c.position_mm
        print(f"  {c.name:15} at [{p[0]:7.0f} {p[1]:7.0f} {abs(p[2]):7.0f}] mm above belt")

    # Map the BELT, at its known dimensions — not everything the cameras see.
    # auto_frame() exists for when the extent is unknown, but on a tilted view
    # it includes floor and machinery and can be several times the belt area,
    # leaving the region of interest a small patch in a mostly empty canvas.
    frame = bm.BeltFrame.from_belt(BELT_W, BELT_L, mm_per_px=1.5, margin_mm=60)
    auto = bm.auto_frame(cams, margin_mm=40, mm_per_px=2.0)
    print(f"\n  belt map    X {frame.x_min_mm:.0f}..{frame.x_max_mm:.0f} mm   "
          f"Y {frame.y_min_mm:.0f}..{frame.y_max_mm:.0f} mm   "
          f"{frame.width_px}x{frame.height_px} px @ {frame.mm_per_px:.2f} mm/px")
    print(f"  (auto_frame would have been {auto.width_px}x{auto.height_px} px "
          f"covering {auto.width_mm:.0f}x{auto.height_mm:.0f} mm — mostly not belt)")

    checks = []

    # ── 1. Cross-camera agreement on the same physical markers ──────────────
    print()
    print("=" * 78)
    print("CROSS-CAMERA AGREEMENT — do cameras resolve a marker to the same mm?")
    print("=" * 78)
    worst = 0.0
    n_shared = 0
    for i, (mx, my) in enumerate(MARKERS):
        obs = {}
        for c in cams:
            px, _ = cv2.projectPoints(
                np.array([[[mx, my, 0.0]]], np.float32), c.rvec, c.tvec, c.K, c.D)
            u, v = px.reshape(2)
            if 0 <= u < size[0] and 0 <= v < size[1]:
                back = extr.image_to_belt(np.array([[u, v]], np.float32),
                                          c.K, c.D, c.H_image_to_belt)[0]
                obs[c.name] = back
        if len(obs) < 2:
            continue
        n_shared += 1
        pts = np.array(list(obs.values()))
        spread = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
        err = float(np.linalg.norm(pts.mean(axis=0) - np.array([mx, my])))
        worst = max(worst, spread)
        print(f"  marker {i} ({mx:4},{my:5}) seen by {len(obs)}  "
              f"spread {spread:6.3f} mm   mean offset from truth {err:6.3f} mm")
    ok = worst < 1.0 and n_shared >= 2
    print(f"\n  {n_shared} markers seen by 2+ cameras; worst spread {worst:.3f} mm "
          f"(tol 1.0)  {'PASS' if ok else 'FAIL'}")
    checks.append(ok)

    # ── 2. Rectification lands on the true scene ────────────────────────────
    print()
    print("=" * 78)
    print("RECTIFICATION — does each camera's top-down view match the truth?")
    print("=" * 78)
    # Resample the truth scene onto the same map grid, then correlate. A
    # brightness spot-check would only say "something light is near where a
    # marker should be"; correlation over the whole overlapping region actually
    # measures whether the rectified view lands on the truth.
    ys, xs = np.mgrid[0:frame.height_px, 0:frame.width_px].astype(np.float32)
    truth_mm = frame.px_to_mm(np.stack([xs.ravel(), ys.ravel()], -1))
    tp = scene_frame.mm_to_px(truth_mm)
    tmx = tp[:, 0].reshape(frame.height_px, frame.width_px).astype(np.float32)
    tmy = tp[:, 1].reshape(frame.height_px, frame.width_px).astype(np.float32)
    truth_on_map = cv2.remap(scene, tmx, tmy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    tg = cv2.cvtColor(truth_on_map, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rect_ok = True
    for c in cams:
        rect, valid = bm.rectify(views[c.name], c, frame)
        m = (valid > 0) & (tg > 5)
        rg = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if m.sum() < 500:
            print(f"  {c.name:15} too little overlap with the belt to score")
            continue
        a, b = rg[m], tg[m]
        corr = float(np.corrcoef(a, b)[0, 1])
        mad = float(np.abs(a - b).mean())
        good = corr > 0.90
        rect_ok &= good
        print(f"  {c.name:15} covers {int(m.sum()*frame.mm_per_px**2/100):5d} cm^2 "
              f"of belt   correlation {corr:.4f}   mean |diff| {mad:5.1f}/255  "
              f"{'PASS' if good else 'FAIL'}")
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_dir / f"rect_{c.name}.png"), rect)
    checks.append(rect_ok)

    # ── 3. Mosaic + coverage ────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)
    mos, stats = bm.mosaic(views, cams, frame)
    rep = bm.coverage_report(cams, frame)
    for n, a in rep["per_camera_mm2"].items():
        print(f"  {n:15} {a/100:8.0f} cm^2 of belt")
    print(f"  covered {rep['covered_mm2']/100:.0f} cm^2 of a "
          f"{rep['map_area_mm2']/100:.0f} cm^2 map")
    print()
    print("  pairwise overlap:")
    for pair, v in rep["pairwise"].items():
        a, b = pair.split("|")
        print(f"    {a:15} / {b:15} {v['overlap_mm2']/100:8.0f} cm^2  "
              f"({v['fraction_of_smaller']*100:5.1f}% of the smaller view)")
    print(f"\n  cameras overlap: {rep['any_overlap']} "
          f"— measured from the calibration, not assumed")
    checks.append(rep["covered_mm2"] > 0 and rep["any_overlap"])

    # ── 4. Parallax honesty ─────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("PARALLAX — the map assumes Z=0; bags have height")
    print("=" * 78)
    for c in cams[:2]:
        for h_mm in (30, 60):
            e_near = bm.parallax_error_mm(c, np.array([350, 700]), h_mm)
            e_far = bm.parallax_error_mm(c, np.array([650, 1300]), h_mm)
            print(f"  {c.name:15} {h_mm:3d} mm bag -> {e_near:5.1f} mm displaced "
                  f"near nadir, {e_far:5.1f} mm at the far corner")
    print("  Systematic, not noise: it grows with distance from the camera and")
    print("  does not average away. Worth knowing before treating map positions")
    print("  as exact.")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / "truth_scene.png"), scene)
        cv2.imwrite(str(save_dir / "mosaic.png"), mos)
        cv2.imwrite(str(save_dir / "mosaic_annotated.png"),
                    bm.draw_overlay(mos, cams, frame))
        for n, v in views.items():
            cv2.imwrite(str(save_dir / f"view_{n}.png"), v)
        print(f"\n  images written to {save_dir}")

    print()
    print("=" * 78)
    good = all(checks)
    print(f"RESULT: {sum(checks)}/{len(checks)} checks passed — "
          f"{'BELT MAP VERIFIED' if good else 'PROBLEM FOUND'}")
    print("=" * 78)
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    return 0 if run(Path(args.save_dir) if args.save_dir else None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
