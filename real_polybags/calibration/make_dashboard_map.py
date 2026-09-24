#!/usr/bin/env python3
"""
Dashboard conveyor map — a clean, empty belt to spawn/showcase polybags on.

A flat textured .glb sized to the real belt (580 mm wide, arbitrary length),
origin at the CENTRE, in the same frame the calibration uses: world X across the
belt, world Y along the conveyor (glTF X = across, glTF Z = along, Y = up = 0).
Unlike the calibration map this carries no scan clutter or boards — just a
synthetic belt (dark surface + blue side rails) so spawned bags read clearly.

    python3 make_dashboard_map.py --out datasets/dashboard/conveyor_dashboard.glb
    python3 make_dashboard_map.py --length-mm 1400 --width-mm 580
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2

# reuse the stdlib glTF writer from the calibration exporter
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exp", str(Path(__file__).resolve().parent / "export_conveyor_glb.py"))
_exp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_exp)
write_glb = _exp.write_glb


def sample_scan_colors(scan_path):
    """Median belt colour and rail (blue) colour from the real scan, so the
    dashboard resembles the actual conveyor. Falls back to sane defaults."""
    belt, rail = (58, 62, 66), (150, 96, 40)                   # BGR fallbacks
    try:
        js, b = _exp.load_glb(scan_path)
        pr = js['meshes'][0]['primitives'][0]
        V = _exp.accessor(js, b, pr['attributes']['POSITION'])
        UV = _exp.accessor(js, b, pr['attributes']['TEXCOORD_0'])
        F = _exp.accessor(js, b, pr['indices']).astype(int).reshape(-1, 3)
        tex = cv2.imdecode(np.frombuffer(_exp.bv_bytes(js, b, js['images'][0]['bufferView']),
                                         np.uint8), cv2.IMREAD_COLOR)
        th, tw = tex.shape[:2]
        C = V[F].mean(1); fuv = UV[F].mean(1)
        tx = np.clip((fuv[:, 0]*tw).astype(int), 0, tw-1); ty = np.clip((fuv[:, 1]*th).astype(int), 0, th-1)
        col = tex[ty, tx].astype(float)
        Ymed = np.median(C[:, 1]); onbelt = np.abs(C[:, 1]-Ymed) < 0.05
        lum = col @ [0.114, 0.587, 0.299]
        beltm = onbelt & (lum > 60) & (lum < 190)              # belt, not boards/shadow
        if beltm.sum() > 50: belt = tuple(int(v) for v in np.median(col[beltm], 0))
        blue = col[:, 0] - 0.5*(col[:, 1]+col[:, 2]) > 25      # blue-dominant = rails
        if blue.sum() > 30: rail = tuple(int(v) for v in np.median(col[blue], 0))
    except Exception as e:
        print("  (scan colour sampling skipped:", e, ")")
    return belt, rail


def belt_texture(width_mm, length_mm, mmpp, origin_off_mm, belt_col, rail_col):
    """Top-down belt image. Columns = across (X), rows = along (Z), row 0 = the
    release/near end. Origin is `origin_off_mm` in from that end."""
    W = max(2, int(round(width_mm / mmpp)))
    H = max(2, int(round(length_mm / mmpp)))
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), belt_col, np.uint8)
    # subtle rubber-belt grain
    noise = rng.normal(0, 6, (H, W, 1))
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    # lateral belt seams every ~250 mm (darker lines across the width)
    for y in range(0, H, max(1, int(250 / mmpp))):
        cv2.line(img, (0, y), (W-1, y), tuple(int(c*0.7) for c in belt_col), 1)
    # faint 100 mm reference grid
    g = tuple(min(255, int(c*1.25)+6) for c in belt_col)
    for x in range(0, W, max(1, int(100/mmpp))): cv2.line(img, (x, 0), (x, H-1), g, 1)
    for y in range(0, H, max(1, int(100/mmpp))): cv2.line(img, (0, y), (W-1, y), g, 1)
    # blue side rails down the two long edges
    rail = max(4, int(18 / mmpp))
    img[:, :rail] = rail_col; img[:, -rail:] = rail_col
    # origin (offset toward the release end) + a +Y travel arrow
    cx = W // 2; oy = int(round(origin_off_mm / mmpp))
    cv2.line(img, (cx, oy-14), (cx, oy+14), (210, 210, 210), 1)
    cv2.line(img, (cx-14, oy), (cx+14, oy), (210, 210, 210), 1)
    cv2.circle(img, (cx, oy), 5, (80, 200, 90), -1)                    # origin dot
    cv2.arrowedLine(img, (cx, oy+8), (cx, min(H-8, oy+130)), (90, 180, 230), 3, tipLength=0.16)  # +Y travel
    return img, W, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/dashboard/conveyor_dashboard.glb")
    ap.add_argument("--width-mm", type=float, default=580.0)
    ap.add_argument("--length-mm", type=float, default=2000.0)
    ap.add_argument("--origin-offset-mm", type=float, default=400.0,
                    help="origin distance in from the release (near) end; bags enter here "
                         "and travel +Y")
    ap.add_argument("--mmpp", type=float, default=1.0)
    ap.add_argument("--scan", default="maps/Scan_conveyor_2.glb",
                    help="scan to sample belt/rail colour from (for realism)")
    args = ap.parse_args()

    belt_col, rail_col = sample_scan_colors(args.scan)
    print(f"  belt colour BGR {belt_col}, rail colour BGR {rail_col}")
    img, W, H = belt_texture(args.width_mm, args.length_mm, args.mmpp,
                             args.origin_offset_mm, belt_col, rail_col)
    ok, buf = cv2.imencode(".png", img)

    # flat quad: X across (± width/2), Z along, Y = 0. Origin sits `origin_offset`
    # in from the release (near, min-Z) end, so most of the belt is ahead (+Y).
    hw = args.width_mm / 2000.0
    z_near = -args.origin_offset_mm / 1000.0
    z_far = (args.length_mm - args.origin_offset_mm) / 1000.0
    V = np.array([[-hw, 0, z_near], [hw, 0, z_near], [hw, 0, z_far], [-hw, 0, z_far]], np.float32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    F = np.array([[0, 2, 1], [0, 3, 2]], np.uint32)                # normal +Y (faces up)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    write_glb(str(out), V, UV, F, buf.tobytes(), "image/png")
    cv2.imwrite(str(out.with_suffix(".png")), img)                 # sidecar preview
    print(f"wrote {out}")
    print(f"  belt {args.width_mm:.0f} x {args.length_mm:.0f} mm")
    print(f"  +X across (± {args.width_mm/2:.0f} mm); +Y along, from {z_near*1000:+.0f} "
          f"(release end) to {z_far*1000:+.0f} mm; origin {args.origin_offset_mm:.0f} mm in")
    print(f"  texture {W}x{H}px @ {args.mmpp} mm/px")


if __name__ == "__main__":
    main()
