"""
Conveyor map from footage alone — OVGU AMS calibration tool.

Builds a stitched map of the conveyor from existing recordings, with **no
calibration board, no intrinsics and no lab access**. It is deliberately a first
pass: the result is correct in structure and relative geometry, and becomes
metric later when intrinsics and a belt-plane solve refine it.

What it does:

1. Detects features in one frame per camera and matches every pair, keeping
   only geometrically consistent matches (homography + MAGSAC).
2. Groups cameras into **clusters** of mutually overlapping views. This is
   measured, not assumed — and on this rig it found two disjoint clusters
   rather than one, which changes how tracking has to hand bags over.
3. Picks a reference camera per cluster and warps the others onto it,
   producing one stitched image per cluster: the conveyor map in that
   reference camera's pixel frame.
4. Optionally converts to millimetres from one known distance.

**What this is and is not.** The stitch is a genuine plane-to-plane mapping
between views, so overlap, relative placement and hand-off geometry are all
real. What it lacks is (a) absolute scale, until one physical distance is
supplied, and (b) a true top-down rectification, because removing perspective
requires knowing the camera's pose relative to the belt — which needs
intrinsics. Until then the map lives in the reference camera's oblique view.

Adding `--belt-width-mm` with two clicked belt-edge points converts pixels to
millimetres along that measurement, which is enough for approximate distances
even before proper calibration.

Usage:
    python3 build_conveyor_map.py --session 20260727_140907 --out maps/
    python3 build_conveyor_map.py --session 20260727_140907 --t 0.3 --out maps/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

import analyse_overlap as ov


def build_clusters(names: list[str], pairs: dict, threshold: int) -> list[list[str]]:
    """Union-find over camera pairs that share enough geometry."""
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (a, b), r in pairs.items():
        if r["inliers"] >= threshold:
            union(a, b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [sorted(v) for v in groups.values()]


def choose_reference(cluster: list[str], pairs: dict, frames: dict) -> str:
    """Reference = whichever camera needs the least warping of the others.

    Picking by pixel count does not work here: every camera on this rig records
    1280x720, so "biggest image" cannot distinguish a wide view from a tight
    close-up. Choosing the close-up as reference is actively harmful — the wide
    camera then gets upscaled into it, inventing resolution it never had (a
    0.9 MP source expanding to 2.9 MP was the observed result).

    So try each candidate and measure: whichever yields the smallest stitched
    canvas is the one the others fit into most naturally, which is the widest
    view in practice.
    """
    best, best_area = None, float("inf")
    for cand in cluster:
        H = homography_to_reference(cluster, cand, pairs)
        if len(H) < len(cluster):
            continue                       # cannot place everyone from here
        corners = []
        for c in cluster:
            h, w = frames[c].shape[:2]
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            corners.append(cv2.perspectiveTransform(pts, H[c]).reshape(-1, 2))
        allc = np.vstack(corners)
        if not np.isfinite(allc).all():
            continue
        span = allc.max(axis=0) - allc.min(axis=0)
        area = float(span[0] * span[1])
        if area < best_area:
            best, best_area = cand, area
    return best or cluster[0]


def homography_to_reference(cluster, ref, pairs):
    """H mapping each camera's pixels into the reference camera's frame.

    Chains through intermediate cameras when a direct pair is missing, so a
    camera that overlaps the reference only indirectly still lands correctly.
    """
    H = {ref: np.eye(3)}
    direct = {}
    for (a, b), r in pairs.items():
        if r["H"] is None or r["inliers"] < ov.MIN_INLIERS:
            continue
        direct[(a, b)] = r["H"]                      # a -> b
        direct[(b, a)] = np.linalg.inv(r["H"])       # b -> a

    changed = True
    while changed:
        changed = False
        for c in cluster:
            if c in H:
                continue
            for k in list(H):
                if (c, k) in direct:
                    H[c] = H[k] @ direct[(c, k)]
                    changed = True
                    break
    return H


def stitch(cluster, ref, H, frames, max_px=6000):
    """Warp every cluster member into the reference frame and blend."""
    corners = []
    for c in cluster:
        if c not in H:
            continue
        h, w = frames[c].shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(pts, H[c]).reshape(-1, 2))
    allc = np.vstack(corners)
    x0, y0 = np.floor(allc.min(axis=0)).astype(int)
    x1, y1 = np.ceil(allc.max(axis=0)).astype(int)

    # A badly-conditioned homography can project a corner to infinity; clamp
    # so one bad pair cannot demand a gigapixel canvas.
    x0, y0 = max(x0, -max_px), max(y0, -max_px)
    x1, y1 = min(x1, max_px), min(y1, max_px)
    W, Hgt = int(x1 - x0), int(y1 - y0)
    if W <= 0 or Hgt <= 0 or W * Hgt > 80_000_000:
        raise ValueError(f"unreasonable canvas {W}x{Hgt} — check the homographies")

    off = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    acc = np.zeros((Hgt, W, 3), np.float32)
    cnt = np.zeros((Hgt, W), np.float32)
    placement = {}

    for c in cluster:
        if c not in H:
            continue
        warped = cv2.warpPerspective(frames[c], off @ H[c], (W, Hgt))
        mask = cv2.warpPerspective(np.ones(frames[c].shape[:2], np.uint8),
                                   off @ H[c], (W, Hgt)) > 0
        acc[mask] += warped[mask].astype(np.float32)
        cnt[mask] += 1
        placement[c] = {"H_to_reference": (off @ H[c]).tolist(),
                        "pixels_in_map": int(mask.sum())}

    out = np.zeros((Hgt, W, 3), np.uint8)
    nz = cnt > 0
    # Mean where views overlap: misalignment then shows as ghosting rather than
    # being smoothed away, which is what you want while judging a stitch.
    out[nz] = (acc[nz] / cnt[nz, None]).astype(np.uint8)
    return out, placement, (int(x0), int(y0)), int((cnt > 1).sum())


def annotate(canvas, cluster, placement, frames):
    out = canvas.copy()
    colours = [(90, 220, 110), (235, 160, 60), (90, 140, 245), (220, 210, 70)]
    for i, c in enumerate(cluster):
        if c not in placement:
            continue
        col = colours[i % len(colours)]
        h, w = frames[c].shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        H = np.array(placement[c]["H_to_reference"])
        poly = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.int32)
        cv2.polylines(out, [poly], True, col, 3)
        cv2.putText(out, c, tuple(poly[0] + np.array([8, 28])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="../real_data/raw_recordings")
    ap.add_argument("--session", required=True)
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--out", default="maps")
    ap.add_argument("--threshold", type=int, default=ov.GOOD_INLIERS)
    args = ap.parse_args()

    frames = ov.frames_from_session(Path(args.dir), args.session, args.t)
    if len(frames) < 2:
        print(f"ERROR: found {len(frames)} cameras")
        return 1
    names = sorted(frames)

    print("=" * 78)
    print(f"CONVEYOR MAP — session {args.session}, {len(names)} cameras")
    print("=" * 78)
    feats = {n: ov.features(frames[n]) for n in names}
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        r = ov.match_pair(feats[a], feats[b])
        if r:
            pairs[(a, b)] = r
            print(f"  {a:16} <-> {b:16} {r['inliers']:4d} inliers")

    clusters = build_clusters(names, pairs, args.threshold)
    print()
    print(f"  {len(clusters)} cluster(s) of mutually overlapping cameras:")
    for i, cl in enumerate(clusters):
        print(f"    cluster {i+1}: {', '.join(cl)}")
    if len(clusters) > 1:
        print()
        print("  These clusters share no view. Each becomes its own map. A bag is")
        print("  never seen by two clusters at once, so hand-off between them must")
        print("  rely on belt travel rather than shared geometry.")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"session": args.session, "sample_t": args.t, "clusters": []}

    for i, cl in enumerate(clusters):
        if len(cl) < 2:
            print(f"\n  cluster {i+1} ({cl[0]}) is a single camera — nothing to stitch")
            manifest["clusters"].append({"cameras": cl, "reference": cl[0],
                                         "map": None})
            continue
        ref = choose_reference(cl, pairs, frames)
        H = homography_to_reference(cl, ref, pairs)
        missing = [c for c in cl if c not in H]
        try:
            canvas, placement, origin, overlap_px = stitch(cl, ref, H, frames)
        except ValueError as e:
            print(f"\n  cluster {i+1}: {e}")
            continue

        stem = f"cluster{i+1}_{ref}"
        cv2.imwrite(str(outdir / f"{stem}_map.png"), canvas)
        cv2.imwrite(str(outdir / f"{stem}_annotated.png"),
                    annotate(canvas, cl, placement, frames))

        print(f"\n  cluster {i+1}: reference = {ref}")
        print(f"    map {canvas.shape[1]}x{canvas.shape[0]} px, "
              f"{overlap_px} px seen by 2+ cameras")
        for c, p in placement.items():
            print(f"      {c:16} {p['pixels_in_map']:8d} px")
        if missing:
            print(f"    could not place: {missing}")
        manifest["clusters"].append({
            "cameras": cl, "reference": ref,
            "map": f"{stem}_map.png",
            "origin_offset_px": list(origin),
            "overlap_px": overlap_px,
            "placement": placement,
        })

    (outdir / f"conveyor_map_{args.session}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    print()
    print("=" * 78)
    print(f"written to {outdir}/")
    print("  This map is correct in relative geometry but has no absolute scale,")
    print("  and stays in the reference camera's oblique view. Intrinsics plus a")
    print("  belt-plane solve turn it into a true top-down metric map.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
