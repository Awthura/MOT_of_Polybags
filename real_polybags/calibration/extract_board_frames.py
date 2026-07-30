#!/usr/bin/env python3
"""
Pull calibration frames out of a recorded video — OVGU AMS calibration tool.

Bridges the recorder to the calibration tool. `pypylon` and `pyrealsense2` are
not installed here, so the calibration app cannot open a Basler or a RealSense
directly — but `real_data/utils/record_all_5_cameras_macos.py` already can, and
already writes every camera at the resolution the rig records at. So: record the
board with the tool that works, extract the usable frames here, and point the
app's **Folder** source at the result.

That ordering has a second advantage even when the SDKs are present. A recording
is a permanent artefact. A calibration made from it can be re-run, re-checked and
argued with months later, which a live session that exists only as its own result
cannot.

Two things this does beyond dumping frames:

- **Keeps only frames where the board is actually detected**, with enough
  corners to be worth using. A folder of frames where the board is blurred, cut
  off or absent produces a calibration that fails for reasons that are hard to
  see afterwards.
- **Selects for pose variety rather than taking every Nth frame.** Video is
  highly redundant: at 15 fps a slowly moved board gives hundreds of nearly
  identical views. Twenty of those constrain the lens no better than one does,
  while looking like a healthy sample — the exact failure mode the manual warns
  about, where a confident calibration comes out badly wrong. Frames are chosen
  greedily to spread out in position, apparent size and tilt.

    python3 extract_board_frames.py board_basler_1.avi --out frames/basler_1
    python3 extract_board_frames.py board_lucid.avi --out frames/lucid \
        --board large --max 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "core"))

import board as board_mod          # noqa: E402
import intrinsics as intr          # noqa: E402


def pose_features(det) -> np.ndarray:
    """A cheap descriptor of 'what kind of view is this'.

    Position in frame, apparent size and aspect. Two frames that are close in
    this space add nothing to each other; spreading out in it is what gives the
    calibration something to work with. Deliberately not the solved pose — that
    would need the intrinsics this is being run to obtain.
    """
    pts = det.image_points.reshape(-1, 2)
    w, h = det.image_size
    cx, cy = pts.mean(axis=0)
    spread_x = pts[:, 0].max() - pts[:, 0].min()
    spread_y = pts[:, 1].max() - pts[:, 1].min()
    size = np.sqrt(max(spread_x * spread_y, 1.0))
    # Foreshortening: a square board seen obliquely has a lopsided bounding box.
    aspect = spread_x / max(spread_y, 1.0)
    return np.array([cx / w, cy / h, size / max(w, h), np.log(max(aspect, 1e-3))])


def select_diverse(cands: list, k: int) -> list[int]:
    """Farthest-point sampling over the descriptors above.

    Greedy and O(n·k), which is nothing at these sizes, and it does the one
    thing that matters: never returns two views that are the same view.
    """
    if len(cands) <= k:
        return list(range(len(cands)))
    F = np.stack([c["feat"] for c in cands])
    # Start from the largest board — usually the sharpest and best conditioned.
    picked = [int(np.argmax(F[:, 2]))]
    d = np.linalg.norm(F - F[picked[0]], axis=1)
    while len(picked) < k:
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(F - F[nxt], axis=1))
    return picked


def coverage_grid(dets, grid: int = 4) -> tuple[np.ndarray, int]:
    w, h = dets[0].image_size
    occ = np.zeros((grid, grid), bool)
    for d in dets:
        for p in d.image_points.reshape(-1, 2):
            gx = min(grid - 1, max(0, int(p[0] / w * grid)))
            gy = min(grid - 1, max(0, int(p[1] / h * grid)))
            occ[gy, gx] = True
    return occ, int(occ.sum())


def sharpness(img) -> float:
    """Variance of Laplacian. Motion blur moves corners; a blurred view is
    worse than no view, because it is still confidently detected."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="recorded video containing the board")
    ap.add_argument("--out", required=True, help="output frame directory")
    ap.add_argument("--board", default="small", choices=list(board_mod.PRESETS),
                    help="which printed board is in the video")
    ap.add_argument("--max", type=int, default=25,
                    help="frames to keep (15-25 is the useful range)")
    ap.add_argument("--stride", type=int, default=2,
                    help="examine every Nth frame")
    ap.add_argument("--min-corners", type=int, default=0,
                    help="minimum detected corners; default is a third of the board")
    ap.add_argument("--min-sharpness", type=float, default=25.0,
                    help="reject blurred frames below this Laplacian variance")
    args = ap.parse_args()

    spec = board_mod.PRESETS[args.board]
    min_corners = args.min_corners or max(6, spec.n_corners // 3)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  video       {args.video}")
    print(f"  board       {spec.name}  ({spec.n_corners} corners, "
          f"{spec.square_mm} mm squares)")
    print(f"  keeping     >= {min_corners} corners, sharpness >= {args.min_sharpness}")

    cands, seen, blurred, thin = [], 0, 0, 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride:
            i += 1
            continue
        i += 1
        seen += 1
        det = intr.detect_charuco(frame, spec, source=f"frame_{i:06d}")
        if det is None:
            continue
        if det.n_points < min_corners:
            thin += 1
            continue
        s = sharpness(frame)
        if s < args.min_sharpness:
            blurred += 1
            continue
        cands.append({"frame": frame, "det": det, "feat": pose_features(det),
                      "sharp": s, "idx": i})
    cap.release()

    print(f"\n  examined    {seen} frames (of {total})")
    print(f"  detected    {len(cands) + thin + blurred}")
    print(f"    too few corners  {thin}")
    print(f"    too blurred      {blurred}")
    print(f"  usable      {len(cands)}")

    if not cands:
        raise SystemExit(
            "\nNo usable frames. Either the board never appears, the wrong "
            "--board preset was given, or every view is blurred. Check one "
            "frame by eye before re-running.")

    keep = select_diverse(cands, args.max)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for n, ci in enumerate(sorted(keep, key=lambda j: cands[j]["idx"])):
        cv2.imwrite(str(out / f"shot_{n:03d}.png"), cands[ci]["frame"])

    dets = [cands[j]["det"] for j in keep]
    occ, seen_cells = coverage_grid(dets)
    print(f"\n  wrote       {len(keep)} frames -> {out}")
    print(f"  coverage    {seen_cells}/16 regions of the frame")
    for row in occ:
        print("              " + " ".join("#" if v else "." for v in row))

    if seen_cells < 12:
        print("\n  WARNING: the board never reached "
              f"{16 - seen_cells} of the 16 frame regions. Lens distortion is "
              "strongest at the edges and corners, so those coefficients will "
              "be poorly constrained. Re-record with the board pushed right "
              "into the corners of the frame.")
    if len(keep) < 10:
        print(f"\n  WARNING: only {len(keep)} usable frames. Aim for 15-25.")
    print("\n  Next: run the calibration app, choose the Folder source, and "
          f"point it at\n        {out}")


if __name__ == "__main__":
    main()
