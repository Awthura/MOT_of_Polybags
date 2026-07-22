#!/usr/bin/env python3
"""
Build a 2x2 grid video from raw 4-camera frames — no overlays, no inference.
Layout:  Front | Back
         Left  | Right
Output cell size = 50% of source (960x540 per cell → 1920x1080 total).
"""
import cv2
import numpy as np
from pathlib import Path
import subprocess, tempfile, shutil, argparse

IMG_DIR   = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset_mcmot/images")
OUT_VIDEO = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset_mcmot/raw_4cam_grid.mp4")

CAMS = ["front", "back", "left", "right"]  # grid order: top-left, top-right, bottom-left, bottom-right

SRC_W, SRC_H = 1920, 1080
CELL_W, CELL_H = SRC_W // 2, SRC_H // 2   # 960 x 540  (50% zoom)
OUT_W,  OUT_H  = CELL_W * 2, CELL_H * 2   # 1920 x 1080
CROP_RATIO = 0.55   # show center 55% of each frame (polybag action zone)

FPS = 25
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICK = 2
LABEL_CLR  = (220, 220, 220)


def load_cell(cam: str, frame_num: int) -> np.ndarray:
    path = IMG_DIR / f"{cam}_frame_{frame_num:04d}.png"
    img = cv2.imread(str(path)) if path.exists() else None
    if img is None:
        cell = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
        cv2.putText(cell, f"{cam.upper()} — missing", (CELL_W // 2 - 130, CELL_H // 2),
                    FONT, 0.8, (80, 80, 80), 2)
        return cell
    ih, iw = img.shape[:2]
    cw = int(iw * CROP_RATIO)
    ch = int(ih * CROP_RATIO)
    x0 = (iw - cw) // 2
    y0 = (ih - ch) // 2
    img = img[y0:y0 + ch, x0:x0 + cw]
    cell = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    # camera label
    cv2.putText(cell, cam.upper(), (12, 32), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK + 2)
    cv2.putText(cell, cam.upper(), (12, 32), FONT, FONT_SCALE, LABEL_CLR, FONT_THICK)
    return cell


def make_grid(frame_num: int) -> np.ndarray:
    cells = [load_cell(c, frame_num) for c in CAMS]
    grid = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    label = f"Frame {frame_num}"
    cv2.putText(grid, label, (OUT_W - 200, OUT_H - 18), FONT, 0.8, (0, 0, 0), 3)
    cv2.putText(grid, label, (OUT_W - 200, OUT_H - 18), FONT, 0.8, (200, 200, 200), 2)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_VIDEO)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    frames = sorted(set(
        int(p.stem.split("_frame_")[1])
        for p in IMG_DIR.glob("front_frame_*.png")
    ))
    if not frames:
        print(f"No front_frame_*.png found in {IMG_DIR}")
        return

    print(f"Frames: {frames[0]}–{frames[-1]}  ({len(frames)} total)")
    print(f"Output: {args.out}  ({OUT_W}x{OUT_H} @ {args.fps} fps)")

    tmpdir = Path(tempfile.mkdtemp(prefix="raw_grid_"))
    try:
        for i, fn in enumerate(frames):
            cv2.imwrite(str(tmpdir / f"frame_{i:05d}.png"), make_grid(fn))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(frames)}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(args.out),
        ], check=True)
        print(f"Done → {args.out}")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
