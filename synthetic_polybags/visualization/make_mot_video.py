#!/usr/bin/env python3
"""
Build a 2x2 grid MOT overlay video from 4-camera OBB-MOT overlays.
Layout:  Front | Back
         Left  | Right
Missing camera frames are filled with a dark placeholder.
"""
import cv2
import numpy as np
from pathlib import Path
import subprocess, tempfile, shutil, os, sys

IMG_BASE = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset/images")
OV_BASE  = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset/mot_obb/overlays")
OUT_VIDEO = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/mot_tracking_4cam.mp4")

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]

W, H = 1920, 1080
CELL_W, CELL_H = W // 2, H // 2
FPS = 25
LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 1.0
LABEL_THICK = 2
LABEL_COLOR = (220, 220, 220)


def load_or_blank(short: str, sub: str, frame_num: int) -> np.ndarray:
    """Return overlay image if it exists, else original image, else dark blank."""
    ov_path  = OV_BASE  / sub / f"{short}_frame_{frame_num:04d}.png"
    img_path = IMG_BASE / f"{short}_frame_{frame_num:04d}.png"
    if ov_path.exists():
        img = cv2.imread(str(ov_path))
    elif img_path.exists():
        img = cv2.imread(str(img_path))
    else:
        img = None

    if img is None:
        img = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
        cv2.putText(img, f"{short.upper()} — no data", (CELL_W//2 - 150, CELL_H//2),
                    LABEL_FONT, 1.0, (80, 80, 80), 2)
        return img

    img = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    # camera label top-left
    label = short.upper()
    cv2.putText(img, label, (14, 38), LABEL_FONT, LABEL_SCALE, (0, 0, 0), LABEL_THICK + 2)
    cv2.putText(img, label, (14, 38), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK)
    return img


def make_grid(frame_num: int) -> np.ndarray:
    cells = [load_or_blank(short, sub, frame_num) for short, sub in CAMS]
    top    = np.hstack([cells[0], cells[1]])
    bottom = np.hstack([cells[2], cells[3]])
    grid   = np.vstack([top, bottom])
    # frame number overlay bottom-right
    label = f"Frame {frame_num}"
    cv2.putText(grid, label, (W - 220, H - 20), LABEL_FONT, 0.9, (0,0,0), 3)
    cv2.putText(grid, label, (W - 220, H - 20), LABEL_FONT, 0.9, (200,200,200), 2)
    return grid


def collect_all_frames():
    frames = set()
    for short, sub in CAMS:
        for f in OV_BASE.glob(f"{sub}/{short}_frame_*.png"):
            frames.add(int(f.stem.split("_frame_")[1]))
    return sorted(frames)


def main():
    frames = collect_all_frames()
    print(f"Building video: {len(frames)} composite frames → {OUT_VIDEO}")

    tmpdir = Path(tempfile.mkdtemp(prefix="mot_video_"))
    try:
        for i, fn in enumerate(frames):
            grid = make_grid(fn)
            out_path = tmpdir / f"frame_{i:05d}.png"
            cv2.imwrite(str(out_path), grid)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(frames)} frames rendered")

        print("Encoding video with ffmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-crf", "20",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(OUT_VIDEO),
        ]
        subprocess.run(cmd, check=True)
        print(f"Done: {OUT_VIDEO}")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()
