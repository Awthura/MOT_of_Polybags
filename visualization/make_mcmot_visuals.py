#!/usr/bin/env python3
"""
make_mcmot_visuals.py
Three steps for synth_dataset_mcmot/:
  1. YOLO OBB overlays  -> yolo_overlays/
  2. MOT overlays        -> mot_overlays/{cam_short}/
  3. 2x2 grid video      -> mot_tracking_4cam_mcmot.mp4
"""

import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

BASE    = Path("/Users/awthura/OVGU/AMS")
SD      = BASE / "synth_dataset_mcmot"
IMG_DIR = SD / "images"
LBL_DIR = SD / "labels"
YOLO_OV = SD / "yolo_overlays"
MOT_OV  = SD / "mot_overlays"
VIDEO   = SD / "mot_tracking_4cam_mcmot.mp4"

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]

CLASS_NAMES  = ["pink","blue","yellow","grey","green","red","teal"]
CLASS_COLORS = [        # BGR
    (180,  80, 255),    # 0 pink
    (255, 150,  80),    # 1 blue
    (  0, 220, 255),    # 2 yellow
    (180, 180, 180),    # 3 grey
    ( 60, 220,  60),    # 4 green
    ( 40,  40, 255),    # 5 red
    (200, 180,  50),    # 6 teal
]

# 11 distinct track colours (BGR)
TRACK_COLORS = [
    (  0,   0, 255),   # 1  red
    ( 60, 220,  60),   # 2  green
    (255, 100,  30),   # 3  blue
    (  0, 220, 255),   # 4  yellow
    (200,  80, 255),   # 5  pink/magenta
    (200, 180,  50),   # 6  teal
    ( 80, 160, 255),   # 7  orange
    (160, 160, 160),   # 8  grey
    (255,  50, 180),   # 9  violet
    (  0, 180, 255),   # 10 gold
    (120, 255, 180),   # 11 lime
]

W, H = 1920, 1080
CELL_W, CELL_H = W // 2, H // 2
FPS = 25
CROP_RATIO = 0.55   # show center 55% of each frame (polybag action zone)


# ── Step 1: YOLO OBB overlays ─────────────────────────────────────────────────

def yolo_overlay_one(args):
    img_path, lbl_path, out_path = args
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [coords[i] * w for i in range(0, 8, 2)]
        ys = [coords[i] * h for i in range(1, 8, 2)]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        label = CLASS_NAMES[cid % len(CLASS_NAMES)]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        cv2.putText(img, label, (cx - 20, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 1


def step1_yolo_overlays():
    YOLO_OV.mkdir(exist_ok=True)
    tasks = []
    for img_path in sorted(IMG_DIR.glob("*.png")):
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        out_path = YOLO_OV / (img_path.stem + "_overlay.jpg")
        if lbl_path.exists():
            tasks.append((img_path, lbl_path, out_path))

    print(f"[1/3] YOLO OBB overlays: {len(tasks)} images...")
    workers = max(1, cpu_count() - 1)
    with Pool(workers) as pool:
        done = sum(pool.map(yolo_overlay_one, tasks))
    print(f"      {done} overlays written -> {YOLO_OV}")


# ── Step 2: MOT overlays ──────────────────────────────────────────────────────

def step2_mot_overlays():
    print(f"[2/3] MOT overlays (4 cameras x 500 frames)...")

    # Parse all gt_obb.txt files: {cam_short: {seq_idx: [(track_id, box_4x2), ...]}}
    mot_data = {}
    for cam_short, cam_sub in CAMS:
        gt_path = SD / "mot_obb" / cam_sub / "gt" / "gt_obb.txt"
        if not gt_path.exists():
            continue
        frames = {}
        for line in gt_path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 11:
                continue
            seq_idx  = int(parts[0])
            track_id = int(parts[1])
            corners  = np.array([float(v) for v in parts[2:10]], dtype=np.float32).reshape(4, 2)
            frames.setdefault(seq_idx, []).append((track_id, corners))
        mot_data[cam_short] = frames

    total = 0
    for cam_short, cam_sub in CAMS:
        out_dir = MOT_OV / cam_short
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = mot_data.get(cam_short, {})

        for seq_idx, annotations in sorted(frames.items()):
            frame_num = seq_idx + 99          # seq 1 = frame 100
            img_path  = IMG_DIR / f"{cam_short}_frame_{frame_num:04d}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            for track_id, corners in annotations:
                color = TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]
                pts   = corners.astype(np.int32)
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
                cx = int(corners[:, 0].mean())
                cy = int(corners[:, 1].mean())
                cv2.putText(img, f"ID{track_id}", (cx - 16, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            out_path = out_dir / f"{cam_short}_frame_{frame_num:04d}.png"
            cv2.imwrite(str(out_path), img)
            total += 1

        print(f"      {cam_short}: {len(frames)} frames")

    print(f"      {total} MOT overlay images written -> {MOT_OV}")


# ── Step 3: 2x2 grid video ────────────────────────────────────────────────────

def load_cell(cam_short: str, frame_num: int) -> np.ndarray:
    ov_path  = MOT_OV  / cam_short / f"{cam_short}_frame_{frame_num:04d}.png"
    img_path = IMG_DIR / f"{cam_short}_frame_{frame_num:04d}.png"
    img = None
    if ov_path.exists():
        img = cv2.imread(str(ov_path))
    elif img_path.exists():
        img = cv2.imread(str(img_path))
    if img is None:
        img = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
        cv2.putText(img, f"{cam_short.upper()} no data",
                    (CELL_W // 2 - 120, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        return img
    ih, iw = img.shape[:2]
    cw = int(iw * CROP_RATIO)
    ch = int(ih * CROP_RATIO)
    x0 = (iw - cw) // 2
    y0 = (ih - ch) // 2
    img = img[y0:y0+ch, x0:x0+cw]
    img = cv2.resize(img, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    lbl = cam_short.upper()
    cv2.putText(img, lbl, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img, lbl, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return img


def step3_video():
    print(f"[3/3] Building 2x2 grid video...")

    # All frame numbers present in any camera's MOT overlays
    frame_nums = set()
    for cam_short, _ in CAMS:
        for f in (MOT_OV / cam_short).glob("*.png"):
            frame_nums.add(int(f.stem.split("_frame_")[1]))
    frame_nums = sorted(frame_nums)
    print(f"      {len(frame_nums)} frames (frame {frame_nums[0]}..{frame_nums[-1]})")

    tmpdir = Path(tempfile.mkdtemp(prefix="mcmot_video_"))
    try:
        for i, fn in enumerate(frame_nums):
            cells = [load_cell(cam_short, fn) for cam_short, _ in CAMS]
            grid  = np.vstack([np.hstack([cells[0], cells[1]]),
                               np.hstack([cells[2], cells[3]])])
            # frame number bottom-right
            label = f"Frame {fn}"
            cv2.putText(grid, label, (W - 230, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(grid, label, (W - 230, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.imwrite(str(tmpdir / f"frame_{i:05d}.png"), grid)
            if (i + 1) % 100 == 0:
                print(f"      {i+1}/{len(frame_nums)} grid frames built")

        print("      Encoding with ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "20",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            str(VIDEO),
        ], check=True, capture_output=True)
        print(f"      Video -> {VIDEO}")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    step1_yolo_overlays()
    step2_mot_overlays()
    step3_video()
    print("\nAll done.")
