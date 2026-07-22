#!/usr/bin/env python3
"""
make_val_test_visuals.py
Generates YOLO OBB overlays, MOT overlays, 2x2 grid videos, and report
figures for synth_dataset_val/ (frames 1000-1250) and synth_dataset_test/
(frames 1500-1750).
"""

import cv2
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

BASE = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]

CLASS_NAMES  = ["pink", "blue", "yellow", "grey", "green", "red", "teal"]
CLASS_COLORS = [        # BGR
    (180,  80, 255),    # 0 pink
    (255, 150,  80),    # 1 blue
    (  0, 220, 255),    # 2 yellow
    (180, 180, 180),    # 3 grey
    ( 60, 220,  60),    # 4 green
    ( 40,  40, 255),    # 5 red
    (200, 180,  50),    # 6 teal
]

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


# ── YOLO OBB overlay (single image) ──────────────────────────────────────────

def _yolo_one(args):
    img_path, lbl_path, out_path = args
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cid    = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [coords[i] * w for i in range(0, 8, 2)]
        ys = [coords[i] * h for i in range(1, 8, 2)]
        pts   = np.array(list(zip(xs, ys)), dtype=np.int32)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        cv2.putText(img, CLASS_NAMES[cid % len(CLASS_NAMES)],
                    (cx - 20, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return 1


def step_yolo_overlays(ds_dir: Path, label: str):
    img_dir  = ds_dir / "images"
    lbl_dir  = ds_dir / "labels"
    out_dir  = ds_dir / "yolo_overlays"
    out_dir.mkdir(exist_ok=True)

    tasks = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            tasks.append((img_path, lbl_path, out_dir / (img_path.stem + "_overlay.jpg")))

    print(f"  [{label}] YOLO OBB overlays: {len(tasks)} images...")
    with Pool(max(1, cpu_count() - 1)) as pool:
        done = sum(tqdm(pool.imap(_yolo_one, tasks), total=len(tasks),
                        desc=f"    {label} YOLO", unit="img", ncols=72))
    print(f"    {done} overlays -> {out_dir}")


# ── MOT overlays ──────────────────────────────────────────────────────────────

def step_mot_overlays(ds_dir: Path, label: str, frame_offset: int):
    """frame_offset: seq_idx + frame_offset = actual frame number."""
    img_dir = ds_dir / "images"
    out_dir = ds_dir / "mot_overlays"

    mot_data = {}
    for cam_short, cam_sub in CAMS:
        gt_path = ds_dir / "mot_obb" / cam_sub / "gt" / "gt_obb.txt"
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
    for cam_short, _ in CAMS:
        cam_out = out_dir / cam_short
        cam_out.mkdir(parents=True, exist_ok=True)
        frames = mot_data.get(cam_short, {})
        items  = sorted(frames.items())

        for seq_idx, annotations in tqdm(items, desc=f"    {label} MOT {cam_short}",
                                         unit="frame", ncols=72):
            frame_num = seq_idx + frame_offset
            img_path  = img_dir / f"{cam_short}_frame_{frame_num:04d}.png"
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
            out_path = cam_out / f"{cam_short}_frame_{frame_num:04d}.png"
            cv2.imwrite(str(out_path), img)
            total += 1

    print(f"  [{label}] {total} MOT overlay images -> {out_dir}")


# ── 2x2 grid helpers ──────────────────────────────────────────────────────────

def _load_cell(ds_dir: Path, cam_short: str, frame_num: int,
               use_mot: bool) -> np.ndarray:
    if use_mot:
        img_path = ds_dir / "mot_overlays" / cam_short / \
                   f"{cam_short}_frame_{frame_num:04d}.png"
    else:
        img_path = ds_dir / "yolo_overlays" / \
                   f"{cam_short}_frame_{frame_num:04d}_overlay.jpg"

    fallback = ds_dir / "images" / f"{cam_short}_frame_{frame_num:04d}.png"
    img = None
    if img_path.exists():
        img = cv2.imread(str(img_path))
    elif fallback.exists():
        img = cv2.imread(str(fallback))
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


def _build_grid(ds_dir: Path, frame_num: int, use_mot: bool) -> np.ndarray:
    cells = [_load_cell(ds_dir, cam_short, frame_num, use_mot)
             for cam_short, _ in CAMS]
    grid  = np.vstack([np.hstack([cells[0], cells[1]]),
                       np.hstack([cells[2], cells[3]])])
    label = f"Frame {frame_num}"
    cv2.putText(grid, label, (W - 230, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(grid, label, (W - 230, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    return grid


# ── 2x2 grid video ────────────────────────────────────────────────────────────

def step_video(ds_dir: Path, label: str, frame_start: int, frame_end: int,
               video_name: str):
    frame_nums = sorted(
        int(f.stem.split("_frame_")[1])
        for f in (ds_dir / "mot_overlays" / "front").glob("front_frame_*.png")
    )
    if not frame_nums:
        print(f"  [{label}] No MOT overlay frames found, skipping video.")
        return

    print(f"  [{label}] Building 2x2 grid video ({len(frame_nums)} frames)...")
    video_path = ds_dir / video_name
    tmpdir = Path(tempfile.mkdtemp(prefix=f"mcmot_{label}_"))
    try:
        for i, fn in enumerate(tqdm(frame_nums, desc=f"    {label} grid",
                                     unit="frame", ncols=72)):
            grid = _build_grid(ds_dir, fn, use_mot=True)
            cv2.imwrite(str(tmpdir / f"frame_{i:05d}.png"), grid)

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "20",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            str(video_path),
        ], check=True, capture_output=True)
        print(f"    Video -> {video_path}")
    finally:
        shutil.rmtree(tmpdir)


# ── Report figures ────────────────────────────────────────────────────────────

def save_report_figures(ds_dir: Path, label: str, early_frame: int,
                        late_frame: int, prefix: str):
    """Save 2x2 grid figures (YOLO and MOT) to BASE for report inclusion."""
    for use_mot, kind in [(False, "yolo"), (True, "mot")]:
        for frame_num, tag in [(early_frame, "early"), (late_frame, "late")]:
            grid = _build_grid(ds_dir, frame_num, use_mot)
            out  = BASE / f"fig_{kind}_{prefix}_{tag}.png"
            cv2.imwrite(str(out), grid)
            print(f"    Saved {out.name}  (frame {frame_num})")


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = [
    {
        "label":        "val",
        "ds_dir":       BASE / "synth_dataset_val",
        "frame_start":  1000,
        "frame_end":    1250,
        "frame_offset": 999,   # seq_idx 1 = frame 1000
        "early_frame":  1050,
        "late_frame":   1225,
        "video_name":   "mot_tracking_4cam_val.mp4",
        "fig_prefix":   "val",
    },
    {
        "label":        "test",
        "ds_dir":       BASE / "synth_dataset_test",
        "frame_start":  1500,
        "frame_end":    1750,
        "frame_offset": 1499,  # seq_idx 1 = frame 1500
        "early_frame":  1550,
        "late_frame":   1725,
        "video_name":   "mot_tracking_4cam_test.mp4",
        "fig_prefix":   "test",
    },
]

if __name__ == "__main__":
    for ds in DATASETS:
        label = ds["label"]
        print(f"\n{'='*60}")
        print(f"  Processing {label.upper()} dataset")
        print(f"{'='*60}")

        step_yolo_overlays(ds["ds_dir"], label)
        step_mot_overlays(ds["ds_dir"], label, ds["frame_offset"])
        step_video(ds["ds_dir"], label, ds["frame_start"], ds["frame_end"],
                   ds["video_name"])

        print(f"  [{label}] Saving report figures...")
        save_report_figures(ds["ds_dir"], label, ds["early_frame"],
                            ds["late_frame"], ds["fig_prefix"])

    print("\nAll done.")
