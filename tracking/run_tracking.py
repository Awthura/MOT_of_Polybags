#!/usr/bin/env python3
"""
tracking/run_tracking.py

Runs YOLO OBB tracking (ByteTrack or BoT-SORT) on the val/test datasets,
one camera at a time. Outputs per-camera annotated frames, MOT16 prediction
files, and a 2×2 grid video.

Usage:
  python run_tracking.py --model ../training/weights_synth_hires.pt \\
      --dataset val  --tracker bytetrack
  python run_tracking.py --model ../training/weights_synth_hires.pt \\
      --dataset test --tracker botsort
  python run_tracking.py --model ../training/weights_synth_hires.pt \\
      --dataset both --tracker both          # run all 4 combinations
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[1]
BASE    = Path("/Users/awthura/OVGU/AMS")
OUT_ROOT = REPO / "tracking_results"

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]

CLASS_NAMES = ["pink", "blue", "yellow", "grey", "green", "red", "teal"]

# BGR colors per class (for detection overlay)
CLASS_COLORS = [
    (180,  80, 255),  # pink
    (255, 150,  80),  # blue
    (  0, 220, 255),  # yellow
    (180, 180, 180),  # grey
    ( 60, 220,  60),  # green
    ( 40,  40, 255),  # red
    (200, 180,  50),  # teal
]

# BGR colors per track ID (cycles)
TRACK_COLORS = [
    (  0,   0, 255), ( 60, 220,  60), (255, 100,  30), (  0, 220, 255),
    (200,  80, 255), (200, 180,  50), ( 80, 160, 255), (160, 160, 160),
    (255,  50, 180), (  0, 180, 255), (120, 255, 180), (255, 200,   0),
    (  0, 128, 255), (128,   0, 255), (255,   0, 128), ( 50, 255, 200),
]

DATASETS = {
    "val": {
        "ds_dir":       BASE / "synth_dataset_val",
        "frame_start":  1000,
        "frame_end":    1250,
        "frame_offset": 999,   # seq_idx = frame_num - frame_offset  (1-based)
    },
    "test": {
        "ds_dir":       BASE / "synth_dataset_test",
        "frame_start":  1500,
        "frame_end":    1750,
        "frame_offset": 1499,
    },
}

TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort":   "botsort.yaml",
}

W, H = 1920, 1080
CELL_W, CELL_H = W // 2, H // 2
FPS = 25
CROP_RATIO = 0.55   # show center 55% of each frame (polybag action zone)


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_obb(img: np.ndarray, corners: np.ndarray, track_id: int, cls_id: int):
    pts   = corners.astype(np.int32)
    color = TRACK_COLORS[(track_id - 1) % len(TRACK_COLORS)]
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    label = f"{CLASS_NAMES[cls_id % len(CLASS_NAMES)]} #{track_id}"
    # black outline, colored text
    cv2.putText(img, label, (cx - 32, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (cx - 32, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_frame(img: np.ndarray, frame_num: int, cam_label: str):
    cv2.putText(img, cam_label, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img, cam_label, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    cv2.putText(img, f"Frame {frame_num}", (W - 270, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
    cv2.putText(img, f"Frame {frame_num}", (W - 270, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)


# ── Per-camera tracking ────────────────────────────────────────────────────────

def track_camera(model: YOLO, cam_short: str, ds_cfg: dict,
                 tracker_cfg: str, cam_out: Path,
                 imgsz: int, conf: float) -> int:
    """
    Track one camera's frame sequence. Returns number of track rows written.
    cam_out/frames/  → annotated JPEG frames
    cam_out/pred.txt → MOT16 predictions (1-based seq_idx)
    """
    frames_dir = cam_out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    img_dir      = ds_cfg["ds_dir"] / "images"
    frame_start  = ds_cfg["frame_start"]
    frame_end    = ds_cfg["frame_end"]
    frame_offset = ds_cfg["frame_offset"]

    img_paths = sorted(
        [p for p in img_dir.glob(f"{cam_short}_frame_*.png")
         if frame_start <= int(p.stem.split("_frame_")[1]) <= frame_end],
        key=lambda p: int(p.stem.split("_frame_")[1]),
    )

    if not img_paths:
        print(f"    {cam_short}: no frames found in {img_dir}")
        return 0

    print(f"    {cam_short}: {len(img_paths)} frames")

    # Reset tracker state before each new camera sequence
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    mot_lines = []
    for img_path in img_paths:
        frame_num = int(img_path.stem.split("_frame_")[1])
        seq_idx   = frame_num - frame_offset  # 1-based

        img     = cv2.imread(str(img_path))
        results = model.track(img, tracker=tracker_cfg, persist=True,
                              conf=conf, imgsz=imgsz, verbose=False)
        result  = results[0]

        if result.obb is not None and result.obb.id is not None:
            ids     = result.obb.id.cpu().numpy().astype(int)
            clss    = result.obb.cls.cpu().numpy().astype(int)
            confs   = result.obb.conf.cpu().numpy()
            corners = result.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)

            for i in range(len(ids)):
                draw_obb(img, corners[i], ids[i], clss[i])

                # AABB for standard MOT16 evaluation format
                x1 = int(corners[i, :, 0].min())
                y1 = int(corners[i, :, 1].min())
                x2 = int(corners[i, :, 0].max())
                y2 = int(corners[i, :, 1].max())
                mot_lines.append(
                    f"{seq_idx},{ids[i]},{x1},{y1},{x2-x1},{y2-y1},"
                    f"{confs[i]:.4f},-1,-1,-1"
                )

        annotate_frame(img, frame_num, cam_short.upper())
        out_name = f"{cam_short}_frame_{frame_num:04d}.jpg"
        cv2.imwrite(str(frames_dir / out_name), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    pred_path = cam_out / "pred.txt"
    pred_path.write_text("\n".join(mot_lines))
    print(f"      → {len(mot_lines)} detections → {pred_path.name}")
    return len(mot_lines)


# ── 2×2 grid video ────────────────────────────────────────────────────────────

def build_grid_video(tracker_name: str, dataset_name: str,
                     ds_cfg: dict, tracker_out: Path):
    """Assemble 2×2 grid MP4 from the four per-camera annotated frame dirs."""
    frame_start  = ds_cfg["frame_start"]
    frame_end    = ds_cfg["frame_end"]
    frame_nums   = list(range(frame_start, frame_end + 1))

    print(f"  Building 2×2 grid video ({len(frame_nums)} frames)…")
    video_path = tracker_out / f"4cam_grid_{tracker_name}_{dataset_name}.mp4"
    tmpdir = Path(tempfile.mkdtemp(prefix="track_grid_"))

    try:
        for i, fn in enumerate(frame_nums):
            cells = []
            for cam_short, _ in CAMS:
                img_path = tracker_out / cam_short / "frames" / \
                           f"{cam_short}_frame_{fn:04d}.jpg"
                if img_path.exists():
                    cell = cv2.imread(str(img_path))
                    cw = int(W * CROP_RATIO)
                    ch = int(H * CROP_RATIO)
                    x0 = (W - cw) // 2
                    y0 = (H - ch) // 2
                    cell = cell[y0:y0+ch, x0:x0+cw]
                    cell = cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
                else:
                    cell = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
                    cv2.putText(cell, f"{cam_short.upper()} missing",
                                (CELL_W // 2 - 120, CELL_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
                cells.append(cell)

            grid = np.vstack([np.hstack([cells[0], cells[1]]),
                              np.hstack([cells[2], cells[3]])])
            cv2.imwrite(str(tmpdir / f"frame_{i:05d}.png"), grid)

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "20",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            str(video_path),
        ], check=True, capture_output=True)
        print(f"    Video → {video_path}")
    finally:
        shutil.rmtree(tmpdir)


# ── Inference-only (no tracking) on test set ──────────────────────────────────

def run_inference_only(model: YOLO, ds_cfg: dict, out_dir: Path,
                       imgsz: int, conf: float):
    """
    Pure detection (no tracker) on every frame of every camera.
    Uses class-color coding instead of track-color coding.
    """
    print("\n  [inference-only] running detection on all cameras…")
    img_dir     = ds_cfg["ds_dir"] / "images"
    frame_start = ds_cfg["frame_start"]
    frame_end   = ds_cfg["frame_end"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(
        [p for p in img_dir.glob("*.png")
         if frame_start <= int(p.stem.split("_frame_")[1]) <= frame_end],
        key=lambda p: (p.stem.split("_frame_")[0],
                       int(p.stem.split("_frame_")[1])),
    )
    print(f"    {len(all_paths)} frames total")

    for img_path in all_paths:
        img     = cv2.imread(str(img_path))
        results = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)
        result  = results[0]

        if result.obb is not None:
            clss   = result.obb.cls.cpu().numpy().astype(int)
            confs  = result.obb.conf.cpu().numpy()
            corners = result.obb.xyxyxyxy.cpu().numpy()
            for i in range(len(clss)):
                pts   = corners[i].astype(np.int32)
                color = CLASS_COLORS[clss[i] % len(CLASS_COLORS)]
                cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                label = f"{CLASS_NAMES[clss[i] % 7]} {confs[i]:.2f}"
                cv2.putText(img, label, (cx - 32, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, label, (cx - 32, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        frame_num = int(img_path.stem.split("_frame_")[1])
        cam_short = img_path.stem.split("_frame_")[0]
        annotate_frame(img, frame_num, cam_short.upper())
        cv2.imwrite(str(out_dir / (img_path.stem + "_det.jpg")), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"    Detection frames → {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=str(REPO / "training/weights_synth_hires.pt"))
    ap.add_argument("--dataset", default="both",
                    choices=["val", "test", "both"])
    ap.add_argument("--tracker", default="both",
                    choices=["bytetrack", "botsort", "both"])
    ap.add_argument("--imgsz",   type=int,   default=1920)
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--inference-only", action="store_true",
                    help="Run detection only (no tracking) on test set")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    datasets  = ["val", "test"] if args.dataset == "both" else [args.dataset]
    trackers  = ["bytetrack", "botsort"] if args.tracker == "both" else [args.tracker]

    model = YOLO(str(model_path))

    if args.inference_only:
        ds_cfg  = DATASETS["test"]
        out_dir = OUT_ROOT / "inference_only"
        run_inference_only(model, ds_cfg, out_dir, args.imgsz, args.conf)
        return

    for dataset_name in datasets:
        ds_cfg = DATASETS[dataset_name]
        if not ds_cfg["ds_dir"].exists():
            print(f"  Dataset not found: {ds_cfg['ds_dir']} — skipping")
            continue

        for tracker_name in trackers:
            tracker_cfg = TRACKERS[tracker_name]
            tracker_out = OUT_ROOT / tracker_name / dataset_name
            tracker_out.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  Tracker={tracker_name}  Dataset={dataset_name}")
            print(f"  Model:   {model_path.name}")
            print(f"  Output:  {tracker_out}")
            print(f"{'='*60}")

            for cam_short, _ in CAMS:
                cam_out = tracker_out / cam_short
                track_camera(model, cam_short, ds_cfg, tracker_cfg,
                             cam_out, args.imgsz, args.conf)

            build_grid_video(tracker_name, dataset_name, ds_cfg, tracker_out)

    print("\nDone. Run evaluate_mot.py to compute metrics.")


if __name__ == "__main__":
    main()
