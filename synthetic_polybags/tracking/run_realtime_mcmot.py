#!/usr/bin/env python3
"""
tracking/run_realtime_mcmot.py

Real-time MCMOT simulation: all 4 cameras are processed at each timestep t,
inter-camera association runs on the live active tracks of that frame, then
annotated frames with GLOBAL IDs are rendered before advancing to t+1.

Architecture per frame
----------------------
  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
  │ front  │  │  back  │  │  left  │  │ right  │  ← load frame t
  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
      │            │            │            │
  [ByteTrack]  [ByteTrack]  [ByteTrack]  [ByteTrack]   ← intra-cam tracking
  (parallel threads)
      │            │            │            │
      └────────────┴────────────┴────────────┘
                        │
              [OnlineAssociator]    ← inter-camera association on active tracks
                        │
          global_id assigned to every active detection
                        │
          annotated 2×2 grid frame → video writer
                        │
                    frame t+1

Association strategies (--method):
  class_rank   — same class → rank by x-center → assign same global ID
                 O(n) per class, zero memory, works for non-overlapping bags
  class_iou    — same class → predict next position with constant velocity →
                 match across cameras by predicted-box IoU → Hungarian
                 Handles bags that swap x-order between frames
  class_smooth — class_rank but with temporal smoothing: new assignments
                 are dampened by recent history (EMA on global ID votes)
                 Reduces ID flicker for partially occluded bags

Usage
-----
  cd repo/tracking
  python run_realtime_mcmot.py --dataset val  --method class_rank
  python run_realtime_mcmot.py --dataset test --method class_iou
  python run_realtime_mcmot.py --dataset both --method all   # benchmark all 3
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
BASE     = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
OUT_ROOT = REPO / "tracking_results" / "realtime"

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]

CLASS_NAMES  = ["pink", "blue", "yellow", "grey", "green", "red", "teal"]
TRACK_COLORS = [
    (  0,   0, 255), ( 60, 220,  60), (255, 100,  30), (  0, 220, 255),
    (200,  80, 255), (200, 180,  50), ( 80, 160, 255), (160, 160, 160),
    (255,  50, 180), (  0, 180, 255), (120, 255, 180), (255, 200,   0),
    (  0, 128, 255), (128,   0, 255), (255,   0, 128), ( 50, 255, 200),
    (200, 100, 100), (100, 200, 100), (100, 100, 200), (220, 180,  80),
]

DATASETS = {
    "train": {
        "ds_dir":       BASE / "synth_dataset_mcmot",
        "frame_start":  100,
        "frame_end":    599,
        "frame_offset": 99,
    },
    "val": {
        "ds_dir":       BASE / "synth_dataset_val",
        "frame_start":  1000,
        "frame_end":    1250,
        "frame_offset": 999,
    },
    "test": {
        "ds_dir":       BASE / "synth_dataset_test",
        "frame_start":  1500,
        "frame_end":    1750,
        "frame_offset": 1499,
    },
}

W, H = 1920, 1080
CELL_W, CELL_H = W // 2, H // 2
CROP_RATIO = 0.55
FPS = 25


# ══════════════════════════════════════════════════════════════════════════════
# Online inter-camera associators
# ══════════════════════════════════════════════════════════════════════════════

class OnlineAssociator:
    """Base class. Subclasses implement associate_frame()."""

    def __init__(self):
        # Persistent mapping: (cam, local_id) → global_id
        self.local_to_global: dict[tuple, int] = {}
        self.global_class: dict[int, int]      = {}
        self.next_gid = 1

    def _new_gid(self, class_id: int) -> int:
        gid = self.next_gid
        self.next_gid += 1
        self.global_class[gid] = class_id
        return gid

    def _resolve_group(self, cam_local_pairs: list[tuple[str, int]],
                       class_id: int) -> int:
        """Find or create a global ID for a group of (cam, local_id) pairs."""
        for cam, lid in cam_local_pairs:
            gid = self.local_to_global.get((cam, lid))
            if gid is not None:
                return gid
        return self._new_gid(class_id)

    def associate_frame(self, cam_tracks: dict[str, list]) -> dict[tuple, int]:
        """
        cam_tracks: {cam: [(local_id, class_id, x_center, corners_4x2)]}
        Returns: updated local_to_global for all currently active tracks.
        """
        raise NotImplementedError

    def get_global_id(self, cam: str, local_id: int) -> int:
        return self.local_to_global.get((cam, local_id), -1)


class ClassRankAssociator(OnlineAssociator):
    """
    Per-frame: within each color class, sort tracks by x-center and
    assign the same global ID to same-rank detections across cameras.
    O(n log n) per class, stateless except for the ID mapping.
    """
    def associate_frame(self, cam_tracks: dict[str, list]) -> dict[tuple, int]:
        class_groups: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for cam, tracks in cam_tracks.items():
            for lid, cid, xc, corners in tracks:
                class_groups[cid][cam].append((xc, lid))

        for cid, cam_dets in class_groups.items():
            for cam in cam_dets:
                cam_dets[cam].sort()     # sort by x-center left→right
            max_n = max(len(d) for d in cam_dets.values())
            for rank in range(max_n):
                rank_pairs = [(cam, dets[rank][1])
                              for cam, dets in cam_dets.items()
                              if rank < len(dets)]
                gid = self._resolve_group(rank_pairs, cid)
                for cam, lid in rank_pairs:
                    self.local_to_global[(cam, lid)] = gid

        return self.local_to_global


class ClassSpatialHungarianAssociator(OnlineAssociator):
    """
    Per-frame: within each color class, Hungarian matching on normalized x-center
    distance between cameras.

    Cross-camera pixel IoU is always ~0 (same object projects to completely
    different pixel coordinates in each uncalibrated view), so IoU-based matching
    is useless here. Normalized x-center is comparable across views because all
    cameras observe the same left-to-right conveyor direction — the relative
    horizontal position of a bag is the same fraction of frame width in every view.

    This gives something strictly different from class_rank: class_rank assigns
    the same global ID to equal-rank detections (1st-from-left in cam1 = 1st-from-
    left in cam2), while class_spatial_hungarian minimises total |Δnorm_x| across
    all possible assignments, tolerating small rank swaps when two bags are very
    close together.
    """
    MATCH_THRESHOLD = 0.30  # max |Δnorm_x| for a valid match (~576 px at 1920 wide)

    def associate_frame(self, cam_tracks: dict[str, list]) -> dict[tuple, int]:
        from scipy.optimize import linear_sum_assignment

        class_groups: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for cam, tracks in cam_tracks.items():
            for lid, cid, xc, corners in tracks:
                norm_xc = xc / W
                class_groups[cid][cam].append((norm_xc, lid))

        for cid, cam_dets in class_groups.items():
            all_dets = [(cam, lid, nxc)
                        for cam, dets in cam_dets.items()
                        for nxc, lid in dets]

            known   = [(cam, lid, nxc) for cam, lid, nxc in all_dets
                       if (cam, lid) in self.local_to_global]
            unknown = [(cam, lid, nxc) for cam, lid, nxc in all_dets
                       if (cam, lid) not in self.local_to_global]

            if not unknown:
                continue

            if not known:
                # First-frame fallback: rank-match within class (same as class_rank)
                cam_sorted: dict[str, list] = defaultdict(list)
                for cam, lid, nxc in all_dets:
                    cam_sorted[cam].append((nxc, lid))
                for cam in cam_sorted:
                    cam_sorted[cam].sort()
                max_n = max(len(d) for d in cam_sorted.values())
                for rank in range(max_n):
                    rank_pairs = [(cam, dets[rank][1])
                                  for cam, dets in cam_sorted.items()
                                  if rank < len(dets)]
                    gid = self._resolve_group(rank_pairs, cid)
                    for cam, lid in rank_pairs:
                        self.local_to_global[(cam, lid)] = gid
                continue

            # Hungarian on |Δnorm_x|: rows=unknown, cols=known
            known_gids = [self.local_to_global[(cam, lid)] for cam, lid, _ in known]
            unk_xc = np.array([nxc for _, _, nxc in unknown])
            kn_xc  = np.array([nxc for _, _, nxc in known])
            cost   = np.abs(unk_xc[:, None] - kn_xc[None, :])  # (nu, nk)

            # Block same-camera pairs (one object can't appear twice in one camera)
            for ui, (ucam, _, _) in enumerate(unknown):
                for ki, (kcam, _, _) in enumerate(known):
                    if ucam == kcam:
                        cost[ui, ki] = 1.0

            rows, cols = linear_sum_assignment(cost)
            matched_unk: set[int] = set()
            for r, c in zip(rows, cols):
                if cost[r, c] < self.MATCH_THRESHOLD:
                    ucam, ulid, _ = unknown[r]
                    self.local_to_global[(ucam, ulid)] = known_gids[c]
                    matched_unk.add(r)

            for ui, (ucam, ulid, _) in enumerate(unknown):
                if ui not in matched_unk:
                    self.local_to_global[(ucam, ulid)] = self._new_gid(cid)

        return self.local_to_global


class ClassSmoothAssociator(ClassRankAssociator):
    """
    Class_rank + temporal smoothing: each (cam, local_id) tracks a vote
    history of global_id assignments. The majority vote over the last N
    frames wins, reducing flicker from brief mismatches.
    """
    WINDOW = 8   # number of frames to smooth over

    def __init__(self):
        super().__init__()
        # Vote history: (cam, local_id) → deque of global_id votes
        self.vote_history: dict[tuple, list[int]] = defaultdict(list)

    def associate_frame(self, cam_tracks: dict[str, list]) -> dict[tuple, int]:
        # Get raw assignments from class_rank
        super().associate_frame(cam_tracks)

        # Record vote and apply majority
        active_keys = {(cam, lid)
                       for cam, tracks in cam_tracks.items()
                       for lid, _, _, _ in tracks}
        for key in active_keys:
            raw_gid = self.local_to_global.get(key, -1)
            if raw_gid < 0:
                continue
            hist = self.vote_history[key]
            hist.append(raw_gid)
            if len(hist) > self.WINDOW:
                hist.pop(0)
            smoothed = Counter(hist).most_common(1)[0][0]
            self.local_to_global[key] = smoothed

        return self.local_to_global


ASSOCIATOR_CLS = {
    "class_rank":   ClassRankAssociator,
    "class_iou":    ClassSpatialHungarianAssociator,   # Hungarian on norm-x, not pixel IoU
    "class_smooth": ClassSmoothAssociator,
}


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_obb_global(img: np.ndarray, corners: np.ndarray,
                    global_id: int, cls_id: int):
    pts   = corners.astype(np.int32)
    color = TRACK_COLORS[(global_id - 1) % len(TRACK_COLORS)] if global_id > 0 \
            else (128, 128, 128)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
    cx = int(pts[:, 0].mean())
    cy = int(pts[:, 1].mean())
    cls_name = CLASS_NAMES[cls_id % len(CLASS_NAMES)]
    label = f"{cls_name} G#{global_id}" if global_id > 0 else cls_name
    cv2.putText(img, label, (cx - 40, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (cx - 40, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_overlay(img: np.ndarray, cam_label: str, frame_num: int):
    cv2.putText(img, cam_label, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(img, cam_label, (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    cv2.putText(img, f"Frame {frame_num}", (W - 270, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
    cv2.putText(img, f"Frame {frame_num}", (W - 270, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)


def make_grid_frame(annotated: dict[str, np.ndarray]) -> np.ndarray:
    cells = []
    for cam_short, _ in CAMS:
        img = annotated.get(cam_short)
        if img is None:
            img = np.zeros((H, W, 3), dtype=np.uint8)
        ih, iw = img.shape[:2]
        cw = int(iw * CROP_RATIO)
        ch = int(ih * CROP_RATIO)
        x0 = (iw - cw) // 2
        y0 = (ih - ch) // 2
        cell = img[y0:y0+ch, x0:x0+cw]
        cells.append(cv2.resize(cell, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA))
    return np.vstack([np.hstack([cells[0], cells[1]]),
                      np.hstack([cells[2], cells[3]])])


# ── Per-camera inference (runs in thread) ─────────────────────────────────────

def infer_camera(model: YOLO, img: np.ndarray, tracker_cfg: str,
                 conf: float, imgsz: int) -> tuple:
    """Returns (result, tracks) where tracks = [(local_id, class_id, x_center, corners)]"""
    if img is None:
        return None, []
    results = model.track(img, tracker=tracker_cfg, persist=True,
                          conf=conf, imgsz=imgsz, verbose=False)
    result  = results[0]
    tracks  = []
    if result.obb is not None and result.obb.id is not None:
        ids     = result.obb.id.cpu().numpy().astype(int)
        clss    = result.obb.cls.cpu().numpy().astype(int)
        corners = result.obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
        for i in range(len(ids)):
            x_center = float(corners[i, :, 0].mean())
            tracks.append((int(ids[i]), int(clss[i]), x_center, corners[i]))
    return result, tracks


# ── MOT prediction writer (for evaluation) ────────────────────────────────────

class MOTWriter:
    def __init__(self, path: Path, frame_offset: int):
        self.path   = path
        self.offset = frame_offset  # seq_idx = frame_num - offset
        self.lines: list[str] = []

    def record(self, frame_num: int, cam_short: str,
               cam_idx: int, global_id: int, cls_id: int, corners: np.ndarray):
        seq_idx = frame_num - self.offset
        x1 = int(corners[:, 0].min());  y1 = int(corners[:, 1].min())
        x2 = int(corners[:, 0].max());  y2 = int(corners[:, 1].max())
        # Camera-offset frame for cross-camera merged evaluation
        merged_frame = seq_idx + cam_idx * 10000
        self.lines.append(
            f"{merged_frame},{global_id},{x1},{y1},{x2-x1},{y2-y1},1.0,{cls_id},-1,-1"
        )

    def flush(self):
        self.path.write_text("\n".join(self.lines))


# ── Main processing loop ───────────────────────────────────────────────────────

def run(model_path: Path, dataset_name: str, tracker_name: str,
        method_name: str, imgsz: int, conf: float) -> Path:

    ds_cfg      = DATASETS[dataset_name]
    img_dir     = ds_cfg["ds_dir"] / "images"
    frame_start = ds_cfg["frame_start"]
    frame_end   = ds_cfg["frame_end"]
    frame_offset = ds_cfg["frame_offset"]

    tracker_cfg = f"{tracker_name}.yaml"
    out_dir     = OUT_ROOT / method_name / tracker_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # One YOLO model per camera (independent tracker state)
    models = {cam: YOLO(str(model_path)) for cam, _ in CAMS}

    # Inter-camera associator
    associator: OnlineAssociator = ASSOCIATOR_CLS[method_name]()

    # MOT writer for post-hoc evaluation
    mot_writer = MOTWriter(out_dir / "global_pred.txt", frame_offset)

    # Temp dir for grid frames
    tmpdir = Path(tempfile.mkdtemp(prefix="rt_mcmot_"))
    frame_count = frame_end - frame_start + 1
    t_start = time.perf_counter()

    print(f"\n  [{method_name} / {tracker_name} / {dataset_name}]  "
          f"{frame_count} frames × 4 cameras")

    try:
        for fn_idx, frame_num in enumerate(range(frame_start, frame_end + 1)):

            # ── Load all 4 camera frames ──────────────────────────────────────
            raw_frames: dict[str, np.ndarray | None] = {}
            for cam_short, _ in CAMS:
                p = img_dir / f"{cam_short}_frame_{frame_num:04d}.png"
                raw_frames[cam_short] = cv2.imread(str(p)) if p.exists() else None

            # ── Parallel intra-camera tracking ────────────────────────────────
            cam_results: dict[str, tuple] = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {
                    cam_short: pool.submit(
                        infer_camera, models[cam_short],
                        raw_frames[cam_short], tracker_cfg, conf, imgsz
                    )
                    for cam_short, _ in CAMS
                }
                for cam_short, fut in futures.items():
                    cam_results[cam_short] = fut.result()

            # ── Inter-camera association ──────────────────────────────────────
            cam_tracks = {cam: tracks for cam, (_, tracks) in cam_results.items()}
            associator.associate_frame(cam_tracks)

            # ── Annotate + record ─────────────────────────────────────────────
            annotated: dict[str, np.ndarray] = {}
            for cam_idx, (cam_short, _) in enumerate(CAMS):
                img = (raw_frames[cam_short].copy()
                       if raw_frames[cam_short] is not None
                       else np.zeros((H, W, 3), dtype=np.uint8))
                _, tracks = cam_results[cam_short]
                for local_id, cls_id, _, corners in tracks:
                    gid = associator.get_global_id(cam_short, local_id)
                    draw_obb_global(img, corners, gid, cls_id)
                    mot_writer.record(frame_num, cam_short, cam_idx,
                                      gid, cls_id, corners)
                annotate_overlay(img, cam_short.upper(), frame_num)
                annotated[cam_short] = img

            # ── Grid frame ────────────────────────────────────────────────────
            grid = make_grid_frame(annotated)
            cv2.imwrite(str(tmpdir / f"frame_{fn_idx:05d}.png"), grid)

            if (fn_idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_start
                fps = (fn_idx + 1) / elapsed
                print(f"    {fn_idx+1}/{frame_count}  {fps:.1f} fps  "
                      f"(~{(frame_count-fn_idx-1)/fps:.0f}s remaining)")

        # ── Encode video ──────────────────────────────────────────────────────
        video_path = out_dir / f"realtime_{method_name}_{tracker_name}_{dataset_name}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-pix_fmt", "yuv420p", str(video_path),
        ], check=True, capture_output=True)

        mot_writer.flush()
        elapsed = time.perf_counter() - t_start
        print(f"    Done in {elapsed:.1f}s ({frame_count/elapsed:.2f} fps)")
        print(f"    Video  → {video_path}")
        print(f"    Tracks → {out_dir / 'global_pred.txt'}")
        return out_dir

    finally:
        shutil.rmtree(tmpdir)


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(out_dir: Path, ds_cfg: dict) -> dict | None:
    """Compare global_pred.txt against merged GT using motmetrics."""
    try:
        import motmetrics as mm
    except ImportError:
        return None

    from associate_cameras import write_global_gt, evaluate_mot_files, _parse_mot_file

    gt_path   = out_dir / "global_gt.txt"
    pred_path = out_dir / "global_pred.txt"

    if not gt_path.exists():
        write_global_gt(ds_cfg, gt_path)

    return evaluate_mot_files(pred_path, gt_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=str(REPO / "training/weights_synth_hires.pt"))
    ap.add_argument("--dataset", default="val", choices=["train", "val", "test", "both"])
    ap.add_argument("--tracker", default="bytetrack",
                    choices=["bytetrack", "botsort", "both"])
    ap.add_argument("--method",  default="class_rank",
                    choices=list(ASSOCIATOR_CLS) + ["all"])
    ap.add_argument("--imgsz",   type=int,   default=1920)
    ap.add_argument("--conf",    type=float, default=0.25)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    datasets = ["val", "test"] if args.dataset == "both" else [args.dataset]
    if "train" in datasets and args.dataset != "train":
        datasets = [d for d in datasets if d != "train"]
    trackers = ["bytetrack", "botsort"] if args.tracker == "both" else [args.tracker]
    methods  = list(ASSOCIATOR_CLS) if args.method == "all" else [args.method]

    results = {}
    for dataset_name in datasets:
        ds_cfg = DATASETS[dataset_name]
        for tracker_name in trackers:
            for method_name in methods:
                out_dir = run(model_path, dataset_name, tracker_name,
                              method_name, args.imgsz, args.conf)
                row = evaluate(out_dir, ds_cfg)
                key = f"{method_name}/{tracker_name}/{dataset_name}"
                results[key] = row

    if len(results) > 1:
        pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else "N/A"
        hdr = f"  {'Method/Tracker/Dataset':<42}  {'MOTA':>7}  {'MOTP':>7}  {'IDF1':>7}  {'IDSW':>6}"
        print("\n" + "=" * len(hdr))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for name, row in results.items():
            if not row:
                print(f"  {name:<42}  (no result)")
            else:
                print(f"  {name:<42}  {pct(row['mota']):>7}  "
                      f"{pct(row['motp']):>7}  {pct(row['idf1']):>7}  "
                      f"{int(row['num_switches']):>6}")
        print("=" * len(hdr))


if __name__ == "__main__":
    main()
