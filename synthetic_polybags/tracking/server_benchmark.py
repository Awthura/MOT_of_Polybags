#!/usr/bin/env python3
"""
tracking/server_benchmark.py

Self-contained MCMOT benchmark designed to run on the OVGU cluster GPU.
Runs the full pipeline (intra-cam tracking → offline association → online
real-time) and writes all results to a JSON file for local analysis.

Quickstart (on server after git pull):
    conda activate ams
    cd repo/tracking
    python server_benchmark.py \\
        --data-root /path/to/AMS \\
        --model ../training/weights_synth_hires.pt \\
        --out results/benchmark_$(date +%Y%m%d_%H%M%S).json

The JSON can then be pulled locally and fed to generate_report.py:
    python generate_report.py --from-json benchmark_YYYYMMDD_HHMMSS.json

Flags
-----
  --data-root   Root directory containing synth_dataset_val/ and synth_dataset_test/
  --model       Path to YOLO OBB .pt weights file
  --device      cuda / cuda:0 / cpu / mps / auto  (default: auto)
  --imgsz       Inference image size (default: 1920)
  --conf        Detection confidence threshold (default: 0.25)
  --datasets    Which datasets to run: val test (default: val test)
  --trackers    Which trackers: bytetrack botsort (default: bytetrack botsort)
  --online      Online methods to benchmark (default: class_rank class_iou class_smooth)
  --offline     Offline methods (default: all five)
  --skip-eval   Skip motmetrics evaluation (just time the inference)
  --out         Output JSON path (default: benchmark_results.json)
"""

import argparse
import json
import platform
import socket
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Optional heavy imports — checked at runtime ────────────────────────────────
try:
    import torch
    TORCH_OK = True
    if torch.cuda.is_available():
        # Workaround: cuDNN sublibrary version mismatch on some cluster configs.
        # Disabling cuDNN makes PyTorch fall back to native CUDA kernels.
        # On a large GPU (A40, A100) the throughput loss is acceptable.
        torch.backends.cudnn.enabled   = False
        torch.backends.cudnn.benchmark = False
except ImportError:
    TORCH_OK = False

try:
    import motmetrics as mm
    MM_OK = True
except ImportError:
    print("WARNING: motmetrics not installed — evaluation will be skipped.")
    MM_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except ImportError:
    print("WARNING: scipy not installed — class_iou method will be skipped.")
    SCIPY_OK = False

from ultralytics import YOLO
import ultralytics

# ── Constants ──────────────────────────────────────────────────────────────────
CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]
CAM_FRAME_STRIDE = 10_000
NUM_CLASSES = 7
W = 1920

OFFLINE_METHODS = ["no_assoc", "class_only",
                   "trk_temporal", "trk_spatial", "trk_combined"]
ONLINE_METHODS  = ["class_rank", "class_iou", "class_smooth"]

MOT_METRICS = ["num_frames", "mota", "motp", "idf1",
               "num_switches", "mostly_tracked", "mostly_lost",
               "num_false_positives", "num_misses"]


# ══════════════════════════════════════════════════════════════════════════════
# System info
# ══════════════════════════════════════════════════════════════════════════════

def get_system_info(device_str: str) -> dict:
    info = {
        "host":      socket.gethostname(),
        "platform":  platform.platform(),
        "python":    sys.version.split()[0],
        "torch":     getattr(torch, "__version__", "N/A") if TORCH_OK else "N/A",
        "ultralytics": ultralytics.__version__,
        "device_requested": device_str,
        "device_used":      "N/A",
        "gpu_name":         "N/A",
        "gpu_memory_gb":    "N/A",
        "cuda_version":     "N/A",
    }
    if TORCH_OK:
        if torch.cuda.is_available():
            idx = 0
            info["device_used"]   = f"cuda:{idx}"
            info["gpu_name"]      = torch.cuda.get_device_name(idx)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
            info["cuda_version"]  = torch.version.cuda or "N/A"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device_used"] = "mps"
            info["gpu_name"]    = "Apple Silicon MPS"
        else:
            info["device_used"] = "cpu"
    return info


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if TORCH_OK and torch.cuda.is_available():
        return "cuda"
    if TORCH_OK and hasattr(torch.backends, "mps") \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# MOT file I/O
# ══════════════════════════════════════════════════════════════════════════════

def parse_mot(path: Path) -> list[tuple]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        frame  = int(p[0])
        obj_id = int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf   = float(p[6]) if len(p) > 6 else 1.0
        cls_id = int(p[7]) if len(p) > 7 and p[7].strip() not in ("-1", "") else -1
        if conf < 0:
            continue
        rows.append((frame, obj_id, x, y, w, h, conf, cls_id))
    return rows


def rows_by_frame(rows: list) -> dict:
    d = defaultdict(list)
    for r in rows:
        d[r[0]].append(r)
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Single-camera intra-tracking
# ══════════════════════════════════════════════════════════════════════════════

def track_camera(model: YOLO, cam_short: str, img_dir: Path,
                 frame_start: int, frame_end: int, frame_offset: int,
                 tracker_cfg: str, conf: float, imgsz: int) -> list[str]:
    """Run YOLO OBB tracking on one camera's frame sequence.
    Returns MOT16 prediction lines."""
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None  # reset tracker state between cameras

    mot_lines = []
    img_paths = sorted(
        [p for p in img_dir.glob(f"{cam_short}_frame_*.png")
         if frame_start <= int(p.stem.split("_frame_")[1]) <= frame_end],
        key=lambda p: int(p.stem.split("_frame_")[1]),
    )
    import cv2
    for img_path in img_paths:
        frame_num = int(img_path.stem.split("_frame_")[1])
        seq_idx   = frame_num - frame_offset
        img       = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model.track(img, tracker=tracker_cfg, persist=True,
                              conf=conf, imgsz=imgsz, verbose=False)
        result  = results[0]
        if result.obb is not None and result.obb.id is not None:
            ids     = result.obb.id.cpu().numpy().astype(int)
            clss    = result.obb.cls.cpu().numpy().astype(int)
            confs_  = result.obb.conf.cpu().numpy()
            corners = result.obb.xyxyxyxy.cpu().numpy()
            for i in range(len(ids)):
                x1 = int(corners[i, :, 0].min()); y1 = int(corners[i, :, 1].min())
                x2 = int(corners[i, :, 0].max()); y2 = int(corners[i, :, 1].max())
                mot_lines.append(
                    f"{seq_idx},{ids[i]},{x1},{y1},{x2-x1},{y2-y1},"
                    f"{confs_[i]:.4f},{clss[i]},-1,-1"
                )
    return mot_lines


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def iou_distance(gt_rows, pred_rows) -> np.ndarray:
    def tlbr(rows):
        arr = np.array([[r[2], r[3], r[2]+r[4], r[3]+r[5]] for r in rows], float)
        return arr
    g = tlbr(gt_rows); p = tlbr(pred_rows)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
    ag = (g[:,2]-g[:,0])*(g[:,3]-g[:,1])
    ap = (p[:,2]-p[:,0])*(p[:,3]-p[:,1])
    union = ag[:, None] + ap[None, :] - inter
    return 1.0 - np.where(union > 0, inter/union, 0.0)


def evaluate_mot(pred_rows: list, gt_rows: list,
                 frame_start: int, frame_end: int,
                 frame_offset: int, max_dist: float = 0.5) -> dict | None:
    if not MM_OK:
        return None
    pred_bf = rows_by_frame(pred_rows)
    gt_bf   = rows_by_frame(gt_rows)
    acc = mm.MOTAccumulator(auto_id=True)
    for fn in range(frame_start, frame_end + 1):
        si = fn - frame_offset
        g = gt_bf.get(si, []);   p = pred_bf.get(si, [])
        g_ids = [r[1] for r in g]; p_ids = [r[1] for r in p]
        if g and p:
            dist = iou_distance(g, p)
            dist = np.where(dist <= max_dist, dist, np.nan)
        else:
            dist = np.empty((len(g_ids), len(p_ids)))
        acc.update(g_ids, p_ids, dist)
    mh = mm.metrics.create()
    try:
        s   = mh.compute(acc, metrics=MOT_METRICS, name="r")
        return {k: (float(s.loc["r", k])
                    if isinstance(s.loc["r", k], (int, float, np.floating))
                    else int(s.loc["r", k]))
                for k in MOT_METRICS}
    except Exception as e:
        print(f"      motmetrics error: {e}")
        return None


def evaluate_global(pred_rows: list, gt_rows: list) -> dict | None:
    if not MM_OK:
        return None
    pred_bf = rows_by_frame(pred_rows)
    gt_bf   = rows_by_frame(gt_rows)
    all_f   = sorted(set(pred_bf) | set(gt_bf))
    acc     = mm.MOTAccumulator(auto_id=True)
    for f in all_f:
        g = gt_bf.get(f, []); p = pred_bf.get(f, [])
        g_ids = [r[1] for r in g]; p_ids = [r[1] for r in p]
        if g and p:
            dist = iou_distance(g, p)
            dist = np.where(1 - dist >= 0.5, 1 - (1-dist), np.nan)
            dist = iou_distance(g, p)
            dist = np.where(dist <= 0.5, dist, np.nan)
        else:
            dist = np.empty((len(g_ids), len(p_ids)))
        acc.update(g_ids, p_ids, dist)
    mh = mm.metrics.create()
    try:
        s = mh.compute(acc, metrics=MOT_METRICS, name="r")
        return {k: (float(s.loc["r", k])
                    if isinstance(s.loc["r", k], (int, float, np.floating))
                    else int(s.loc["r", k]))
                for k in MOT_METRICS}
    except Exception as e:
        print(f"      global metrics error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Offline MCMOT association methods
# ══════════════════════════════════════════════════════════════════════════════

class Tracklet:
    __slots__ = ("cam", "local_id", "class_id", "frames", "bboxes",
                 "mean_x", "global_id")
    def __init__(self, cam, local_id, class_id):
        self.cam = cam; self.local_id = local_id; self.class_id = class_id
        self.frames: set = set(); self.bboxes: dict = {}
        self.mean_x = 0.0; self.global_id = -1
    def finalize(self):
        if self.bboxes:
            self.mean_x = float(np.mean([b[0]+b[2]/2 for b in self.bboxes.values()]))


def build_tracklets(pred_rows: list, cam: str) -> dict[int, Tracklet]:
    tlets: dict[int, Tracklet] = {}
    for frame, oid, x, y, w, h, conf, cls in pred_rows:
        if oid not in tlets:
            tlets[oid] = Tracklet(cam, oid, cls)
        t = tlets[oid]
        t.frames.add(frame); t.bboxes[frame] = (x, y, w, h, conf)
        if cls >= 0:
            t.class_id = cls
    for t in tlets.values():
        votes = [t.class_id for _ in t.bboxes if t.class_id >= 0]
        if votes:
            t.class_id = Counter(votes).most_common(1)[0][0]
        t.finalize()
    return tlets


def jaccard(a: Tracklet, b: Tracklet) -> float:
    inter = len(a.frames & b.frames); union = len(a.frames | b.frames)
    return inter / union if union > 0 else 0.0


def greedy_merge(all_tlets: list[Tracklet], cost_fn, max_cost: float = 0.95,
                  next_gid: list[int] | None = None):
    n = len(all_tlets)
    cost = np.full((n, n), 1e9)
    for i in range(n):
        for j in range(i+1, n):
            if all_tlets[i].cam == all_tlets[j].cam:
                continue
            c = cost_fn(all_tlets[i], all_tlets[j])
            cost[i, j] = cost[j, i] = c

    assigned = [-1] * n
    groups: list[set] = []; cam_in_group: list[set] = []
    pairs = sorted((cost[i,j], i, j)
                   for i in range(n) for j in range(i+1, n)
                   if cost[i,j] < max_cost)
    for c, i, j in pairs:
        gi, gj = assigned[i], assigned[j]
        ci, cj = all_tlets[i].cam, all_tlets[j].cam
        if gi == -1 and gj == -1:
            g = len(groups); groups.append({i,j}); cam_in_group.append({ci,cj})
            assigned[i] = assigned[j] = g
        elif gi >= 0 and gj == -1:
            if cj not in cam_in_group[gi]:
                groups[gi].add(j); cam_in_group[gi].add(cj); assigned[j] = gi
        elif gi == -1 and gj >= 0:
            if ci not in cam_in_group[gj]:
                groups[gj].add(i); cam_in_group[gj].add(ci); assigned[i] = gj
    for i in range(n):
        if assigned[i] == -1:
            g = len(groups); groups.append({i}); cam_in_group.append({all_tlets[i].cam})
            assigned[i] = g
    if next_gid is None:
        next_gid = [1]
    for grp in groups:
        for idx in grp:
            all_tlets[idx].global_id = next_gid[0]
        next_gid[0] += 1


def run_offline_method(per_cam: dict[str, dict], method: str):
    """Apply one offline association strategy in-place."""
    all_tlets_flat = [t for tlets in per_cam.values() for t in tlets.values()]

    if method == "no_assoc":
        cam_list = list(per_cam.keys())
        for i, cam in enumerate(cam_list):
            shift = (i + 1) * 1000
            for t in per_cam[cam].values():
                t.global_id = t.local_id + shift
        return

    if method == "class_only":
        # Frame-level rank match (offline variant)
        l2g: dict[tuple, int] = {}
        gid = [1]
        all_frames: set = set()
        cfd: dict[tuple, list] = defaultdict(list)
        for cam, tlets in per_cam.items():
            for lid, t in tlets.items():
                all_frames |= t.frames
                for f, (x, y, w, h, conf) in t.bboxes.items():
                    cfd[(cam, f)].append((x + w/2, lid, t.class_id))
        for f in sorted(all_frames):
            for cls in range(NUM_CLASSES):
                cam_dets: dict[str, list] = {}
                for cam, _ in CAMS:
                    dets = [(xc, lid) for xc, lid, c in cfd.get((cam, f), [])
                            if c == cls]
                    dets.sort()
                    if dets:
                        cam_dets[cam] = dets
                if len(cam_dets) < 2:
                    for cam, dets in cam_dets.items():
                        for _, lid in dets:
                            if (cam, lid) not in l2g:
                                l2g[(cam, lid)] = gid[0]; gid[0] += 1
                    continue
                max_r = max(len(d) for d in cam_dets.values())
                for rank in range(max_r):
                    rank_dets = {cam: dets[rank] for cam, dets in cam_dets.items()
                                 if rank < len(dets)}
                    g = None
                    for cam, (_, lid) in rank_dets.items():
                        if (cam, lid) in l2g:
                            g = l2g[(cam, lid)]; break
                    if g is None:
                        g = gid[0]; gid[0] += 1
                    for cam, (_, lid) in rank_dets.items():
                        l2g[(cam, lid)] = g
        for cam, tlets in per_cam.items():
            for lid, t in tlets.items():
                t.global_id = l2g.get((cam, lid), -1)
                if t.global_id < 0:
                    t.global_id = gid[0]; gid[0] += 1
        return

    # Tracklet-level methods
    next_gid = [1]
    for cls in range(NUM_CLASSES):
        by_cam = {cam: sorted([t for t in tlets.values() if t.class_id == cls],
                              key=lambda t: t.mean_x)
                  for cam, tlets in per_cam.items()}
        tlets_in_cls = [t for ts in by_cam.values() for t in ts]
        if not tlets_in_cls:
            continue

        rank_of: dict[int, float] = {}
        for cam, ts in by_cam.items():
            n = len(ts)
            for r, t in enumerate(ts):
                rank_of[id(t)] = r / max(n-1, 1)

        if method == "trk_temporal":
            cost_fn = lambda a, b: 1.0 - jaccard(a, b)
        elif method == "trk_spatial":
            cost_fn = lambda a, b: abs(rank_of.get(id(a),0) - rank_of.get(id(b),0))
        else:  # trk_combined
            cost_fn = lambda a, b: (0.5*(1-jaccard(a,b)) +
                                    0.5*abs(rank_of.get(id(a),0)-rank_of.get(id(b),0)))
        greedy_merge(tlets_in_cls, cost_fn, next_gid=next_gid)


def tracklets_to_mot(per_cam: dict) -> list[tuple]:
    rows = []
    for cam_idx, (cam_short, _) in enumerate(CAMS):
        tlets = per_cam.get(cam_short, {})
        shift = cam_idx * CAM_FRAME_STRIDE
        for t in tlets.values():
            for f in sorted(t.frames):
                x, y, w, h, conf = t.bboxes[f]
                rows.append((f + shift, t.global_id, x, y, w, h, conf, t.class_id))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Online real-time associators
# ══════════════════════════════════════════════════════════════════════════════

class OnlineAssoc:
    def __init__(self):
        self.l2g: dict[tuple, int] = {}
        self.gcls: dict[int, int]  = {}
        self.next = 1
    def _new(self, cls):
        g = self.next; self.next += 1; self.gcls[g] = cls; return g
    def _resolve(self, pairs, cls):
        for cam, lid in pairs:
            g = self.l2g.get((cam, lid))
            if g is not None: return g
        return self._new(cls)
    def associate(self, cam_tracks: dict) -> dict:
        raise NotImplementedError
    def get(self, cam, lid): return self.l2g.get((cam, lid), -1)


class ClassRankAssoc(OnlineAssoc):
    def associate(self, cam_tracks):
        cg: dict[int, dict] = defaultdict(lambda: defaultdict(list))
        for cam, tracks in cam_tracks.items():
            for lid, cid, xc, _ in tracks:
                cg[cid][cam].append((xc, lid))
        for cid, cam_dets in cg.items():
            for cam in cam_dets: cam_dets[cam].sort()
            for rank in range(max(len(d) for d in cam_dets.values())):
                pairs = [(cam, dets[rank][1]) for cam, dets in cam_dets.items()
                         if rank < len(dets)]
                g = self._resolve(pairs, cid)
                for cam, lid in pairs: self.l2g[(cam, lid)] = g
        return self.l2g


class ClassHungarianAssoc(OnlineAssoc):
    THRESH = 0.30
    def associate(self, cam_tracks):
        if not SCIPY_OK:
            return ClassRankAssoc.associate(self, cam_tracks)
        cg: dict[int, dict] = defaultdict(lambda: defaultdict(list))
        for cam, tracks in cam_tracks.items():
            for lid, cid, xc, _ in tracks:
                cg[cid][cam].append((xc / W, lid))
        for cid, cam_dets in cg.items():
            all_d = [(cam, lid, nxc) for cam, dets in cam_dets.items()
                     for nxc, lid in dets]
            known   = [(cam, lid, nxc) for cam, lid, nxc in all_d
                       if (cam, lid) in self.l2g]
            unknown = [(cam, lid, nxc) for cam, lid, nxc in all_d
                       if (cam, lid) not in self.l2g]
            if not unknown: continue
            if not known:
                # rank fallback
                cs: dict = defaultdict(list)
                for cam, lid, nxc in all_d:
                    cs[cam].append((nxc, lid))
                for cam in cs: cs[cam].sort()
                max_n = max(len(d) for d in cs.values())
                for rank in range(max_n):
                    pairs = [(cam, dets[rank][1]) for cam, dets in cs.items()
                             if rank < len(dets)]
                    g = self._resolve(pairs, cid)
                    for cam, lid in pairs: self.l2g[(cam, lid)] = g
                continue
            known_gids = [self.l2g[(cam, lid)] for cam, lid, _ in known]
            ux = np.array([nxc for _, _, nxc in unknown])
            kx = np.array([nxc for _, _, nxc in known])
            cost = np.abs(ux[:, None] - kx[None, :])
            for ui, (uc, _, _) in enumerate(unknown):
                for ki, (kc, _, _) in enumerate(known):
                    if uc == kc: cost[ui, ki] = 1.0
            rows, cols = linear_sum_assignment(cost)
            matched = set()
            for r, c in zip(rows, cols):
                if cost[r, c] < self.THRESH:
                    uc, ul, _ = unknown[r]
                    self.l2g[(uc, ul)] = known_gids[c]; matched.add(r)
            for ui, (uc, ul, _) in enumerate(unknown):
                if ui not in matched:
                    self.l2g[(uc, ul)] = self._new(cid)
        return self.l2g


class ClassSmoothAssoc(ClassRankAssoc):
    WINDOW = 8
    def __init__(self):
        super().__init__()
        self.hist: dict[tuple, list] = defaultdict(list)
    def associate(self, cam_tracks):
        super().associate(cam_tracks)
        active = {(cam, lid) for cam, tracks in cam_tracks.items()
                  for lid, _, _, _ in tracks}
        for key in active:
            raw = self.l2g.get(key, -1)
            if raw < 0: continue
            h = self.hist[key]; h.append(raw)
            if len(h) > self.WINDOW: h.pop(0)
            self.l2g[key] = Counter(h).most_common(1)[0][0]
        return self.l2g


ONLINE_ASSOC = {
    "class_rank":   ClassRankAssoc,
    "class_iou":    ClassHungarianAssoc,
    "class_smooth": ClassSmoothAssoc,
}


def infer_one_camera(args):
    model, img_path, tracker_cfg, conf, imgsz = args
    import cv2
    img = cv2.imread(str(img_path))
    if img is None:
        return None, []
    results = model.track(img, tracker=tracker_cfg, persist=True,
                          conf=conf, imgsz=imgsz, verbose=False)
    r = results[0]
    tracks = []
    if r.obb is not None and r.obb.id is not None:
        ids = r.obb.id.cpu().numpy().astype(int)
        cls = r.obb.cls.cpu().numpy().astype(int)
        corn = r.obb.xyxyxyxy.cpu().numpy()
        for i in range(len(ids)):
            xc = float(corn[i, :, 0].mean())
            tracks.append((int(ids[i]), int(cls[i]), xc, corn[i]))
    return r, tracks


def run_online_method(model_path: Path, dataset_cfg: dict,
                      tracker_cfg: str, method: str,
                      conf: float, imgsz: int,
                      device: str, use_threads: bool) -> dict:
    """Run one online MCMOT combination. Returns metrics + timing dict."""
    img_dir     = dataset_cfg["ds_dir"] / "images"
    frame_start = dataset_cfg["frame_start"]
    frame_end   = dataset_cfg["frame_end"]
    frame_offset = dataset_cfg["frame_offset"]
    cam_gt_dir  = dataset_cfg["ds_dir"]
    n_frames    = frame_end - frame_start + 1

    models = {cam: YOLO(str(model_path)) for cam, _ in CAMS}

    assoc: OnlineAssoc = ONLINE_ASSOC[method]()
    mot_rows: list[str] = []

    t0 = time.perf_counter()
    for frame_num in range(frame_start, frame_end + 1):
        img_paths = {cam: img_dir / f"{cam}_frame_{frame_num:04d}.png"
                     for cam, _ in CAMS}
        # Parallel or sequential inference per camera
        cam_results: dict = {}
        if use_threads:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = {cam: pool.submit(infer_one_camera,
                                         (models[cam], img_paths[cam],
                                          tracker_cfg, conf, imgsz))
                        for cam, _ in CAMS}
                for cam, fut in futs.items():
                    cam_results[cam] = fut.result()
        else:
            for cam, _ in CAMS:
                cam_results[cam] = infer_one_camera(
                    (models[cam], img_paths[cam], tracker_cfg, conf, imgsz))

        cam_tracks = {cam: tracks for cam, (_, tracks) in cam_results.items()}
        assoc.associate(cam_tracks)
        seq_idx = frame_num - frame_offset
        for cam_idx, (cam, _) in enumerate(CAMS):
            merged_f = seq_idx + cam_idx * CAM_FRAME_STRIDE
            for lid, cls_id, _, corners in cam_tracks.get(cam, []):
                gid = assoc.get(cam, lid)
                x1 = int(corners[:, 0].min()); y1 = int(corners[:, 1].min())
                x2 = int(corners[:, 0].max()); y2 = int(corners[:, 1].max())
                mot_rows.append(
                    f"{merged_f},{gid},{x1},{y1},{x2-x1},{y2-y1},1.0,{cls_id},-1,-1")

    elapsed = time.perf_counter() - t0
    fps     = n_frames / elapsed

    # Build global GT
    gt_rows: list = []
    for cam_idx, (cam_short, cam_sub) in enumerate(CAMS):
        shift   = cam_idx * CAM_FRAME_STRIDE
        gt_path = cam_gt_dir / "mot_obb" / cam_sub / "gt" / "gt.txt"
        for row in parse_mot(gt_path):
            gt_rows.append((row[0] + shift,) + row[1:])

    pred_parsed = []
    for l in mot_rows:
        p = l.split(",")
        pred_parsed.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                            float(p[4]), float(p[5]), float(p[6]), int(p[7])))

    metrics = evaluate_global(pred_parsed, gt_rows) if gt_rows else None
    return {
        "fps":           round(fps, 2),
        "wall_time_s":   round(elapsed, 1),
        "frames":        n_frames,
        "realtime_25fps": fps >= 25.0,
        "metrics":       metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dedicated FPS benchmarks  (the two numbers the server run is mainly for)
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_single_cam_fps(model_path: Path, img_dir: Path,
                              cam_short: str,
                              frame_start: int, frame_end: int,
                              tracker_cfg: str,
                              conf: float, imgsz: int,
                              n_warmup: int = 5) -> dict:
    """Measure pure single-camera throughput: YOLO+tracker on one view only.

    n_warmup frames are run first (not timed) to fill the GPU pipeline and
    initialise the tracker state so the timing reflects steady-state speed.
    Returns fps, ms_per_frame, n_frames_timed.
    """
    import cv2
    model = YOLO(str(model_path))
    img_paths = sorted(
        [p for p in img_dir.glob(f"{cam_short}_frame_*.png")
         if frame_start <= int(p.stem.split("_frame_")[1]) <= frame_end],
        key=lambda p: int(p.stem.split("_frame_")[1]),
    )
    # warm-up
    for p in img_paths[:n_warmup]:
        img = cv2.imread(str(p))
        if img is not None:
            model.track(img, tracker=tracker_cfg, persist=True,
                        conf=conf, imgsz=imgsz, verbose=False)

    # timed run
    timed_paths = img_paths[n_warmup:]
    t0 = time.perf_counter()
    for p in timed_paths:
        img = cv2.imread(str(p))
        if img is not None:
            model.track(img, tracker=tracker_cfg, persist=True,
                        conf=conf, imgsz=imgsz, verbose=False)
    elapsed = time.perf_counter() - t0
    n = len(timed_paths)
    fps = n / elapsed if elapsed > 0 else 0.0
    return {
        "fps":            round(fps, 2),
        "ms_per_frame":   round(1000 / fps, 1) if fps > 0 else 0,
        "n_frames_timed": n,
        "n_warmup":       n_warmup,
        "realtime_25fps": fps >= 25.0,
    }


def benchmark_multicam_fps(model_path: Path, img_dir: Path,
                           frame_start: int, frame_end: int,
                           tracker_cfg: str, method: str,
                           conf: float, imgsz: int,
                           use_threads: bool,
                           n_warmup: int = 5) -> dict:
    """Measure 4-camera MCMOT throughput without recording any output files.

    fps here means: scene-frames (time steps) per second, each step
    processing all 4 camera views + running the cross-camera associator.
    Directly comparable to single-cam fps — same denominator (scene frames).
    """
    import cv2
    models = {cam: YOLO(str(model_path)) for cam, _ in CAMS}
    assoc: OnlineAssoc = ONLINE_ASSOC[method]()

    all_frame_nums = list(range(frame_start, frame_end + 1))
    warmup_frames  = all_frame_nums[:n_warmup]
    timed_frames   = all_frame_nums[n_warmup:]

    def process_frame(frame_num: int):
        cam_tracks: dict = {}
        if use_threads:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = {cam: pool.submit(
                    infer_one_camera,
                    (models[cam],
                     img_dir / f"{cam}_frame_{frame_num:04d}.png",
                     tracker_cfg, conf, imgsz))
                    for cam, _ in CAMS}
                for cam, fut in futs.items():
                    _, tracks = fut.result()
                    cam_tracks[cam] = tracks
        else:
            for cam, _ in CAMS:
                _, tracks = infer_one_camera(
                    (models[cam],
                     img_dir / f"{cam}_frame_{frame_num:04d}.png",
                     tracker_cfg, conf, imgsz))
                cam_tracks[cam] = tracks
        assoc.associate(cam_tracks)

    for fn in warmup_frames:
        process_frame(fn)

    t0 = time.perf_counter()
    for fn in timed_frames:
        process_frame(fn)
    elapsed = time.perf_counter() - t0
    n = len(timed_frames)
    fps = n / elapsed if elapsed > 0 else 0.0
    return {
        "fps":            round(fps, 2),
        "ms_per_frame":   round(1000 / fps, 1) if fps > 0 else 0,
        "n_frames_timed": n,
        "n_warmup":       n_warmup,
        "realtime_25fps": fps >= 25.0,
        "note":           "fps = scene-frames/s (4 cameras + MCMOT per frame)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="MCMOT server benchmark — outputs JSON with all results")
    ap.add_argument("--data-root", default="/Users/awthura/OVGU/AMS/synthetic_polybags",
                    help="Root dir containing synth_dataset_val/ etc.")
    ap.add_argument("--model",   default="../training/weights_synth_hires.pt")
    ap.add_argument("--device",  default="auto",
                    help="cuda / cuda:0 / cpu / mps / auto")
    ap.add_argument("--imgsz",   type=int,   default=1920)
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--datasets", nargs="+", default=["val", "test"])
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "botsort"])
    ap.add_argument("--online",   nargs="+", default=ONLINE_METHODS)
    ap.add_argument("--offline",  nargs="+", default=OFFLINE_METHODS)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    data_root  = Path(args.data_root)
    model_path = Path(args.model)
    device     = resolve_device(args.device)
    # Sequential inference on GPU (faster — one model call per GPU is optimal);
    # threaded on CPU to exploit multiple cores.
    use_threads = device in ("cpu", "mps")

    DATASETS = {
        "val":  {"ds_dir": data_root / "synth_dataset_val",
                 "frame_start": 1000, "frame_end": 1250, "frame_offset": 999},
        "test": {"ds_dir": data_root / "synth_dataset_test",
                 "frame_start": 1500, "frame_end": 1750, "frame_offset": 1499},
        "train":{"ds_dir": data_root / "synth_dataset_mcmot",
                 "frame_start": 100,  "frame_end": 599,  "frame_offset": 99},
    }

    print(f"\n{'='*64}")
    print(f"  MCMOT Server Benchmark")
    print(f"  Device: {device}   Datasets: {args.datasets}")
    print(f"  Trackers: {args.trackers}")
    print(f"  Model: {model_path.name}   imgsz={args.imgsz}")
    print(f"{'='*64}\n")

    sys_info = get_system_info(device)
    print(f"  Host:   {sys_info['host']}")
    print(f"  GPU:    {sys_info['gpu_name']}")
    print(f"  CUDA:   {sys_info['cuda_version']}")
    print(f"  PyTorch:{sys_info['torch']}\n")

    results = {
        "meta": {
            "generated":          datetime.now().isoformat(),
            "system":             sys_info,
            "model":              model_path.name,
            "imgsz":              args.imgsz,
            "conf":               args.conf,
            "datasets":           args.datasets,
            "trackers":           args.trackers,
            "online_methods":     args.online,
            "offline_methods":    args.offline,
            "parallel_inference": use_threads,
        },
        # Key comparison — always written even without full evaluation
        "fps_benchmark": {
            "note": (
                "single_cam: YOLO+tracker on ONE camera view alone (steady-state). "
                "multicam_mcmot: all 4 cameras + cross-camera association per frame. "
                "Both fps values use the same denominator: scene-frames per second."
            ),
            "single_cam":    {},   # tracker → dataset → fps dict
            "multicam_mcmot":{},   # method → tracker → dataset → fps dict
        },
        "single_camera": {},
        "offline_mcmot":  {},
        "online_mcmot":   {},
    }

    # ── Step 0: FPS benchmarks ────────────────────────────────────────────────
    print(">>> Step 0 / 4 — FPS benchmarks (single-cam vs 4-cam MCMOT)")
    # Use front camera (cam_01_front) as the representative single-cam view
    FPS_CAM = "front"
    for tracker in args.trackers:
        results["fps_benchmark"]["single_cam"].setdefault(tracker, {})
        for dataset in args.datasets:
            ds_cfg = DATASETS[dataset]
            if not ds_cfg["ds_dir"].exists():
                continue
            img_dir = ds_cfg["ds_dir"] / "images"
            print(f"\n  [single-cam / {tracker} / {dataset}]  (front view, "
                  f"{ds_cfg['frame_end']-ds_cfg['frame_start']+1} frames)")
            r = benchmark_single_cam_fps(
                model_path, img_dir, FPS_CAM,
                ds_cfg["frame_start"], ds_cfg["frame_end"],
                f"{tracker}.yaml", args.conf, args.imgsz)
            results["fps_benchmark"]["single_cam"][tracker][dataset] = r
            rt = "✓ REAL-TIME" if r["realtime_25fps"] else "  below 25fps"
            print(f"    {r['fps']:.1f} fps  ({r['ms_per_frame']:.1f} ms/frame)  {rt}")

    print()
    FPS_ONLINE_METHOD = "class_rank"   # fastest; used for the baseline comparison
    for tracker in args.trackers:
        results["fps_benchmark"]["multicam_mcmot"].setdefault(
            FPS_ONLINE_METHOD, {}).setdefault(tracker, {})
        for dataset in args.datasets:
            ds_cfg = DATASETS[dataset]
            if not ds_cfg["ds_dir"].exists():
                continue
            img_dir = ds_cfg["ds_dir"] / "images"
            print(f"  [4-cam MCMOT ({FPS_ONLINE_METHOD}) / {tracker} / {dataset}]")
            r = benchmark_multicam_fps(
                model_path, img_dir,
                ds_cfg["frame_start"], ds_cfg["frame_end"],
                f"{tracker}.yaml", FPS_ONLINE_METHOD,
                args.conf, args.imgsz, use_threads)
            results["fps_benchmark"]["multicam_mcmot"] \
                .setdefault(FPS_ONLINE_METHOD, {}) \
                .setdefault(tracker, {})[dataset] = r
            rt = "✓ REAL-TIME" if r["realtime_25fps"] else "  below 25fps"
            print(f"    {r['fps']:.1f} fps  ({r['ms_per_frame']:.1f} ms/frame)  {rt}")

    # ── Step 1: Intra-camera tracking ─────────────────────────────────────────
    print("\n>>> Step 1 / 4 — Intra-camera tracking")
    # Store pred_rows per (tracker, dataset, cam) for offline use
    pred_store: dict = {}
    model_cache: dict[str, YOLO] = {}

    for tracker in args.trackers:
        results["single_camera"].setdefault(tracker, {})
        for dataset in args.datasets:
            ds_cfg = DATASETS[dataset]
            if not ds_cfg["ds_dir"].exists():
                print(f"  SKIP {dataset}: directory not found")
                continue

            print(f"\n  [{tracker} / {dataset}]")
            model = YOLO(str(model_path))
            model_cache[(tracker, dataset)] = model
            accs = []; names = []

            for cam_short, cam_sub in CAMS:
                t0 = time.perf_counter()
                mot_lines = track_camera(
                    model, cam_short,
                    ds_cfg["ds_dir"] / "images",
                    ds_cfg["frame_start"], ds_cfg["frame_end"],
                    ds_cfg["frame_offset"],
                    f"{tracker}.yaml", args.conf, args.imgsz)
                elapsed = time.perf_counter() - t0
                n_frames = ds_cfg["frame_end"] - ds_cfg["frame_start"] + 1
                fps      = n_frames / elapsed

                pred_rows = parse_mot_from_lines(mot_lines)
                pred_store[(tracker, dataset, cam_short)] = pred_rows

                cam_metrics = None
                if not args.skip_eval and MM_OK:
                    gt_path = ds_cfg["ds_dir"] / "mot_obb" / cam_sub / "gt" / "gt.txt"
                    gt_rows = parse_mot(gt_path)
                    if gt_rows:
                        cam_metrics = evaluate_mot(
                            pred_rows, gt_rows,
                            ds_cfg["frame_start"], ds_cfg["frame_end"],
                            ds_cfg["frame_offset"])

                out = {"fps": round(fps, 2), "n_detections": len(mot_lines)}
                if cam_metrics:
                    out["metrics"] = cam_metrics
                results["single_camera"][tracker] \
                    .setdefault(dataset, {})[cam_short] = out
                print(f"    {cam_short}: {len(mot_lines)} dets  {fps:.1f} fps"
                      + (f"  MOTA={cam_metrics['mota']*100:.1f}%"
                         f"  IDF1={cam_metrics['idf1']*100:.1f}%"
                         if cam_metrics else ""))

    # ── Step 2: Offline MCMOT ─────────────────────────────────────────────────
    print("\n>>> Step 2 / 4 — Offline MCMOT association")
    for tracker in args.trackers:
        results["offline_mcmot"].setdefault(tracker, {})
        for dataset in args.datasets:
            ds_cfg = DATASETS[dataset]
            if not ds_cfg["ds_dir"].exists():
                continue
            print(f"\n  [{tracker} / {dataset}]")
            results["offline_mcmot"][tracker].setdefault(dataset, {})

            # Build tracklets from stored pred_rows
            per_cam: dict = {}
            for cam_short, _ in CAMS:
                rows = pred_store.get((tracker, dataset, cam_short), [])
                per_cam[cam_short] = build_tracklets(rows, cam_short)

            # Build global GT once
            gt_rows_global: list = []
            for ci, (cs, cam_sub) in enumerate(CAMS):
                shift = ci * CAM_FRAME_STRIDE
                gt_path = ds_cfg["ds_dir"] / "mot_obb" / cam_sub / "gt" / "gt.txt"
                for row in parse_mot(gt_path):
                    gt_rows_global.append((row[0]+shift,) + row[1:])

            for method in args.offline:
                import copy
                per_cam_copy = {cam: {lid: copy.copy(t) for lid, t in tlets.items()}
                                for cam, tlets in per_cam.items()}
                run_offline_method(per_cam_copy, method)
                pred_global = tracklets_to_mot(per_cam_copy)
                metrics = evaluate_global(pred_global, gt_rows_global) \
                          if (gt_rows_global and not args.skip_eval) else None
                results["offline_mcmot"][tracker][dataset][method] = {
                    "metrics": metrics}
                if metrics:
                    print(f"    {method:<20} MOTA={metrics['mota']*100:.1f}%"
                          f"  IDF1={metrics['idf1']*100:.1f}%"
                          f"  IDSW={metrics['num_switches']}")

    # ── Step 3: Online / real-time ────────────────────────────────────────────
    print("\n>>> Step 3 / 4 — Online real-time MCMOT")
    for method in args.online:
        if method == "class_iou" and not SCIPY_OK:
            print(f"  SKIP {method}: scipy not installed"); continue
        results["online_mcmot"].setdefault(method, {})
        for tracker in args.trackers:
            results["online_mcmot"][method].setdefault(tracker, {})
            for dataset in args.datasets:
                ds_cfg = DATASETS[dataset]
                if not ds_cfg["ds_dir"].exists():
                    continue
                print(f"\n  [{method} / {tracker} / {dataset}]")
                res = run_online_method(
                    model_path, ds_cfg, f"{tracker}.yaml", method,
                    args.conf, args.imgsz, device, use_threads)
                results["online_mcmot"][method][tracker][dataset] = res
                m = res.get("metrics")
                rt = "✓ REAL-TIME" if res["realtime_25fps"] else f"  {res['fps']:.1f} fps"
                print(f"    {rt}  ({res['wall_time_s']:.0f}s)"
                      + (f"  MOTA={m['mota']*100:.1f}%  IDF1={m['idf1']*100:.1f}%"
                         f"  IDSW={m['num_switches']}" if m else ""))

    # ── Step 4: Save JSON ─────────────────────────────────────────────────────
    print("\n>>> Step 4 / 4 — Saving results")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*64}")
    print(f"  Results saved → {out_path.resolve()}")
    print(f"  Copy to local machine and run:")
    print(f"    python generate_report.py --from-json {out_path.name}")
    print(f"{'='*64}\n")


def parse_mot_from_lines(lines: list[str]) -> list[tuple]:
    rows = []
    for l in lines:
        p = l.split(",")
        rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                     float(p[4]), float(p[5]), float(p[6]), int(p[7])))
    return rows


if __name__ == "__main__":
    main()
