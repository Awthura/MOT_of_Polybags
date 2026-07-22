#!/usr/bin/env python3
"""
tracking/associate_cameras.py

Inter-camera association (MTMCT Stage 2) with multiple strategies.

Pipeline (hierarchical TbD — survey §2.1, §3.3.1):
  Intra-camera tracklets (from run_tracking.py)
    → Feature extraction (class_id = color, spatial position)
    → Inter-camera data association (Hungarian / greedy / frame-level)
    → Global ID assignment
    → Merged MOT files → MCMOT metrics

NOTE: pred.txt must contain class_id in column 7.
      Re-run run_tracking.py if you have old files with class=-1.

Methods
-------
  class_only        Frame-by-frame: match same-class tracks across cameras.
                    Simple 1:1 within class; ties broken by x-order.
                    Closest to a naive approach or the POM-style occupancy map
                    evaluated per frame.

  tracklet_temporal Tracklet-level: class gate + temporal Jaccard distance
                    as cost for Hungarian assignment. High overlap → likely
                    same object (cameras are synchronized, overlapping FOVs).
                    Models temporal coherence (JPDAF-inspired).

  tracklet_spatial  Tracklet-level: class gate + mean x-position rank across
                    cameras. Bags on a conveyor have consistent left-to-right
                    order in all views.
                    Models spatial consistency (POM-inspired, no calibration).

  tracklet_combined Tracklet-level: class gate + weighted combination of
                    temporal Jaccard + spatial rank agreement.
                    Closest to LMGP / DyGLIP without deep features.

  no_assoc          No inter-camera step at all: each camera keeps its local
                    IDs. Baseline to show what happens without MCMOT.

Usage
-----
  # re-generate pred.txt first (must have class_id):
  python run_tracking.py --tracker bytetrack --dataset val

  # then benchmark all methods:
  python associate_cameras.py --tracker bytetrack --dataset val --method all

  # single method:
  python associate_cameras.py --tracker bytetrack --dataset val --method tracklet_combined

  # all tracker × dataset × method combinations:
  python associate_cameras.py --all
"""

import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    sys.exit("scipy not installed — pip install scipy")

try:
    import motmetrics as mm
except ImportError:
    sys.exit("motmetrics not installed — pip install motmetrics")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
BASE     = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
OUT_ROOT = REPO / "tracking_results"

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]
CAM_IDX = {cam: i for i, (cam, _) in enumerate(CAMS)}
CAM_FRAME_STRIDE = 10000   # frame offset per camera in the merged global file

DATASETS = {
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

METHODS = ["no_assoc", "class_only", "tracklet_temporal",
           "tracklet_spatial", "tracklet_combined"]
NUM_CLASSES = 7
INF = 1e9

MOT_METRICS = ["num_frames", "mota", "motp", "idf1",
               "num_switches", "mostly_tracked", "mostly_lost",
               "num_false_positives", "num_misses"]


# ── Data structures ────────────────────────────────────────────────────────────

class Tracklet:
    __slots__ = ("cam", "local_id", "class_id", "frames", "bboxes",
                 "mean_x", "global_id")

    def __init__(self, cam: str, local_id: int, class_id: int):
        self.cam       = cam
        self.local_id  = local_id
        self.class_id  = class_id
        self.frames: set[int] = set()
        self.bboxes: dict[int, tuple] = {}   # seq_idx → (x,y,w,h,conf)
        self.mean_x    = 0.0
        self.global_id = -1

    def finalize(self):
        if self.bboxes:
            self.mean_x = float(np.mean([b[0] + b[2] / 2
                                         for b in self.bboxes.values()]))


# ── Loaders ────────────────────────────────────────────────────────────────────

def _parse_mot_file(path: Path) -> list[tuple]:
    """Return list of (frame, id, x, y, w, h, conf, class_id) tuples."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split(",")
        frame    = int(p[0])
        obj_id   = int(p[1])
        x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
        conf     = float(p[6]) if len(p) > 6 else 1.0
        cls_id   = int(p[7]) if len(p) > 7 and p[7].strip() not in ("-1", "") else -1
        if conf < 0:
            continue
        rows.append((frame, obj_id, x, y, w, h, conf, cls_id))
    return rows


def load_tracklets(pred_path: Path, cam: str) -> dict[int, Tracklet]:
    rows = _parse_mot_file(pred_path)
    tlets: dict[int, Tracklet] = {}
    for frame, local_id, x, y, w, h, conf, cls_id in rows:
        if local_id not in tlets:
            tlets[local_id] = Tracklet(cam, local_id, cls_id)
        t = tlets[local_id]
        t.frames.add(frame)
        t.bboxes[frame] = (x, y, w, h, conf)
        if cls_id >= 0:
            t.class_id = cls_id   # update with real class if -1 before
    # Majority vote on class_id if inconsistent
    for t in tlets.values():
        cls_votes = [t.class_id for f, b in t.bboxes.items()
                     if t.class_id >= 0]
        if cls_votes:
            t.class_id = Counter(cls_votes).most_common(1)[0][0]
        t.finalize()
    return tlets


def load_gt(cam_sub: str, ds_dir: Path) -> list[tuple]:
    gt_path = ds_dir / "mot_obb" / cam_sub / "gt" / "gt.txt"
    return _parse_mot_file(gt_path)


# ── Temporal Jaccard ───────────────────────────────────────────────────────────

def jaccard(a: Tracklet, b: Tracklet) -> float:
    inter = len(a.frames & b.frames)
    union = len(a.frames | b.frames)
    return inter / union if union > 0 else 0.0


# ── Hungarian solver ───────────────────────────────────────────────────────────

def hungarian_match(cost: np.ndarray, max_cost: float = 0.95) -> list[tuple]:
    """Return list of (row, col) matched pairs with cost < max_cost."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] < max_cost]


# ══════════════════════════════════════════════════════════════════════════════
# Association methods
# ══════════════════════════════════════════════════════════════════════════════

def method_no_assoc(per_cam: dict[str, dict[int, Tracklet]]) -> int:
    """
    Baseline: no inter-camera step. Local IDs become global IDs but each
    camera uses its own independent namespace (shifted by cam_idx * 1000).
    """
    for cam, tlets in per_cam.items():
        shift = (CAM_IDX[cam] + 1) * 1000
        for t in tlets.values():
            t.global_id = t.local_id + shift
    return max(
        (t.global_id for tlets in per_cam.values() for t in tlets.values()),
        default=0
    ) + 1


def method_class_only(per_cam: dict[str, dict[int, Tracklet]]) -> int:
    """
    Frame-by-frame: for each frame, match same-class tracks across cameras.
    Within a color class, sort by x-center and match by rank (1st-to-1st, etc.)
    Maintain track→global_id mapping across frames for temporal consistency.
    """
    local_to_global: dict[tuple[str, int], int] = {}  # (cam, local_id) → global_id
    next_gid = [1]

    # Gather all frame indices across cameras
    all_frames: set[int] = set()
    for tlets in per_cam.values():
        for t in tlets.values():
            all_frames |= t.frames
    all_frames = sorted(all_frames)

    # Build lookup: (cam, frame) → list of (x_center, local_id, class_id)
    cam_frame_det: dict[tuple, list] = defaultdict(list)
    for cam, tlets in per_cam.items():
        for local_id, t in tlets.items():
            for frame, (x, y, w, h, conf) in t.bboxes.items():
                cam_frame_det[(cam, frame)].append(
                    (x + w / 2, local_id, t.class_id))

    for frame in all_frames:
        for cls_id in range(NUM_CLASSES):
            # Collect detections for this class across all cameras
            cam_dets: dict[str, list] = {}
            for cam, _ in CAMS:
                dets = [(xc, lid) for (xc, lid, cid)
                        in cam_frame_det.get((cam, frame), [])
                        if cid == cls_id]
                dets.sort()   # sort by x-center
                if dets:
                    cam_dets[cam] = dets

            if len(cam_dets) < 2:
                # Assign new global ID to any new unassigned detections
                for cam, dets in cam_dets.items():
                    for _, lid in dets:
                        key = (cam, lid)
                        if key not in local_to_global:
                            local_to_global[key] = next_gid[0]
                            next_gid[0] += 1
                continue

            # Match by rank across cameras (sorted by x → same rank = same bag)
            max_rank = max(len(d) for d in cam_dets.values())
            for rank in range(max_rank):
                # Find which cameras have a detection at this rank
                rank_dets = {cam: dets[rank]
                             for cam, dets in cam_dets.items()
                             if rank < len(dets)}
                # Determine global ID for this rank group
                gid = None
                for cam, (_, lid) in rank_dets.items():
                    key = (cam, lid)
                    if key in local_to_global:
                        gid = local_to_global[key]
                        break
                if gid is None:
                    gid = next_gid[0]
                    next_gid[0] += 1
                for cam, (_, lid) in rank_dets.items():
                    local_to_global[(cam, lid)] = gid

    # Assign any remaining tracklets that never appeared in any frame match
    for cam, tlets in per_cam.items():
        for lid, t in tlets.items():
            key = (cam, lid)
            if key not in local_to_global:
                local_to_global[key] = next_gid[0]
                next_gid[0] += 1

    # Write global_id onto tracklets
    for cam, tlets in per_cam.items():
        for lid, t in tlets.items():
            t.global_id = local_to_global.get((cam, lid), -1)

    return next_gid[0]


def _assign_within_class(tlets_by_cam: dict[str, list[Tracklet]],
                          cost_fn, next_gid: list[int], max_cost: float = 0.95):
    """
    Generic: build cost matrix across all tracklet pairs from different cameras
    using cost_fn(ti, tj) → float.  Run Hungarian, form groups, assign global IDs.
    """
    cam_names = [cam for cam, ts in tlets_by_cam.items() if ts]
    all_tlets = [t for ts in tlets_by_cam.values() for t in ts]
    n = len(all_tlets)
    if n == 0:
        return

    # Build n×n cost matrix (only cross-camera pairs matter)
    cost = np.full((n, n), INF)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = all_tlets[i], all_tlets[j]
            if ti.cam == tj.cam:
                continue   # same camera → can't be same object
            c = cost_fn(ti, tj)
            cost[i, j] = c
            cost[j, i] = c

    # Greedy merging: repeatedly take lowest-cost compatible pair
    assigned = [-1] * n      # group index for each tracklet
    groups: list[set[int]] = []
    cam_in_group: list[set[str]] = []

    pairs = sorted((cost[i, j], i, j)
                   for i in range(n) for j in range(i + 1, n)
                   if cost[i, j] < max_cost)

    for c, i, j in pairs:
        gi, gj = assigned[i], assigned[j]
        ci, cj = all_tlets[i].cam, all_tlets[j].cam

        if gi == -1 and gj == -1:
            # New group
            g = len(groups)
            groups.append({i, j})
            cam_in_group.append({ci, cj})
            assigned[i] = assigned[j] = g

        elif gi >= 0 and gj == -1:
            # Add j to i's group if camera not yet in it
            if cj not in cam_in_group[gi]:
                groups[gi].add(j)
                cam_in_group[gi].add(cj)
                assigned[j] = gi

        elif gi == -1 and gj >= 0:
            if ci not in cam_in_group[gj]:
                groups[gj].add(i)
                cam_in_group[gj].add(ci)
                assigned[i] = gj

        # Merging two existing groups: skip to avoid complex bookkeeping

    # Singletons for unassigned
    for i in range(n):
        if assigned[i] == -1:
            g = len(groups)
            groups.append({i})
            cam_in_group.append({all_tlets[i].cam})
            assigned[i] = g

    # Assign global IDs to groups
    for grp in groups:
        gid = next_gid[0]
        next_gid[0] += 1
        for idx in grp:
            all_tlets[idx].global_id = gid


def method_tracklet_temporal(per_cam: dict[str, dict[int, Tracklet]]) -> int:
    next_gid = [1]
    for cls_id in range(NUM_CLASSES):
        by_cam = {cam: [t for t in tlets.values() if t.class_id == cls_id]
                  for cam, tlets in per_cam.items()}
        _assign_within_class(by_cam, lambda a, b: 1.0 - jaccard(a, b),
                              next_gid)
    return next_gid[0]


def method_tracklet_spatial(per_cam: dict[str, dict[int, Tracklet]]) -> int:
    """
    Match tracklets by spatial x-rank within same class.
    Sort tracklets per camera by mean_x, then cost = |rank_a - rank_b| / max_rank.
    """
    next_gid = [1]
    for cls_id in range(NUM_CLASSES):
        by_cam: dict[str, list[Tracklet]] = {}
        for cam, tlets in per_cam.items():
            subset = sorted([t for t in tlets.values() if t.class_id == cls_id],
                            key=lambda t: t.mean_x)
            by_cam[cam] = subset

        # Rank-based cost
        rank_of: dict[int, float] = {}   # id(tracklet) → normalised rank
        for cam, subset in by_cam.items():
            n = len(subset)
            for r, t in enumerate(subset):
                rank_of[id(t)] = r / max(n - 1, 1)

        def spatial_cost(a: Tracklet, b: Tracklet) -> float:
            return abs(rank_of.get(id(a), 0) - rank_of.get(id(b), 0))

        _assign_within_class(by_cam, spatial_cost, next_gid)
    return next_gid[0]


def method_tracklet_combined(per_cam: dict[str, dict[int, Tracklet]]) -> int:
    """
    Weighted combination: 0.5 * temporal + 0.5 * spatial.
    """
    next_gid = [1]
    for cls_id in range(NUM_CLASSES):
        by_cam: dict[str, list[Tracklet]] = {}
        for cam, tlets in per_cam.items():
            subset = sorted([t for t in tlets.values() if t.class_id == cls_id],
                            key=lambda t: t.mean_x)
            by_cam[cam] = subset

        rank_of: dict[int, float] = {}
        for cam, subset in by_cam.items():
            n = len(subset)
            for r, t in enumerate(subset):
                rank_of[id(t)] = r / max(n - 1, 1)

        def combined_cost(a: Tracklet, b: Tracklet) -> float:
            temporal = 1.0 - jaccard(a, b)
            spatial  = abs(rank_of.get(id(a), 0) - rank_of.get(id(b), 0))
            return 0.5 * temporal + 0.5 * spatial

        _assign_within_class(by_cam, combined_cost, next_gid)
    return next_gid[0]


# ── Method dispatch ────────────────────────────────────────────────────────────

METHOD_FN = {
    "no_assoc":          method_no_assoc,
    "class_only":        method_class_only,
    "tracklet_temporal": method_tracklet_temporal,
    "tracklet_spatial":  method_tracklet_spatial,
    "tracklet_combined": method_tracklet_combined,
}


# ── Output writers ─────────────────────────────────────────────────────────────

def write_global_pred(per_cam: dict[str, dict[int, Tracklet]],
                      out_path: Path) -> int:
    lines = []
    for cam, tlets in per_cam.items():
        shift = CAM_IDX[cam] * CAM_FRAME_STRIDE
        for t in tlets.values():
            for seq_idx in sorted(t.frames):
                x, y, w, h, conf = t.bboxes[seq_idx]
                lines.append(
                    f"{seq_idx + shift},{t.global_id},{x:.1f},{y:.1f},"
                    f"{w:.1f},{h:.1f},{conf:.4f},{t.class_id},-1,-1"
                )
    lines.sort(key=lambda l: int(l.split(",")[0]))
    out_path.write_text("\n".join(lines))
    return len(lines)


def write_global_gt(ds_cfg: dict, out_path: Path) -> int:
    lines = []
    for cam_idx, (cam_short, cam_sub) in enumerate(CAMS):
        shift = cam_idx * CAM_FRAME_STRIDE
        rows = load_gt(cam_sub, ds_cfg["ds_dir"])
        for frame, obj_id, x, y, w, h, conf, cls_id in rows:
            lines.append(
                f"{frame + shift},{obj_id},{x:.1f},{y:.1f},"
                f"{w:.1f},{h:.1f},{conf:.4f},{cls_id},-1,-1"
            )
    lines.sort(key=lambda l: int(l.split(",")[0]))
    out_path.write_text("\n".join(lines))
    return len(lines)


# ── MOT evaluation ─────────────────────────────────────────────────────────────

def evaluate_mot_files(pred_path: Path, gt_path: Path) -> dict | None:
    pred_rows = _parse_mot_file(pred_path)
    gt_rows   = _parse_mot_file(gt_path)

    pred_by_frame: dict[int, list] = defaultdict(list)
    for r in pred_rows:
        pred_by_frame[r[0]].append(r)
    gt_by_frame: dict[int, list] = defaultdict(list)
    for r in gt_rows:
        gt_by_frame[r[0]].append(r)

    all_frames = sorted(set(pred_by_frame) | set(gt_by_frame))
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in all_frames:
        g_rows = gt_by_frame.get(frame, [])
        p_rows = pred_by_frame.get(frame, [])
        g_ids  = [r[1] for r in g_rows]
        p_ids  = [r[1] for r in p_rows]

        if not g_rows or not p_rows:
            acc.update(g_ids, p_ids, np.empty((len(g_ids), len(p_ids))))
            continue

        # IoU distance matrix
        def to_tlbr(rows):
            return np.array([[r[2], r[3], r[2]+r[4], r[3]+r[5]]
                             for r in rows], dtype=float)
        g_tlbr = to_tlbr(g_rows)
        p_tlbr = to_tlbr(p_rows)
        inter_x1 = np.maximum(g_tlbr[:, None, 0], p_tlbr[None, :, 0])
        inter_y1 = np.maximum(g_tlbr[:, None, 1], p_tlbr[None, :, 1])
        inter_x2 = np.minimum(g_tlbr[:, None, 2], p_tlbr[None, :, 2])
        inter_y2 = np.minimum(g_tlbr[:, None, 3], p_tlbr[None, :, 3])
        inter = (np.clip(inter_x2 - inter_x1, 0, None) *
                 np.clip(inter_y2 - inter_y1, 0, None))
        area_g = (g_tlbr[:, 2]-g_tlbr[:, 0]) * (g_tlbr[:, 3]-g_tlbr[:, 1])
        area_p = (p_tlbr[:, 2]-p_tlbr[:, 0]) * (p_tlbr[:, 3]-p_tlbr[:, 1])
        union  = area_g[:, None] + area_p[None, :] - inter
        iou    = np.where(union > 0, inter / union, 0.0)
        dist   = np.where(1 - iou <= 0.5, 1 - iou, np.nan)
        acc.update(g_ids, p_ids, dist)

    mh = mm.metrics.create()
    try:
        s   = mh.compute(acc, metrics=MOT_METRICS, name="r")
        return {k: s.loc["r", k] for k in MOT_METRICS}
    except Exception as e:
        print(f"      metrics error: {e}")
        return None


# ── Per-run orchestration ──────────────────────────────────────────────────────

def run_one(tracker_name: str, dataset_name: str, method: str) -> dict | None:
    ds_cfg      = DATASETS[dataset_name]
    tracker_out = OUT_ROOT / tracker_name / dataset_name

    # Load tracklets
    per_cam: dict[str, dict[int, Tracklet]] = {}
    any_class = False
    for cam_short, _ in CAMS:
        pred_path = tracker_out / cam_short / "pred.txt"
        tlets = load_tracklets(pred_path, cam_short)
        per_cam[cam_short] = tlets
        if any(t.class_id >= 0 for t in tlets.values()):
            any_class = True

    if not any_class and method != "no_assoc":
        print(f"    WARNING: pred.txt has no class_id — re-run run_tracking.py")
        print(f"    Falling back to no_assoc.")
        method = "no_assoc"

    # Run association
    METHOD_FN[method](per_cam)

    # Write outputs
    method_dir = tracker_out / method
    method_dir.mkdir(exist_ok=True)
    pred_out = method_dir / "global_pred.txt"
    gt_out   = method_dir / "global_gt.txt"

    n_pred = write_global_pred(per_cam, pred_out)

    # GT only needs to be written once per tracker/dataset (same for all methods)
    if not gt_out.exists():
        write_global_gt(ds_cfg, gt_out)

    # Evaluate
    row = evaluate_mot_files(pred_out, gt_out)
    return row


# ── Pretty table ───────────────────────────────────────────────────────────────

def print_benchmark_table(results: dict[str, dict | None]):
    pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else "N/A"
    hdr = f"  {'Run (tracker/dataset/method)':<40}  {'MOTA':>7}  {'MOTP':>7}  {'IDF1':>7}  {'IDSW':>6}  {'MT':>5}  {'ML':>5}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, row in results.items():
        if row is None:
            print(f"  {name:<40}  (failed)")
            continue
        print(f"  {name:<40}  {pct(row['mota']):>7}  {pct(row['motp']):>7}  "
              f"{pct(row['idf1']):>7}  {int(row['num_switches']):>6}  "
              f"{row['mostly_tracked']:>5}  {row['mostly_lost']:>5}")
    print("=" * len(hdr))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"],
                    default="bytetrack")
    ap.add_argument("--dataset", choices=["val", "test"],
                    default="val")
    ap.add_argument("--method", choices=METHODS + ["all"],
                    default="all")
    ap.add_argument("--all", action="store_true",
                    help="Run all tracker × dataset × method combinations")
    args = ap.parse_args()

    if args.all:
        trackers = ["bytetrack", "botsort"]
        datasets = ["val", "test"]
        methods  = METHODS
    else:
        trackers = [args.tracker]
        datasets = [args.dataset]
        methods  = METHODS if args.method == "all" else [args.method]

    results: dict[str, dict | None] = {}
    for tracker in trackers:
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"  {tracker.upper()} / {dataset.upper()}")
            print(f"{'='*60}")
            for method in methods:
                print(f"  [{method}]", end="  ", flush=True)
                row = run_one(tracker, dataset, method)
                key = f"{tracker}/{dataset}/{method}"
                results[key] = row
                if row:
                    pct = lambda v: f"{v*100:.1f}%"
                    print(f"MOTA={pct(row['mota'])}  IDF1={pct(row['idf1'])}  "
                          f"IDSW={int(row['num_switches'])}")
                else:
                    print("(no result)")

    if len(results) > 1:
        print_benchmark_table(results)


if __name__ == "__main__":
    main()
