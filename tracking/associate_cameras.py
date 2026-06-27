#!/usr/bin/env python3
"""
tracking/associate_cameras.py

Inter-camera association step (MTMCT Stage 2).

Approach — hierarchical TbD (Tracking-by-Detection):
  1. Load per-camera intra-camera tracklets from pred.txt files
     (already produced by run_tracking.py with ByteTrack or BoT-SORT)
  2. Build a tracklet-level cost matrix across all camera pairs:
       - Different class_id  → cost = inf  (cannot be same object)
       - Same class_id       → cost = 1 - Jaccard(frame_sets)
     Jaccard temporal overlap captures synchronization: all 4 cameras see the
     same conveyor belt simultaneously, so the same polybag appears in the same
     frame range across cameras.
  3. Hungarian assignment within each color-class group assigns globally
     consistent IDs to local tracklets.
  4. Write:
       tracking_results/{tracker}/{dataset}/global_pred.txt   — merged MOT file
       tracking_results/{tracker}/{dataset}/global_gt.txt     — merged GT file
  5. Evaluate global_pred vs global_gt with motmetrics → true MCMOT metrics.

Disambiguation when two bags of the same color appear simultaneously:
  Within a same-class group, remaining unmatched tracklets are ordered by their
  mean x-center in image space and matched by rank across cameras (left-to-right
  spatial ordering is consistent on a conveyor viewed from the same side).

Usage:
  cd repo/tracking
  python associate_cameras.py --tracker bytetrack --dataset val
  python associate_cameras.py --tracker botsort   --dataset test
  python associate_cameras.py --all   # all 4 combinations + compare table
"""

import argparse
import sys
from collections import defaultdict
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
BASE     = Path("/Users/awthura/OVGU/AMS")
OUT_ROOT = REPO / "tracking_results"

CAMS = [
    ("front", "cam_01_front"),
    ("back",  "cam_02_back"),
    ("left",  "cam_03_left"),
    ("right", "cam_04_right"),
]
CAM_OFFSET = {cam: i * 1000 for i, (cam, _) in enumerate(CAMS)}  # frame offset per cam in merged file

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

NUM_CLASSES = 7
INF = 1e9


# ── Data structures ────────────────────────────────────────────────────────────

class Tracklet:
    """One continuous track in one camera view."""
    __slots__ = ("cam", "local_id", "class_id", "frames",
                 "bboxes", "mean_x", "global_id")

    def __init__(self, cam: str, local_id: int, class_id: int):
        self.cam       = cam
        self.local_id  = local_id
        self.class_id  = class_id
        self.frames: set[int] = set()
        self.bboxes: dict[int, tuple] = {}  # seq_idx → (x, y, w, h, conf)
        self.mean_x    = 0.0
        self.global_id = -1

    def finalize(self):
        if self.bboxes:
            self.mean_x = float(np.mean([b[0] + b[2] / 2
                                         for b in self.bboxes.values()]))


# ── File I/O ───────────────────────────────────────────────────────────────────

def load_tracklets(pred_path: Path, cam: str) -> dict[int, Tracklet]:
    """Load pred.txt → {local_id: Tracklet}."""
    tracklets: dict[int, Tracklet] = {}
    if not pred_path.exists():
        return tracklets
    for line in pred_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        seq_idx  = int(parts[0])
        local_id = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf     = float(parts[6]) if len(parts) > 6 else 1.0
        # class_id field: pred.txt has -1 here (standard MOT format doesn't carry class)
        # We recover class later from the OBB pred format... but our pred.txt omits it.
        # Fall back to class_id=0 if missing; real class comes from OBB pred files.
        cls_id   = int(parts[7]) if len(parts) > 7 and parts[7].strip() not in ("-1", "") else -1

        if local_id not in tracklets:
            tracklets[local_id] = Tracklet(cam, local_id, cls_id)
        tracklets[local_id].frames.add(seq_idx)
        tracklets[local_id].bboxes[seq_idx] = (x, y, w, h, conf)
    for t in tracklets.values():
        t.finalize()
    return tracklets


def load_tracklets_with_class(pred_path: Path, cam: str) -> dict[int, Tracklet]:
    """
    Load pred.txt and recover class_id from the OBB pred file stored alongside.
    The OBB pred file (pred_obb.txt) carries the class field if available,
    otherwise we use the majority class per track from the detection conf.
    Since our run_tracking.py writes standard MOT (class=-1), we use a separate
    class lookup from per-frame annotated data if needed.

    Fallback: infer class from track color via the per-frame YOLO predictions
    stored in pred_obb.txt (if it exists next to pred.txt).
    """
    tracklets = load_tracklets(pred_path, cam)

    # If class_id is missing (-1), try to recover from pred_obb.txt
    obb_path = pred_path.parent / "pred_obb.txt"
    if obb_path.exists():
        cls_votes: dict[int, list[int]] = defaultdict(list)
        for line in obb_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 12:
                continue
            local_id = int(parts[1])
            cls_id   = int(parts[11])
            cls_votes[local_id].append(cls_id)
        for local_id, votes in cls_votes.items():
            if local_id in tracklets and tracklets[local_id].class_id == -1:
                # majority vote
                from collections import Counter
                tracklets[local_id].class_id = Counter(votes).most_common(1)[0][0]

    return tracklets


def load_gt_tracklets(gt_path: Path, cam_sub: str, ds_dir: Path,
                      frame_start: int, frame_end: int,
                      frame_offset: int) -> dict[int, Tracklet]:
    """Load GT tracklets from gt.txt (standard AABB MOT)."""
    gt_path = ds_dir / "mot_obb" / cam_sub / "gt" / "gt.txt"
    tracklets: dict[int, Tracklet] = {}
    if not gt_path.exists():
        return tracklets
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        seq_idx  = int(parts[0])
        obj_id   = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf     = float(parts[6]) if len(parts) > 6 else 1.0
        cls_id   = int(parts[7]) if len(parts) > 7 else 0
        if conf < 0:
            continue
        if obj_id not in tracklets:
            tracklets[obj_id] = Tracklet("gt", obj_id, cls_id)
        tracklets[obj_id].frames.add(seq_idx)
        tracklets[obj_id].bboxes[seq_idx] = (x, y, w, h, conf)
    for t in tracklets.values():
        t.finalize()
    return tracklets


# ── Temporal Jaccard overlap ───────────────────────────────────────────────────

def temporal_jaccard(a: Tracklet, b: Tracklet) -> float:
    inter = len(a.frames & b.frames)
    union = len(a.frames | b.frames)
    return inter / union if union > 0 else 0.0


# ── Hungarian assignment within one class group ────────────────────────────────

def assign_global_ids_for_class(tracklets_by_cam: dict[str, list[Tracklet]],
                                 next_gid: list[int]) -> int:
    """
    Match tracklets of the same class across cameras and assign global IDs.
    Uses Hungarian algorithm on temporal Jaccard distance.
    Returns next available global ID.
    """
    cam_names = list(tracklets_by_cam.keys())
    if not cam_names:
        return next_gid[0]

    # Collect all unassigned tracklets across cameras
    all_tracklets = [t for cam in cam_names for t in tracklets_by_cam[cam]]
    if not all_tracklets:
        return next_gid[0]

    # Build N×N cost matrix (all pairs)
    n = len(all_tracklets)
    cost = np.full((n, n), INF)
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = all_tracklets[i], all_tracklets[j]
            if ti.cam == tj.cam:
                cost[i, j] = INF  # same camera, can't be same object
                cost[j, i] = INF
            else:
                jac = temporal_jaccard(ti, tj)
                c = 1.0 - jac  # low cost = high overlap = likely same object
                cost[i, j] = c
                cost[j, i] = c

    # Greedy cluster formation: iteratively merge highest-overlap pairs
    # into groups, one tracklet per camera per group.
    assigned = [False] * n
    groups: list[list[int]] = []

    # Sort pairs by cost ascending (best overlaps first)
    pairs = [(cost[i, j], i, j)
             for i in range(n) for j in range(i + 1, n)
             if cost[i, j] < 0.99]  # only consider pairs with >1% overlap
    pairs.sort()

    cam_of = {i: all_tracklets[i].cam for i in range(n)}

    for c, i, j in pairs:
        if assigned[i] or assigned[j]:
            continue
        # Find which group to add both into, if any
        merged = False
        for grp in groups:
            cams_in_grp = {cam_of[k] for k in grp}
            if cam_of[i] not in cams_in_grp and cam_of[j] not in cams_in_grp:
                grp += [i, j]
                assigned[i] = assigned[j] = True
                merged = True
                break
            elif cam_of[i] not in cams_in_grp and not assigned[j]:
                # j is already in group; add i
                if not assigned[i]:
                    grp.append(i)
                    assigned[i] = True
                    merged = True
                    break
            elif cam_of[j] not in cams_in_grp and not assigned[i]:
                if not assigned[j]:
                    grp.append(j)
                    assigned[j] = True
                    merged = True
                    break
        if not merged:
            groups.append([i, j])
            assigned[i] = assigned[j] = True

    # Remaining unassigned tracklets get their own singleton group
    for i in range(n):
        if not assigned[i]:
            groups.append([i])

    # Assign global IDs to groups
    for grp in groups:
        gid = next_gid[0]
        next_gid[0] += 1
        for idx in grp:
            all_tracklets[idx].global_id = gid

    return next_gid[0]


# ── Main association ───────────────────────────────────────────────────────────

def associate(tracker_name: str, dataset_name: str) -> Path | None:
    ds_cfg      = DATASETS[dataset_name]
    tracker_out = OUT_ROOT / tracker_name / dataset_name

    print(f"\n  [{tracker_name.upper()} / {dataset_name}] Inter-camera association")

    # Load per-camera tracklets
    all_tracklets: dict[str, dict[int, Tracklet]] = {}
    for cam_short, _ in CAMS:
        pred_path = tracker_out / cam_short / "pred.txt"
        tlets = load_tracklets_with_class(pred_path, cam_short)
        all_tracklets[cam_short] = tlets
        print(f"    {cam_short}: {len(tlets)} tracklets")

    # Check if any class information available
    has_class = any(
        t.class_id >= 0
        for cam_tlets in all_tracklets.values()
        for t in cam_tlets.values()
    )

    if not has_class:
        print("    WARNING: no class_id in pred.txt — re-run tracking with pred_obb.txt")
        print("    Falling back to temporal-only association (ignores color).")

    # Group tracklets by class, then associate within each class
    next_gid = [1]
    for cls_id in range(NUM_CLASSES if has_class else 1):
        if has_class:
            group: dict[str, list[Tracklet]] = {
                cam: [t for t in tlets.values() if t.class_id == cls_id]
                for cam, tlets in all_tracklets.items()
            }
        else:
            group = {cam: list(tlets.values())
                     for cam, tlets in all_tracklets.items()}

        assign_global_ids_for_class(group, next_gid)

    # Write merged global_pred.txt
    # Frame index: use seq_idx + cam_offset so each camera occupies its own
    # frame window. Cameras are synchronized, so seq_idx 1 = same moment for all.
    global_pred_path = tracker_out / "global_pred.txt"
    lines = []
    for cam_short, tlets in all_tracklets.items():
        frame_shift = list(CAMS).index(
            next(c for c in CAMS if c[0] == cam_short)) * 10000
        for t in tlets.values():
            gid = t.global_id
            for seq_idx in sorted(t.frames):
                x, y, w, h, conf = t.bboxes[seq_idx]
                lines.append(
                    f"{seq_idx + frame_shift},{gid},{x:.1f},{y:.1f},"
                    f"{w:.1f},{h:.1f},{conf:.4f},{t.class_id},-1,-1"
                )
    global_pred_path.write_text("\n".join(sorted(lines,
                                                  key=lambda l: int(l.split(",")[0]))))
    print(f"    global_pred.txt: {len(lines)} rows  ({next_gid[0]-1} global IDs)")

    # Write merged global_gt.txt
    global_gt_path = tracker_out / "global_gt.txt"
    gt_lines = []
    for cam_short, cam_sub in CAMS:
        frame_shift = list(CAMS).index(
            next(c for c in CAMS if c[0] == cam_short)) * 10000
        gt_tlets = load_gt_tracklets(
            None, cam_sub, ds_cfg["ds_dir"],
            ds_cfg["frame_start"], ds_cfg["frame_end"], ds_cfg["frame_offset"]
        )
        for t in gt_tlets.values():
            for seq_idx in sorted(t.frames):
                x, y, w, h, conf = t.bboxes[seq_idx]
                gt_lines.append(
                    f"{seq_idx + frame_shift},{t.local_id},{x:.1f},{y:.1f},"
                    f"{w:.1f},{h:.1f},{conf:.4f},{t.class_id},-1,-1"
                )
    global_gt_path.write_text("\n".join(sorted(gt_lines,
                                                key=lambda l: int(l.split(",")[0]))))
    print(f"    global_gt.txt:   {len(gt_lines)} rows")
    return tracker_out


def evaluate_global(tracker_out: Path, tracker_name: str, dataset_name: str) -> dict | None:
    """Evaluate global_pred vs global_gt with motmetrics."""
    from evaluate_mot import load_mot_file, iou_distance

    pred_path = tracker_out / "global_pred.txt"
    gt_path   = tracker_out / "global_gt.txt"
    if not pred_path.exists() or not gt_path.exists():
        print("    Missing global files — run association first.")
        return None

    pred_data = load_mot_file(pred_path)
    gt_data   = load_mot_file(gt_path)

    all_frames = sorted(set(pred_data) | set(gt_data))
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in all_frames:
        gt_rows   = gt_data.get(frame, [])
        pred_rows = pred_data.get(frame, [])
        gt_ids    = [r[0] for r in gt_rows]
        pred_ids  = [r[0] for r in pred_rows]
        dist      = iou_distance(gt_rows, pred_rows)
        dist      = np.where(dist <= 0.5, dist, np.nan)
        acc.update(gt_ids, pred_ids, dist)

    METRICS = ["num_frames", "mota", "motp", "idf1",
               "num_switches", "mostly_tracked", "mostly_lost",
               "num_false_positives", "num_misses"]
    mh      = mm.metrics.create()
    pct     = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else str(v)
    try:
        s   = mh.compute(acc, metrics=METRICS, name="GLOBAL")
        row = {k: s.loc["GLOBAL", k] for k in METRICS}
        print(f"\n    [GLOBAL MCMOT — {tracker_name}/{dataset_name}]")
        print(f"    MOTA={pct(row['mota'])}  MOTP={pct(row['motp'])}  "
              f"IDF1={pct(row['idf1'])}  IDSW={int(row['num_switches'])}  "
              f"MT={row['mostly_tracked']}  ML={row['mostly_lost']}")
        return row
    except Exception as e:
        print(f"    Global eval error: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"])
    ap.add_argument("--dataset", choices=["val", "test"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    combos = (
        [("bytetrack", "val"), ("bytetrack", "test"),
         ("botsort",   "val"), ("botsort",   "test")]
        if args.all or (not args.tracker and not args.dataset)
        else [(args.tracker or "bytetrack", args.dataset or "val")]
    )

    results = {}
    for tracker_name, dataset_name in combos:
        out = associate(tracker_name, dataset_name)
        if out:
            row = evaluate_global(out, tracker_name, dataset_name)
            results[f"{tracker_name}/{dataset_name}"] = row

    if len(results) > 1:
        print("\n" + "=" * 72)
        print(f"  {'Run':<22}  {'MOTA':>7}  {'MOTP':>7}  {'IDF1':>7}  {'IDSW':>6}")
        print("  " + "-" * 60)
        for name, row in results.items():
            if row is None:
                print(f"  {name:<22}  (no results)")
            else:
                pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else "N/A"
                print(f"  {name:<22}  {pct(row['mota']):>7}  "
                      f"{pct(row['motp']):>7}  {pct(row['idf1']):>7}  "
                      f"{int(row['num_switches']):>6}")
        print("=" * 72)


if __name__ == "__main__":
    main()
