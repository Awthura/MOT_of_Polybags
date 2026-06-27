#!/usr/bin/env python3
"""
tracking/evaluate_mot.py

Evaluates tracking predictions against ground truth MOT16 files.
Uses py-motmetrics to compute MOTA, MOTP, IDF1, MT, ML, FP, FN, IDSW.

Requires:  pip install motmetrics

Usage:
  # Single tracker + dataset combination
  python evaluate_mot.py --tracker bytetrack --dataset val
  python evaluate_mot.py --tracker botsort   --dataset test

  # All combinations at once (side-by-side comparison)
  python evaluate_mot.py --all

  # Evaluate only a specific camera
  python evaluate_mot.py --tracker bytetrack --dataset val --cam front
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import motmetrics as mm
except ImportError:
    sys.exit("motmetrics not installed — run:  pip install motmetrics")

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

TRACKERS = ["bytetrack", "botsort"]


# ── MOT file loaders ──────────────────────────────────────────────────────────

def load_mot_file(path: Path) -> dict[int, list[tuple]]:
    """
    Load a MOT16 file (standard AABB format):
      frame, id, left, top, width, height, conf, ...

    Returns {seq_idx: [(id, left, top, w, h), ...]}
    Skips rows with conf == 0 or rows where the object is not active (conf < 0).
    """
    data = defaultdict(list)
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        frame  = int(parts[0])
        obj_id = int(parts[1])
        left   = float(parts[2])
        top    = float(parts[3])
        w      = float(parts[4])
        h      = float(parts[5])
        conf   = float(parts[6]) if len(parts) > 6 else 1.0
        if conf < 0:  # MOT16 convention: conf=-1 means ignore region
            continue
        data[frame].append((obj_id, left, top, w, h))
    return data


def load_gt_file(cam_sub: str, ds_dir: Path) -> dict[int, list[tuple]]:
    """Load GT from mot_obb/{cam_sub}/gt/gt.txt (standard MOT16 AABB format)."""
    gt_path = ds_dir / "mot_obb" / cam_sub / "gt" / "gt.txt"
    return load_mot_file(gt_path)


# ── IoU distance matrix ───────────────────────────────────────────────────────

def iou_distance(gt_boxes, pred_boxes):
    """
    Compute 1-IoU distance matrix between GT and predicted AABB boxes.
    gt_boxes / pred_boxes: list of (left, top, w, h)
    Returns: np.ndarray of shape (len(gt), len(pred)), values in [0, 1].
    """
    if not gt_boxes or not pred_boxes:
        return np.empty((len(gt_boxes), len(pred_boxes)))

    def to_tlbr(boxes):
        arr = np.array(boxes, dtype=float)
        tlbr = np.stack([arr[:, 0], arr[:, 1],
                         arr[:, 0] + arr[:, 2], arr[:, 1] + arr[:, 3]], axis=1)
        return tlbr

    g = to_tlbr([(b[1], b[2], b[3], b[4]) for b in gt_boxes])
    p = to_tlbr([(b[1], b[2], b[3], b[4]) for b in pred_boxes])

    # Broadcast intersection
    inter_x1 = np.maximum(g[:, None, 0], p[None, :, 0])
    inter_y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    inter_x2 = np.minimum(g[:, None, 2], p[None, :, 2])
    inter_y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter_w  = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h  = np.clip(inter_y2 - inter_y1, 0, None)
    inter    = inter_w * inter_h

    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    union  = area_g[:, None] + area_p[None, :] - inter

    iou = np.where(union > 0, inter / union, 0.0)
    return 1.0 - iou   # distance = 1 - IoU


# ── Per-camera evaluation ──────────────────────────────────────────────────────

def evaluate_camera(cam_short: str, cam_sub: str, ds_cfg: dict,
                    pred_path: Path, max_dist: float = 0.5):
    """
    Build a MOT accumulator for one camera sequence.
    max_dist: distance threshold for a match (0.5 → IoU ≥ 0.5)
    """
    ds_dir       = ds_cfg["ds_dir"]
    frame_offset = ds_cfg["frame_offset"]
    frame_start  = ds_cfg["frame_start"]
    frame_end    = ds_cfg["frame_end"]

    gt_data   = load_gt_file(cam_sub, ds_dir)
    pred_data = load_mot_file(pred_path)

    if not gt_data:
        print(f"      {cam_short}: GT not found at {ds_dir}/mot_obb/{cam_sub}/gt/gt.txt")
        return None

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_num in range(frame_start, frame_end + 1):
        seq_idx = frame_num - frame_offset  # 1-based

        gt_rows   = gt_data.get(seq_idx, [])
        pred_rows = pred_data.get(seq_idx, [])

        gt_ids   = [r[0] for r in gt_rows]
        pred_ids = [r[0] for r in pred_rows]

        dist = iou_distance(gt_rows, pred_rows)
        # Distances > max_dist are set to NaN so motmetrics treats them as misses
        dist = np.where(dist <= max_dist, dist, np.nan)

        acc.update(gt_ids, pred_ids, dist)

    return acc


# ── Aggregate and print metrics ────────────────────────────────────────────────

METRICS = [
    "num_frames", "mota", "motp", "idf1",
    "num_switches", "mt", "ml", "fp", "fn",
    "precision", "recall",
]

METRIC_NAMES = {
    "num_frames":   "Frames",
    "mota":         "MOTA",
    "motp":         "MOTP",
    "idf1":         "IDF1",
    "num_switches": "IDSW",
    "mt":           "MT",
    "ml":           "ML",
    "fp":           "FP",
    "fn":           "FN",
    "precision":    "Prec",
    "recall":       "Recall",
}


def evaluate_run(tracker_name: str, dataset_name: str,
                 cams: list[tuple] | None = None) -> dict | None:
    """
    Evaluate all cameras for one tracker/dataset combo.
    Returns a dict of metrics, or None if pred files are missing.
    """
    ds_cfg      = DATASETS[dataset_name]
    tracker_out = OUT_ROOT / tracker_name / dataset_name
    cam_list    = cams or CAMS

    print(f"\n  [{tracker_name.upper()} / {dataset_name}]")

    accumulators = []
    names        = []
    any_found    = False

    for cam_short, cam_sub in cam_list:
        pred_path = tracker_out / cam_short / "pred.txt"
        if not pred_path.exists():
            print(f"    {cam_short}: pred.txt not found — skipping")
            continue
        any_found = True
        acc = evaluate_camera(cam_short, cam_sub, ds_cfg, pred_path)
        if acc is not None:
            accumulators.append(acc)
            names.append(cam_short)

    if not any_found:
        print(f"    No predictions found in {tracker_out}")
        return None

    mh  = mm.metrics.create()
    pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else str(v)

    # Per-camera
    for acc, name in zip(accumulators, names):
        try:
            s   = mh.compute(acc, metrics=METRICS, name=name)
            row = {k: s.loc[name, k] for k in METRICS}
            print(f"    {name:5s}  MOTA={pct(row['mota'])}  "
                  f"MOTP={pct(row['motp'])}  IDF1={pct(row['idf1'])}  "
                  f"IDSW={int(row['num_switches'])}  "
                  f"MT={row['mt']}  ML={row['ml']}")
        except Exception as e:
            print(f"    {name}: metrics error — {e}")

    # Overall (using compute_many with generate_overall)
    if not accumulators:
        return None
    try:
        summary = mh.compute_many(
            accumulators, metrics=METRICS, names=names, generate_overall=True
        )
        row = {k: summary.loc["OVERALL", k] for k in METRICS}
        pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else str(v)
        print(f"    {'ALL':5s}  MOTA={pct(row['mota'])}  "
              f"MOTP={pct(row['motp'])}  IDF1={pct(row['idf1'])}  "
              f"IDSW={int(row['num_switches'])}  "
              f"MT={row['mt']}  ML={row['ml']}")
        return row
    except Exception as e:
        print(f"    Aggregate error: {e}")
        return None


def print_comparison_table(results: dict):
    """Print a side-by-side comparison table of all evaluated runs."""
    if not results:
        return
    print("\n" + "=" * 72)
    print(f"  {'Run':<22}  {'MOTA':>7}  {'MOTP':>7}  {'IDF1':>7}  "
          f"{'IDSW':>6}  {'MT':>5}  {'ML':>5}")
    print("  " + "-" * 68)
    for run_name, row in results.items():
        if row is None:
            print(f"  {run_name:<22}  (no results)")
            continue
        pct = lambda v: f"{v*100:.1f}%" if isinstance(v, float) else "N/A"
        print(f"  {run_name:<22}  {pct(row['mota']):>7}  {pct(row['motp']):>7}  "
              f"{pct(row['idf1']):>7}  {int(row['num_switches']):>6}  "
              f"{row['mt']:>5}  {row['ml']:>5}")
    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["bytetrack", "botsort"],
                    help="Tracker to evaluate (omit to use --all)")
    ap.add_argument("--dataset", choices=["val", "test"],
                    help="Dataset to evaluate (omit to use --all)")
    ap.add_argument("--cam", choices=["front", "back", "left", "right"],
                    help="Evaluate only one camera")
    ap.add_argument("--all", action="store_true",
                    help="Evaluate all tracker × dataset combinations")
    ap.add_argument("--iou-threshold", type=float, default=0.5,
                    help="Min IoU for a detection to count as TP (default 0.5)")
    args = ap.parse_args()

    max_dist = 1.0 - args.iou_threshold
    cam_list = None
    if args.cam:
        # single-camera mode
        cam_map = {s: (s, sub) for s, sub in CAMS}
        cam_list = [cam_map[args.cam]]

    if args.all or (not args.tracker and not args.dataset):
        results = {}
        for tracker_name in TRACKERS:
            for dataset_name in DATASETS:
                run_name = f"{tracker_name}/{dataset_name}"
                row = evaluate_run(tracker_name, dataset_name, cam_list)
                results[run_name] = row
        print_comparison_table(results)
    else:
        tracker_name = args.tracker or "bytetrack"
        dataset_name = args.dataset or "val"
        row = evaluate_run(tracker_name, dataset_name, cam_list)
        if row:
            pct = lambda v: f"{v*100:.2f}%"
            print(f"\n  Summary: MOTA={pct(row['mota'])}  MOTP={pct(row['motp'])}  "
                  f"IDF1={pct(row['idf1'])}  IDSW={int(row['num_switches'])}")


if __name__ == "__main__":
    main()
