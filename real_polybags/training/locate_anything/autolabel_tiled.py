"""
Tiled + iterative inference for LocateAnything-3B, targeting the one
residual issue found during manual review of the real_v4_round2 autolabels:
occasional missed bags in the densest, most-overlapping clusters (e.g. 9
visible bags, 8 boxed) -- confirmed not a post-processing bug (YOLO missed
the same bag independently), so nothing in a single full-frame detection
pass can recover it. See AUTOLABEL_REPORT.md and the plan this implements
for the research behind this approach (SAHI-style tiling + IterDet-style
iterative re-detection, both well-established, architecture-agnostic,
zero-retraining techniques for exactly this failure mode).

Two complementary techniques, applied on top of the existing single-shot
full-frame detection:

1. Tiled inference (SAHI-style): slice the image into overlapping tiles,
   run detection on each tile independently (fewer, larger-looking objects
   per call), remap tile-local boxes back to full-image coordinates, and
   merge with the full-frame pass via NMS. Directly reduces objects-per-call
   in the densest frames.

2. Iterative re-detection (IterDet-style): after the merged first pass,
   paint over the already-detected regions (neutral gray, matching belt
   tone) in a copy of the image and re-run full-frame detection on that
   residual image once or twice, merging in any genuinely new boxes (IoU
   below threshold against everything already found). Catches stragglers
   the model terminated before describing.

Usage:
    # Single-image validation (no dataset dir needed) -- use this first:
    python autolabel_tiled.py --eagle-dir ~/Eagle --model ~/runs_locateanything_lora/real_v4_round2 \
        --test-image /path/to/frame.jpg --out-dir /tmp/tiled_test

    # Full dataset-dir mode, same interface as autolabel_unlabelled.py:
    python autolabel_tiled.py --eagle-dir ~/Eagle --model ~/runs_locateanything_lora/real_v4_round2 \
        --dataset-dir ~/MOT_of_Polybags/real_polybags/dataset/unlabelled \
        --out-dir ~/MOT_of_Polybags/real_polybags/dataset/unlabelled_autolabels_tiled
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_CATEGORY = "translucent bubble-wrap polybag"

GRID = (2, 2)          # (cols, rows)
TILE_OVERLAP = 0.20    # fraction of tile size overlapping with neighbors
MERGE_IOU_THRESH = 0.5
# A single object straddling a tile boundary can produce a partial sub-box in
# one tile that doesn't reach MERGE_IOU_THRESH against the full-image box for
# the same object (found during validation: a real tile-boundary duplicate
# had IoU=0.46, just under 0.5). Symmetric IoU penalizes area mismatch, but a
# small box that's mostly *contained* within a larger one is a much stronger
# duplicate signal regardless of the size difference -- the same duplicate
# pair had IoS (intersection / smaller-box-area) = 0.92, while a confirmed
# genuinely-distinct overlapping bag pair measured only 0.61 IoS, a clean
# margin either side of this threshold.
MERGE_IOS_THRESH = 0.85
MAX_ITERATIVE_ROUNDS = 2
PAINT_COLOR = (110, 110, 110)  # neutral gray, close to typical belt tone


def _corners(b):
    return b["x1"], b["y1"], b["x2"], b["y2"]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = _corners(a)
    bx1, by1, bx2, by2 = _corners(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ios_xyxy(a, b):
    """Intersection over smaller-box area -- catches partial tile-boundary
    duplicates that symmetric IoU misses (see MERGE_IOS_THRESH)."""
    ax1, ay1, ax2, ay2 = _corners(a)
    bx1, by1, bx2, by2 = _corners(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def is_duplicate(a, b, iou_thresh=MERGE_IOU_THRESH, ios_thresh=MERGE_IOS_THRESH):
    return iou_xyxy(a, b) > iou_thresh or ios_xyxy(a, b) > ios_thresh


def nms_merge(boxes, iou_thresh=MERGE_IOU_THRESH, ios_thresh=MERGE_IOS_THRESH):
    """Greedy dedup, first-seen wins (no confidence scores available)."""
    kept = []
    for b in boxes:
        if all(not is_duplicate(b, k, iou_thresh, ios_thresh) for k in kept):
            kept.append(b)
    return kept


def make_tiles(w, h, grid=GRID, overlap=TILE_OVERLAP):
    cols, rows = grid
    tile_w = w / (cols - (cols - 1) * overlap) if cols > 1 else w
    tile_h = h / (rows - (rows - 1) * overlap) if rows > 1 else h
    stride_x = tile_w * (1 - overlap)
    stride_y = tile_h * (1 - overlap)

    tiles = []
    for row in range(rows):
        for col in range(cols):
            x1 = min(col * stride_x, w - tile_w) if cols > 1 else 0
            y1 = min(row * stride_y, h - tile_h) if rows > 1 else 0
            x2 = min(x1 + tile_w, w)
            y2 = min(y1 + tile_h, h)
            tiles.append((int(x1), int(y1), int(x2), int(y2)))
    return tiles


def detect_on_crop(worker, img, prompt_categories, crop_box):
    x1, y1, x2, y2 = crop_box
    crop = img.crop((x1, y1, x2, y2))
    cw, ch = crop.size
    result = worker.detect(crop, prompt_categories)
    boxes = worker.parse_boxes(result["answer"], cw, ch)
    # remap to full-image coordinates
    for b in boxes:
        b["x1"] += x1
        b["y1"] += y1
        b["x2"] += x1
        b["y2"] += y1
    return boxes


def paint_over(img, boxes, color=PAINT_COLOR):
    arr = np.array(img).copy()
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        arr[max(0, y1):y2, max(0, x1):x2] = color
    return Image.fromarray(arr)


def detect_enhanced(worker, img, prompt_categories, use_tiling=True, use_iterative=True,
                     grid=GRID, overlap=TILE_OVERLAP, max_rounds=MAX_ITERATIVE_ROUNDS,
                     debug=None):
    w, h = img.size

    # 1. Baseline full-frame pass
    result = worker.detect(img, prompt_categories)
    all_boxes = worker.parse_boxes(result["answer"], w, h)
    if debug is not None:
        debug["full_frame"] = len(all_boxes)

    # 2. Tiled pass
    if use_tiling:
        for tile in make_tiles(w, h, grid, overlap):
            all_boxes.extend(detect_on_crop(worker, img, prompt_categories, tile))
        if debug is not None:
            debug["after_tiling_raw"] = len(all_boxes)

    merged = nms_merge(all_boxes)
    if debug is not None:
        debug["after_tiling_merged"] = len(merged)

    # 3. Iterative re-detection on the residual (already-detected regions painted over)
    if use_iterative:
        for round_idx in range(max_rounds):
            residual_img = paint_over(img, merged)
            result = worker.detect(residual_img, prompt_categories)
            candidate_boxes = worker.parse_boxes(result["answer"], w, h)
            new_boxes = [b for b in candidate_boxes
                         if all(not is_duplicate(b, k) for k in merged)]
            if debug is not None:
                debug[f"iterative_round_{round_idx + 1}_new"] = len(new_boxes)
            if not new_boxes:
                break
            merged.extend(new_boxes)
            merged = nms_merge(merged)

    if debug is not None:
        debug["final"] = len(merged)
    return merged


def draw_boxes(img_bgr, boxes, color=(0, 0, 220)):
    for b in boxes:
        pt1 = (int(b["x1"]), int(b["y1"]))
        pt2 = (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img_bgr, pt1, pt2, color, 2)


def boxes_to_yolo_lines(boxes, img_w, img_h):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def load_worker(eagle_dir, model, device):
    sys.path.insert(0, str(Path(eagle_dir) / "Embodied"))
    from locateanything_worker import LocateAnythingWorker  # noqa: E402
    print(f"Loading {model} on {device} ...")
    t0 = time.time()
    worker = LocateAnythingWorker(model, device=device)
    print(f"Loaded in {time.time() - t0:.1f}s")
    return worker


def run_single_image(worker, img_path, out_dir, prompt_categories):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    debug = {}
    t0 = time.time()
    boxes = detect_enhanced(worker, img, prompt_categories, debug=debug)
    dt = time.time() - t0

    label_path = out_dir / (Path(img_path).stem + "_tiled.txt")
    label_path.write_text("\n".join(boxes_to_yolo_lines(boxes, w, h)) + ("\n" if boxes else ""))

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    draw_boxes(cv_img, boxes)
    overlay_path = out_dir / (Path(img_path).stem + "_tiled_overlay.jpg")
    cv2.imwrite(str(overlay_path), cv_img)

    print(f"{img_path}: {debug} ({dt:.1f}s)")
    print(f"Labels  -> {label_path}")
    print(f"Overlay -> {overlay_path}")
    return boxes, debug


def run_dataset(worker, dataset_dir, out_dir, prompt_categories, summary_every=100):
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    labels_root = out_dir / "labels"
    overlays_root = out_dir / "overlays"
    labels_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)

    cam_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    stats = {"per_camera": {}, "total_images": 0, "total_boxes": 0,
             "total_infer_seconds": 0.0, "skipped_existing": 0, "started": time.time()}
    n_done_since_summary = 0

    def write_summary():
        stats["elapsed_seconds"] = time.time() - stats["started"]
        (out_dir / "summary.json").write_text(json.dumps(stats, indent=2))

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        images = sorted([p for p in cam_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        cam_labels_dir = labels_root / cam_name
        cam_overlays_dir = overlays_root / cam_name
        cam_labels_dir.mkdir(parents=True, exist_ok=True)
        cam_overlays_dir.mkdir(parents=True, exist_ok=True)
        cam_stats = stats["per_camera"].setdefault(
            cam_name, {"n_images": 0, "n_boxes": 0, "infer_seconds": 0.0})

        print(f"\n[{cam_name}] {len(images)} images")
        for img_path in images:
            label_path = cam_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                stats["skipped_existing"] += 1
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            t0 = time.time()
            boxes = detect_enhanced(worker, img, prompt_categories)
            dt = time.time() - t0

            label_path.write_text(
                "\n".join(boxes_to_yolo_lines(boxes, w, h)) + ("\n" if boxes else ""))
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            draw_boxes(cv_img, boxes)
            cv2.imwrite(str(cam_overlays_dir / (img_path.stem + ".jpg")), cv_img)

            cam_stats["n_images"] += 1
            cam_stats["n_boxes"] += len(boxes)
            cam_stats["infer_seconds"] += dt
            stats["total_images"] += 1
            stats["total_boxes"] += len(boxes)
            stats["total_infer_seconds"] += dt
            print(f"  {img_path.name}: boxes={len(boxes)} ({dt:.1f}s)")

            n_done_since_summary += 1
            if n_done_since_summary >= summary_every:
                write_summary()
                n_done_since_summary = 0

    write_summary()
    print(f"\nDone. total_images={stats['total_images']} total_boxes={stats['total_boxes']}")
    print(f"Labels -> {labels_root}")
    print(f"Overlays -> {overlays_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--category", default=DEFAULT_CATEGORY)
    ap.add_argument("--test-image", default=None,
                     help="Single-image validation mode -- run this first")
    ap.add_argument("--dataset-dir", default=None,
                     help="Full dataset-dir mode (parent of the 5 camera folders)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if not args.test_image and not args.dataset_dir:
        raise SystemExit("Provide either --test-image (validation) or --dataset-dir (full run)")

    prompt_categories = [args.category]
    worker = load_worker(args.eagle_dir, args.model, args.device)

    if args.test_image:
        run_single_image(worker, args.test_image, args.out_dir, prompt_categories)
    else:
        run_dataset(worker, args.dataset_dir, args.out_dir, prompt_categories)


if __name__ == "__main__":
    main()
