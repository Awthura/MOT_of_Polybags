"""
Drop "whole-tile fallback" boxes from the tiled+iterative autolabel run --
a third variant of the same "fall back to boxing everything instead of
nothing" pathology already found and fixed at the frame level (is_fullframe)
and the belt level (drop_wholebelt_boxes.py), this time happening *inside
individual tile crops*: on an empty/plain-textured tile, the model
sometimes falls back to a box spanning almost the entire tile instead of
predicting no detections.

Found via cross-validation: after the full tiled+iterative dataset run,
`basler_1`'s LA-agreement against YOLO collapsed to 17.4% (from 63.1% pre-
tiling) despite belt-masking barely touching its boxes (0.7% dropped) --
meaning the extra boxes were landing squarely *on* the belt, not on known
static clutter. Manually inspecting the frames with the biggest box-count
jumps found several completely empty belts with 3-5 new boxes each, and the
box coordinates matched the tile grid rectangles almost exactly (measured
IoU=0.9975 against the actual tile bounds for one case) -- confirming these
are tile-shaped fallback boxes, not real detections or genuine fragments of
real objects (which is why Weighted Box Fusion was not the right fix here:
WBF reconciles multiple *real* overlapping detections into one consolidated
box, it doesn't reject purely spurious detections on empty content).

Since every image in this dataset is the same 1280x720 resolution and
autolabel_tiled.py's tiling grid is fixed, the tile rectangles are known and
identical for every image -- this can run as pure local post-processing on
the already-downloaded labels, no cluster re-run needed.

Input:  unlabelled_autolabels_v4_tiled/labels_final/ (post whole-belt-fix)
Output: unlabelled_autolabels_v4_tiled/labels_final_v2/

Usage:
    python drop_wholetile_boxes.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "locate_anything"))
from autolabel_tiled import make_tiles  # noqa: E402

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
LABELS_IN = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_final"
LABELS_OUT = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_final_v2"
OVERLAYS_CHANGED = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "overlays_wholetile_fix"

IMG_W, IMG_H = 1280, 720
TILES = make_tiles(IMG_W, IMG_H)  # fixed, same for every image at this resolution
TILE_IOU_THRESH = 0.85  # measured ~0.9975 for confirmed whole-tile fallbacks


def parse_label(path):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = map(float, parts)
        boxes.append((cx, cy, w, h))
    return boxes


def to_yolo_line(b):
    cx, cy, w, h = b
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def box_iou_with_tile(box, tile):
    cx, cy, w, h = box
    bx1, by1 = (cx - w / 2) * IMG_W, (cy - h / 2) * IMG_H
    bx2, by2 = (cx + w / 2) * IMG_W, (cy + h / 2) * IMG_H
    tx1, ty1, tx2, ty2 = tile
    ix1, iy1 = max(bx1, tx1), max(by1, ty1)
    ix2, iy2 = min(bx2, tx2), min(by2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    area_t = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = area_b + area_t - inter
    return inter / union if union > 0 else 0.0


def is_wholetile_fallback(box):
    return any(box_iou_with_tile(box, tile) >= TILE_IOU_THRESH for tile in TILES)


def draw_boxes(img_bgr, boxes, color=(0, 0, 220)):
    h, w = img_bgr.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)


def find_image(cam_name, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = UNLABELLED_IMAGES / cam_name / (stem + ext)
        if p.exists():
            return p
    return None


def main():
    stats = {"total_boxes_before": 0, "total_boxes_after": 0,
              "n_dropped": 0, "n_images_changed": 0, "per_camera": {}}

    for cam_dir in sorted(LABELS_IN.iterdir()):
        cam_name = cam_dir.name
        out_dir = LABELS_OUT / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cam_stats = stats["per_camera"].setdefault(cam_name, {"dropped": 0, "images_changed": 0})

        for label_path in sorted(cam_dir.glob("*.txt")):
            boxes = parse_label(label_path)
            kept = [b for b in boxes if not is_wholetile_fallback(b)]
            n_dropped = len(boxes) - len(kept)

            stats["total_boxes_before"] += len(boxes)
            stats["total_boxes_after"] += len(kept)
            stats["n_dropped"] += n_dropped

            (out_dir / label_path.name).write_text(
                "\n".join(to_yolo_line(b) for b in kept) + ("\n" if kept else ""))

            if n_dropped > 0:
                stats["n_images_changed"] += 1
                cam_stats["dropped"] += n_dropped
                cam_stats["images_changed"] += 1
                img_path = find_image(cam_name, label_path.stem)
                if img_path is not None:
                    img = Image.open(img_path).convert("RGB")
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    draw_boxes(cv_img, kept)
                    changed_dir = OVERLAYS_CHANGED / cam_name
                    changed_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(changed_dir / (label_path.stem + ".jpg")), cv_img)

    (DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "wholetile_fix_stats.json").write_text(
        json.dumps(stats, indent=2))

    print(f"Before: {stats['total_boxes_before']} boxes")
    print(f"After:  {stats['total_boxes_after']} boxes")
    print(f"Dropped as whole-tile fallback: {stats['n_dropped']}")
    print(f"Images changed: {stats['n_images_changed']}")
    for cam, s in stats["per_camera"].items():
        if s["dropped"]:
            print(f"  {cam}: {s['dropped']} dropped across {s['images_changed']} images")
    print(f"\nFinal labels -> {LABELS_OUT}")


if __name__ == "__main__":
    main()
