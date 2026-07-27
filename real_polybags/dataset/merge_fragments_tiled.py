"""
Merge surviving tile-boundary fragment boxes in the tiled autolabel run.

`autolabel_tiled.py` merges tile-overlap duplicates with an IoU>0.5 OR
IoS>0.85 test (IoS = intersection over the *smaller* box's area, needed
because a partial fragment mostly contained in a larger box has low
symmetric IoU but high containment). That IoS threshold was calibrated on a
single frame against one confirmed duplicate (IoS=0.92) and one confirmed
genuinely-distinct stacked pair (IoS=0.61) -- a wide gap with only two
points in it, so 0.85 was picked conservatively.

At full-dataset scale that turns out to leave fragments behind: the tiled
set has 191 box pairs sitting in the IoS 0.70-0.85 band (vs 13 in the
non-tiled set), i.e. right under the cut. Sampling that band visually,
4/4 inspected pairs were duplicate boxes on a single bag (one of the two
being a thin partial sliver). Sampling the 0.60-0.70 band, the pairs were
instead genuinely distinct bags in a dense pile -- merging those would be
wrong. So the real duplicate/distinct boundary sits at ~0.70, not 0.85.

This pass re-merges at IoS>=0.70, keeping the LARGER box of each pair and
dropping the contained fragment (the visual evidence is consistently that
the smaller box is a partial sliver of the same bag, not an object).

Runs as pure local post-processing -- no cluster or model call needed.

Input:  unlabelled_autolabels_v4_tiled/labels_final_v2/ (post whole-tile-fix)
Output: unlabelled_autolabels_v4_tiled/labels_final_v3/

Usage:
    python3 merge_fragments_tiled.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
LABELS_IN = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_final_v2"
LABELS_OUT = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_final_v3"
OVERLAYS_CHANGED = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "overlays_fragment_fix"

MERGE_IOS_THRESH = 0.70  # calibrated: >=0.70 duplicates, 0.60-0.70 distinct bags


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


def corners(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def area(c):
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])


def ios(a, b):
    ca, cb = corners(a), corners(b)
    ix1, iy1 = max(ca[0], cb[0]), max(ca[1], cb[1])
    ix2, iy2 = min(ca[2], cb[2]), min(ca[3], cb[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(area(ca), area(cb))
    return inter / smaller if smaller > 0 else 0.0


def merge_fragments(boxes, thresh=MERGE_IOS_THRESH):
    """Keep the largest box of each mutually-contained group, drop fragments."""
    order = sorted(range(len(boxes)), key=lambda i: -area(corners(boxes[i])))
    kept, dropped = [], 0
    for i in order:
        if any(ios(boxes[i], k) >= thresh for k in kept):
            dropped += 1
            continue
        kept.append(boxes[i])
    return kept, dropped


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
    stats = {"ios_thresh": MERGE_IOS_THRESH, "total_boxes_before": 0,
             "total_boxes_after": 0, "n_dropped": 0, "n_images_changed": 0,
             "per_camera": {}}

    for cam_dir in sorted(d for d in LABELS_IN.iterdir() if d.is_dir()):
        cam_name = cam_dir.name
        out_dir = LABELS_OUT / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cam_stats = stats["per_camera"].setdefault(
            cam_name, {"dropped": 0, "images_changed": 0})

        for label_path in sorted(cam_dir.glob("*.txt")):
            boxes = parse_label(label_path)
            kept, n_dropped = merge_fragments(boxes)

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

    (DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "fragment_fix_stats.json").write_text(
        json.dumps(stats, indent=2))

    print(f"IoS threshold: {MERGE_IOS_THRESH}")
    print(f"Before: {stats['total_boxes_before']} boxes")
    print(f"After:  {stats['total_boxes_after']} boxes")
    print(f"Dropped as tile fragment: {stats['n_dropped']}")
    print(f"Images changed: {stats['n_images_changed']}")
    for cam, s in stats["per_camera"].items():
        if s["dropped"]:
            print(f"  {cam}: {s['dropped']} dropped across {s['images_changed']} images")
    print(f"\nFinal labels -> {LABELS_OUT}")


if __name__ == "__main__":
    main()
