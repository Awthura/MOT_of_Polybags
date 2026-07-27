"""
Drop "whole-belt fallback" degenerate boxes from the v4 cleaned+masked
autolabels -- a pattern found by manual visual review (not caught by the
existing full-frame filter) where, on an empty belt, the model sometimes
falls back to drawing one giant box spanning almost the entire *belt*
region instead of predicting no detections. This isn't a literal
whole-FRAME box (analyze_autolabels.py's is_fullframe, w>=0.95 and h>=0.95,
already catches those) -- it's sized to each camera's belt geometry, e.g.
lucid's version is w=0.73/h=1.00, which slips through since w<0.95.

Generalizes is_fullframe using the belt masks (build_belt_masks.py) we
already have: a box is a whole-belt fallback if it's large (>=35% of frame
area -- comfortably bigger than any real single- or few-bag box seen in
practice) AND covers almost the entire belt mask's pixel area (>=90% --
checked against 405 manually-verified candidates: 404/405 covered
>=99.6% of the belt, the one exception at 64.6% was confirmed by visual
inspection to be a real, correctly-boxed bag, giving a wide safety margin
on both sides of the 90% cutoff).

Input:  unlabelled_autolabels_v4/labels_cleaned_masked/
Output: unlabelled_autolabels_v4/labels_final/ (+ overlays_final_v2/ for
        the images that changed)

Usage:
    python drop_wholebelt_boxes.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
BELT_MASKS = DATASET_ROOT / "belt_masks"
LABELS_IN = DATASET_ROOT / "unlabelled_autolabels_v4" / "labels_cleaned_masked"
LABELS_OUT = DATASET_ROOT / "unlabelled_autolabels_v4" / "labels_final"
OVERLAYS_CHANGED = DATASET_ROOT / "unlabelled_autolabels_v4" / "overlays_wholebelt_fix"

AREA_THRESH = 0.35
BELT_COVERAGE_THRESH = 0.90


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


def is_wholebelt_fallback(box, mask):
    cx, cy, w, h = box
    area = w * h
    if area < AREA_THRESH:
        return False
    h_img, w_img = mask.shape
    x1 = max(0, int((cx - w / 2) * w_img))
    y1 = max(0, int((cy - h / 2) * h_img))
    x2 = min(w_img, int((cx + w / 2) * w_img))
    y2 = min(h_img, int((cy + h / 2) * h_img))
    belt_total = (mask > 0).sum()
    if belt_total == 0:
        return False
    belt_covered = (mask[y1:y2, x1:x2] > 0).sum()
    return (belt_covered / belt_total) >= BELT_COVERAGE_THRESH


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
        mask = cv2.imread(str(BELT_MASKS / f"{cam_name}.png"), cv2.IMREAD_GRAYSCALE)
        out_dir = LABELS_OUT / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cam_stats = stats["per_camera"].setdefault(cam_name, {"dropped": 0, "images_changed": 0})

        for label_path in sorted(cam_dir.glob("*.txt")):
            boxes = parse_label(label_path)
            kept = [b for b in boxes if not is_wholebelt_fallback(b, mask)]
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

    (DATASET_ROOT / "unlabelled_autolabels_v4" / "wholebelt_fix_stats.json").write_text(
        json.dumps(stats, indent=2))

    print(f"Before: {stats['total_boxes_before']} boxes")
    print(f"After:  {stats['total_boxes_after']} boxes")
    print(f"Dropped as whole-belt fallback: {stats['n_dropped']}")
    print(f"Images changed: {stats['n_images_changed']}")
    for cam, s in stats["per_camera"].items():
        if s["dropped"]:
            print(f"  {cam}: {s['dropped']} dropped across {s['images_changed']} images")
    print(f"\nFinal labels -> {LABELS_OUT}")
    print(f"Changed-image overlays -> {OVERLAYS_CHANGED}")


if __name__ == "__main__":
    main()
