"""
Same whole-belt-fallback fix as drop_wholebelt_boxes.py, applied to the
tiled+iterative autolabel run's cleaned+masked labels. Belt-coverage
geometry and thresholds are model/inference-strategy-independent -- see
drop_wholebelt_boxes.py for the full rationale and threshold validation.

Input:  unlabelled_autolabels_v4_tiled/labels_cleaned_masked/
Output: unlabelled_autolabels_v4_tiled/labels_final/

Usage:
    python drop_wholebelt_boxes_tiled.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from drop_wholebelt_boxes import parse_label, to_yolo_line, is_wholebelt_fallback, draw_boxes, find_image

DATASET_ROOT = Path(__file__).resolve().parent
LABELS_IN = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_cleaned_masked"
LABELS_OUT = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_final"
OVERLAYS_CHANGED = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "overlays_wholebelt_fix"
BELT_MASKS = DATASET_ROOT / "belt_masks"


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

    (DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "wholebelt_fix_stats.json").write_text(
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
