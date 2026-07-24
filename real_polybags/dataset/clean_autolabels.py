"""
Post-processing cleanup for the raw LocateAnything-3B autolabel output
(unlabelled_autolabels/labels/), run entirely locally (no cluster/model
needed -- pure label-file post-processing).

Fixes the two problems found by inspection:
  1. Degenerate "whole image" boxes on frames with no real polybag
     (w and h both >= FULLFRAME_THRESH of the frame) -> dropped.
  2. Invalid boxes (w<=0 or h<=0, from a decoder drift that occasionally
     inverts the box coordinates) -> dropped.
  3. Duplicate/near-duplicate boxes for the same physical object (decoder
     re-describing one object several times with a slowly sliding border)
     -> collapsed via greedy sequential IoU-based NMS (order-preserving,
     no confidence scores are available so first-seen wins).

Writes cleaned labels to labels_cleaned/ (same structure, one file per
image, always written even if empty -- "no detections" is a valid answer).
Only re-renders an overlay (into overlays_cleaned/) for images where the
box set actually changed, to keep disk usage down; unaffected images keep
using the existing overlays/ from the raw run.

Usage:
    python clean_autolabels.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from analyze_autolabels import (
    LABELS_ROOT, parse_label, is_invalid, is_fullframe, iou, FULLFRAME_THRESH,
    find_drift_chains,
)

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
OUT_ROOT = DATASET_ROOT / "unlabelled_autolabels"
LABELS_CLEANED = OUT_ROOT / "labels_cleaned"
OVERLAYS_CLEANED = OUT_ROOT / "overlays_cleaned"

NMS_IOU_THRESH = 0.5


def nms_dedup(boxes):
    kept = []
    for b in boxes:
        if all(iou(b, k) <= NMS_IOU_THRESH for k in kept):
            kept.append(b)
    return kept


def collapse_drift_chains(boxes):
    """Collapse runs of consecutive (file-order) boxes that each have high
    IoU with their immediate predecessor -- a slow drift still clears this
    pairwise test at every step even though the first and last box of a long
    chain may have near-zero mutual IoU, so a "compare against all kept"
    NMS alone won't fully collapse it. One representative (the first box
    seen) is kept per chain."""
    chains = find_drift_chains(boxes)
    return [boxes[chain[0]] for chain in chains]


def clean_boxes(boxes):
    boxes = [b for b in boxes if not is_invalid(b)]
    boxes = [b for b in boxes if not is_fullframe(b)]
    boxes = collapse_drift_chains(boxes)
    boxes = nms_dedup(boxes)
    return boxes


def to_yolo_line(b):
    cx, cy, w, h = b
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_image(cam_name, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = UNLABELLED_IMAGES / cam_name / (stem + ext)
        if p.exists():
            return p
    return None


def draw_boxes(img_bgr, boxes, color=(0, 0, 220)):
    h, w = img_bgr.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)


def main():
    label_files = sorted(LABELS_ROOT.rglob("*.txt"))
    print(f"Processing {len(label_files)} label files...")

    stats = {
        "total_images": 0,
        "total_boxes_before": 0,
        "total_boxes_after": 0,
        "n_invalid_dropped": 0,
        "n_fullframe_dropped": 0,
        "n_drift_chain_dropped": 0,
        "n_nms_dropped": 0,
        "n_images_changed": 0,
        "n_overlays_rendered": 0,
    }

    for label_path in label_files:
        cam_name = label_path.parent.name
        stem = label_path.stem

        boxes = parse_label(label_path)
        n_before = len(boxes)

        no_invalid = [b for b in boxes if not is_invalid(b)]
        n_invalid = n_before - len(no_invalid)

        no_fullframe = [b for b in no_invalid if not is_fullframe(b)]
        n_fullframe = len(no_invalid) - len(no_fullframe)

        chain_collapsed = collapse_drift_chains(no_fullframe)
        n_chain = len(no_fullframe) - len(chain_collapsed)

        deduped = nms_dedup(chain_collapsed)
        n_nms = len(chain_collapsed) - len(deduped)

        stats["total_images"] += 1
        stats["total_boxes_before"] += n_before
        stats["total_boxes_after"] += len(deduped)
        stats["n_invalid_dropped"] += n_invalid
        stats["n_fullframe_dropped"] += n_fullframe
        stats["n_drift_chain_dropped"] += n_chain
        stats["n_nms_dropped"] += n_nms

        cam_out_dir = LABELS_CLEANED / cam_name
        cam_out_dir.mkdir(parents=True, exist_ok=True)
        out_path = cam_out_dir / label_path.name
        out_path.write_text(
            "\n".join(to_yolo_line(b) for b in deduped) + ("\n" if deduped else ""))

        changed = (n_invalid + n_fullframe + n_chain + n_nms) > 0
        if changed:
            stats["n_images_changed"] += 1
            img_path = find_image(cam_name, stem)
            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                draw_boxes(cv_img, deduped)
                cam_overlay_dir = OVERLAYS_CLEANED / cam_name
                cam_overlay_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cam_overlay_dir / (stem + ".jpg")), cv_img)
                stats["n_overlays_rendered"] += 1

    (OUT_ROOT / "clean_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nBefore: {stats['total_boxes_before']} boxes")
    print(f"After:  {stats['total_boxes_after']} boxes")
    print(f"  dropped as invalid (negative w/h):   {stats['n_invalid_dropped']}")
    print(f"  dropped as full-frame degenerate:    {stats['n_fullframe_dropped']}")
    print(f"  dropped as drift-chain duplicate:     {stats['n_drift_chain_dropped']}")
    print(f"  dropped as NMS duplicate (IoU>{NMS_IOU_THRESH}):  {stats['n_nms_dropped']}")
    print(f"\nImages changed: {stats['n_images_changed']} / {stats['total_images']}")
    print(f"Overlays rendered for review: {stats['n_overlays_rendered']}")
    print(f"\nCleaned labels  -> {LABELS_CLEANED}")
    print(f"Review overlays -> {OVERLAYS_CLEANED}")
    print(f"Stats           -> {OUT_ROOT / 'clean_stats.json'}")


if __name__ == "__main__":
    main()
