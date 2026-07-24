"""
Local (no cluster needed) investigation of the raw LocateAnything-3B autolabel
output in unlabelled_autolabels/labels/. Two known problems reported by
inspection of overlays:

  1. Some frames get a single box spanning almost the whole image, even when
     no polybag is actually present (degenerate "whole image" fallback).
  2. Some frames have a handful of real boxes plus a long run of near-duplicate
     boxes for the same physical object, each shifted by a tiny amount from
     the previous one (a decoder drift/loop pathology in the autoregressive
     "parallel box decoding" -- the model keeps re-describing the same object
     with a slowly sliding border instead of terminating).

This script quantifies both across the full 2,926-image set and prints
representative examples so the drift pattern can be inspected directly.
"""

import math
from pathlib import Path

LABELS_ROOT = Path(__file__).resolve().parent / "unlabelled_autolabels" / "labels"

FULLFRAME_THRESH = 0.95   # w and h both >= this -> "whole image" box
DRIFT_IOU_THRESH = 0.5    # consecutive-box IoU above this -> same drift chain


def parse_label(path: Path):
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append((cx, cy, w, h))
    return boxes


def to_corners(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_corners(a)
    bx1, by1, bx2, by2 = to_corners(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_invalid(b):
    _, _, w, h = b
    return w <= 0 or h <= 0 or math.isnan(w) or math.isnan(h)


def is_fullframe(b):
    _, _, w, h = b
    return w >= FULLFRAME_THRESH and h >= FULLFRAME_THRESH


def find_drift_chains(boxes):
    """Group consecutive boxes (in file order) into chains where each box
    has IoU > DRIFT_IOU_THRESH with the previous box in the same chain.
    Returns list of chains (each a list of box indices)."""
    chains = []
    current = []
    for i, b in enumerate(boxes):
        if is_invalid(b):
            continue
        if current and iou(boxes[current[-1]], b) > DRIFT_IOU_THRESH:
            current.append(i)
        else:
            if current:
                chains.append(current)
            current = [i]
    if current:
        chains.append(current)
    return chains


def main():
    label_files = sorted(LABELS_ROOT.rglob("*.txt"))
    print(f"Total label files: {len(label_files)}")

    n_invalid_boxes = 0
    n_fullframe_boxes = 0
    n_images_with_fullframe = 0
    n_images_with_invalid = 0
    n_images_with_drift_chain = 0  # chain length >= 4
    total_boxes = 0
    total_drift_chain_boxes = 0  # boxes that belong to a chain of length >= 4
    longest_chains = []  # (length, path)

    for path in label_files:
        boxes = parse_label(path)
        total_boxes += len(boxes)

        invalid = [b for b in boxes if is_invalid(b)]
        if invalid:
            n_invalid_boxes += len(invalid)
            n_images_with_invalid += 1

        fullframe = [b for b in boxes if is_fullframe(b) and not is_invalid(b)]
        if fullframe:
            n_fullframe_boxes += len(fullframe)
            n_images_with_fullframe += 1

        chains = find_drift_chains(boxes)
        long_chains = [c for c in chains if len(c) >= 4]
        if long_chains:
            n_images_with_drift_chain += 1
            for c in long_chains:
                total_drift_chain_boxes += len(c)
        max_chain = max((len(c) for c in chains), default=0)
        if max_chain >= 4:
            longest_chains.append((max_chain, path))

    longest_chains.sort(reverse=True)

    print(f"\nTotal boxes across dataset: {total_boxes}")
    print(f"\n--- Invalid boxes (w<=0 or h<=0, i.e. inverted coords) ---")
    print(f"  {n_invalid_boxes} boxes across {n_images_with_invalid} images")
    print(f"\n--- Full-frame degenerate boxes (w>={FULLFRAME_THRESH}, h>={FULLFRAME_THRESH}) ---")
    print(f"  {n_fullframe_boxes} boxes across {n_images_with_fullframe} images")
    print(f"\n--- Drift chains (>=4 consecutive boxes with IoU>{DRIFT_IOU_THRESH}) ---")
    print(f"  {n_images_with_drift_chain} images affected, "
          f"{total_drift_chain_boxes} boxes belong to such chains "
          f"({total_drift_chain_boxes/total_boxes*100:.1f}% of all boxes)")
    print(f"\n  Top 15 longest drift chains:")
    for length, path in longest_chains[:15]:
        print(f"    {length:4d}  {path.relative_to(LABELS_ROOT)}")


if __name__ == "__main__":
    main()
