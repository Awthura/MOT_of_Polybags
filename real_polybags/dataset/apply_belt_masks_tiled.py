"""
Apply the same per-camera belt masks (build_belt_masks.py, model/inference-
strategy-independent -- belt geometry doesn't change) to the tiled+iterative
autolabel run's cleaned labels. YOLO's own masked labels don't need
regenerating -- they're already in cross_validation/yolo_labels_masked/ and
don't depend on which LA run is being cross-validated.

Usage:
    python apply_belt_masks_tiled.py
"""

from pathlib import Path

import cv2

from apply_belt_masks import filter_dir

DATASET_ROOT = Path(__file__).resolve().parent
BELT_MASKS = DATASET_ROOT / "belt_masks"
LA_LABELS_CLEANED_TILED = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_cleaned"
LA_LABELS_MASKED_TILED = DATASET_ROOT / "unlabelled_autolabels_v4_tiled" / "labels_cleaned_masked"


def main():
    cam_names = sorted(p.stem for p in BELT_MASKS.glob("*.png"))
    grand_total = grand_kept = 0
    for cam in cam_names:
        mask = cv2.imread(str(BELT_MASKS / f"{cam}.png"), cv2.IMREAD_GRAYSCALE)
        la_dir = LA_LABELS_CLEANED_TILED / cam
        if not la_dir.exists():
            continue
        total, kept = filter_dir(la_dir, mask, LA_LABELS_MASKED_TILED / cam)
        grand_total += total
        grand_kept += kept
        print(f"[{cam}] LA tiled: {total} -> {kept} "
              f"({total - kept} dropped, {(total - kept) / total:.1%})" if total else f"[{cam}] LA tiled: 0 boxes")

    print("\n=== Totals (tiled+iterative) ===")
    print(f"LA tiled: {grand_total} -> {grand_kept} "
          f"({grand_total - grand_kept} dropped, {(grand_total - grand_kept) / grand_total:.1%})")


if __name__ == "__main__":
    main()
