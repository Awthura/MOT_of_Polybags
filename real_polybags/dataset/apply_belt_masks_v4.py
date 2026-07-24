"""
Apply the same per-camera belt masks (build_belt_masks.py, model-independent
-- geometry doesn't change between LA checkpoint versions) to the round-2
(real_v4_round2) cleaned autolabels. YOLO's own masked labels don't need
regenerating -- they're already in cross_validation/yolo_labels_masked/ and
don't depend on which LA model produced the labels being cross-validated.

Usage:
    python apply_belt_masks_v4.py
"""

from pathlib import Path

import cv2

from apply_belt_masks import filter_dir

DATASET_ROOT = Path(__file__).resolve().parent
BELT_MASKS = DATASET_ROOT / "belt_masks"
LA_LABELS_CLEANED_V4 = DATASET_ROOT / "unlabelled_autolabels_v4" / "labels_cleaned"
LA_LABELS_MASKED_V4 = DATASET_ROOT / "unlabelled_autolabels_v4" / "labels_cleaned_masked"


def main():
    cam_names = sorted(p.stem for p in BELT_MASKS.glob("*.png"))
    grand_total = grand_kept = 0
    for cam in cam_names:
        mask = cv2.imread(str(BELT_MASKS / f"{cam}.png"), cv2.IMREAD_GRAYSCALE)
        la_dir = LA_LABELS_CLEANED_V4 / cam
        if not la_dir.exists():
            continue
        total, kept = filter_dir(la_dir, mask, LA_LABELS_MASKED_V4 / cam)
        grand_total += total
        grand_kept += kept
        print(f"[{cam}] LA v4: {total} -> {kept} "
              f"({total - kept} dropped, {(total - kept) / total:.1%})" if total else f"[{cam}] LA v4: 0 boxes")

    print("\n=== Totals (real_v4_round2) ===")
    print(f"LA v4: {grand_total} -> {grand_kept} "
          f"({grand_total - grand_kept} dropped, {(grand_total - grand_kept) / grand_total:.1%})")


if __name__ == "__main__":
    main()
