"""
Apply the per-camera belt masks (built by build_belt_masks.py) to filter out
detections centered on known static background clutter (rollers, cables,
monitors, etc.) -- see AUTOLABEL_REPORT.md section 5/6.

Filters both:
  - LocateAnything cleaned autolabels (dataset/unlabelled_autolabels/labels_cleaned/)
  - The cross-validation YOLO labels (dataset/unlabelled_autolabels/cross_validation/yolo_labels/)

A box is dropped if its center falls outside the camera's belt mask.
Writes masked copies alongside the originals and prints before/after counts.

Usage:
    python apply_belt_masks.py
"""

from pathlib import Path

import cv2

DATASET_ROOT = Path(__file__).resolve().parent
BELT_MASKS = DATASET_ROOT / "belt_masks"
LA_LABELS_CLEANED = DATASET_ROOT / "unlabelled_autolabels" / "labels_cleaned"
YOLO_LABELS = DATASET_ROOT / "unlabelled_autolabels" / "cross_validation" / "yolo_labels"

LA_LABELS_MASKED = DATASET_ROOT / "unlabelled_autolabels" / "labels_cleaned_masked"
YOLO_LABELS_MASKED = DATASET_ROOT / "unlabelled_autolabels" / "cross_validation" / "yolo_labels_masked"


def filter_dir(label_dir, mask, out_dir):
    h, w = mask.shape
    total, kept = 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in label_dir.glob("*.txt"):
        lines_out = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            total += 1
            cls, cx, cy, bw, bh = parts
            px, py = min(int(float(cx) * w), w - 1), min(int(float(cy) * h), h - 1)
            if mask[py, px] != 0:
                kept += 1
                lines_out.append(line)
        (out_dir / f.name).write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return total, kept


def main():
    cam_names = sorted(p.stem for p in BELT_MASKS.glob("*.png"))
    grand_la_total = grand_la_kept = grand_yolo_total = grand_yolo_kept = 0
    for cam in cam_names:
        mask = cv2.imread(str(BELT_MASKS / f"{cam}.png"), cv2.IMREAD_GRAYSCALE)

        la_dir = LA_LABELS_CLEANED / cam
        if la_dir.exists():
            la_total, la_kept = filter_dir(la_dir, mask, LA_LABELS_MASKED / cam)
            grand_la_total += la_total
            grand_la_kept += la_kept
            print(f"[{cam}] LA:   {la_total} -> {la_kept} "
                  f"({la_total - la_kept} dropped, {(la_total - la_kept) / la_total:.1%})")

        yolo_dir = YOLO_LABELS / cam
        if yolo_dir.exists():
            yolo_total, yolo_kept = filter_dir(yolo_dir, mask, YOLO_LABELS_MASKED / cam)
            grand_yolo_total += yolo_total
            grand_yolo_kept += yolo_kept
            print(f"[{cam}] YOLO: {yolo_total} -> {yolo_kept} "
                  f"({yolo_total - yolo_kept} dropped, {(yolo_total - yolo_kept) / yolo_total:.1%})")

    print("\n=== Totals ===")
    print(f"LA:   {grand_la_total} -> {grand_la_kept} "
          f"({grand_la_total - grand_la_kept} dropped, "
          f"{(grand_la_total - grand_la_kept) / grand_la_total:.1%})")
    print(f"YOLO: {grand_yolo_total} -> {grand_yolo_kept} "
          f"({grand_yolo_total - grand_yolo_kept} dropped, "
          f"{(grand_yolo_total - grand_yolo_kept) / grand_yolo_total:.1%})")


if __name__ == "__main__":
    main()
