#!/usr/bin/env python3
"""
Fix YOLO OBB class labels in synth_dataset by sampling the actual rendered
pixel colour at each OBB centroid.

Previous class assignment: mat_idx % 6 (STL separation order — arbitrary, wrong).
New class assignment: sample ~13×13 patch at OBB centroid → median HSV →
  classify using the same 6-class / HSV scheme from polybag_pipeline.py
  (which was tuned for 3-D renders), with slightly relaxed saturation thresholds
  to handle the softer palette of the Blender renders.

Classes (unchanged, 6):
  0  pink_polybag
  1  blue_polybag
  2  yellow_polybag
  3  grey_polybag
  4  green_polybag
  5  red_polybag
"""

import cv2
import numpy as np
from pathlib import Path

LABELS_DIR  = Path("/Users/awthura/OVGU/AMS/synth_dataset/labels")
IMAGES_DIR  = Path("/Users/awthura/OVGU/AMS/synth_dataset/images")
CLASSES_TXT = Path("/Users/awthura/OVGU/AMS/synth_dataset/classes.txt")

CLASS_NAMES = [
    "pink_polybag",    # 0
    "blue_polybag",    # 1
    "yellow_polybag",  # 2
    "grey_polybag",    # 3
    "green_polybag",   # 4
    "red_polybag",     # 5
    "teal_polybag",    # 6
]

PATCH_HALF = 6   # 13×13 pixel patch at OBB centroid

# OpenCV HSV: H [0-180], S [0-255], V [0-255]
# Adapted from polybag_pipeline.py HSV_RANGES, with relaxed saturation
# to handle the softer Blender render palette.
#
# Priority order (first match wins):
#   grey   → very low saturation
#   red    → H near 0 / 180 (wraps around)
#   yellow → warm orange/yellow
#   green  → lime, yellow-green, teal-green, cyan
#   blue   → periwinkle, mid-blue, cool purple
#   pink   → magenta, hot-pink, warm purple
THRESHOLDS = [
    # (class_id, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)  — H is OpenCV [0-180]
    (3,   0, 180,   0,  30,  80, 255),   # grey: any H, very low S
    (5,   0,  12,  40, 255,  80, 255),   # red low side
    (5, 165, 180,  40, 255,  80, 255),   # red high side (wraps)
    (2,  13,  40,  30, 255,  80, 255),   # yellow / orange
    (4,  41,  72,  25, 255,  50, 255),   # green / lime / yellow-green  (H 41-72)
    (6,  73, 100,  25, 255,  50, 255),   # teal / cyan-green            (H 73-100)
    (1, 101, 138,  25, 255,  50, 255),   # blue / cool purple
    (0, 139, 167,  25, 255,  50, 255),   # pink / warm purple / magenta
]
# Format: (class_id, h_min, h_max, s_min, s_max, v_min, v_max)

FALLBACK_CLASS = 3   # grey if nothing matched


def classify_hsv(h: float, s: float, v: float) -> int:
    """Map OpenCV HSV (all [0-255] scale with H [0-180]) → class id."""
    for cid, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi in THRESHOLDS:
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return cid
    return FALLBACK_CLASS


def sample_class(img: np.ndarray, coords_norm: list, w: int, h: int) -> int:
    """Sample 13×13 patch at OBB centroid; return colour class."""
    xs = [coords_norm[i] * w for i in range(0, 8, 2)]
    ys = [coords_norm[i] * h for i in range(1, 8, 2)]
    cx = int(round(sum(xs) / 4))
    cy = int(round(sum(ys) / 4))

    x1 = max(0, cx - PATCH_HALF);  x2 = min(w, cx + PATCH_HALF + 1)
    y1 = max(0, cy - PATCH_HALF);  y2 = min(h, cy + PATCH_HALF + 1)

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return FALLBACK_CLASS

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    med = np.median(hsv.reshape(-1, 3), axis=0)
    return classify_hsv(float(med[0]), float(med[1]), float(med[2]))


def relabel_file(lf: Path, img: np.ndarray, w: int, h: int) -> tuple[int, int]:
    lines = lf.read_text().strip().splitlines()
    if not lines:
        return 0, 0
    new_lines, changed = [], 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            new_lines.append(line); continue
        old_cid   = int(parts[0])
        coords    = list(map(float, parts[1:]))
        new_cid   = sample_class(img, coords, w, h)
        if new_cid != old_cid:
            changed += 1
        new_lines.append(f"{new_cid} " + " ".join(f"{v:.6f}" for v in coords))
    lf.write_text("\n".join(new_lines))
    return len(new_lines), changed


def main():
    label_files = sorted(LABELS_DIR.glob("*.txt"))
    print(f"Relabelling {len(label_files)} label files …")
    CLASSES_TXT.write_text("\n".join(CLASS_NAMES))

    total_anns = total_changed = missing = 0
    class_hist = [0] * len(CLASS_NAMES)

    for lf in label_files:
        img_path = IMAGES_DIR / lf.name.replace(".txt", ".png")
        if not img_path.exists():
            missing += 1; continue
        img = cv2.imread(str(img_path))
        if img is None:
            missing += 1; continue
        h, w = img.shape[:2]
        n, c = relabel_file(lf, img, w, h)
        total_anns += n; total_changed += c
        # tally new class distribution
        for line in lf.read_text().splitlines():
            p = line.split()
            if p: class_hist[int(p[0])] += 1

    print(f"\n{'='*50}")
    print(f"  Annotations processed  : {total_anns}")
    print(f"  Class assignments fixed: {total_changed} "
          f"({100*total_changed/max(1,total_anns):.1f}%)")
    print(f"  Images skipped         : {missing}")
    print(f"\n  New class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * (class_hist[i] // 50)
        print(f"    {i} {name:18s} {class_hist[i]:6d}  {bar}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
