"""
Generate overlay images for the entire dataset (manual + auto labels).
Reads YOLO OBB .txt from full_dataset/labels/, draws OBBs on each image,
saves to full_dataset/overlays/.
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path("/Users/awthura/OVGU/AMS/real_polybags")
TRAIN_DIR  = BASE / "YOLO11_OBB_training_data"
LABELS_DIR = BASE / "full_dataset" / "labels"
OUT_DIR    = BASE / "full_dataset" / "overlays"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["pink_polybag", "blue_polybag", "yellow_polybag",
           "grey_polybag",  "green_polybag", "red_polybag"]

# BGR colours, one per class
PALETTE_BGR = [
    (180,  80, 200),   # pink
    (200,  80,   0),   # blue
    (  0, 200, 220),   # yellow
    (160, 160, 160),   # grey
    (  0, 180,  60),   # green
    ( 40,  40, 220),   # red
]

# ── Worker ────────────────────────────────────────────────────────────────────

def process(img_path: Path):
    lbl_path = LABELS_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return img_path.name, 0

    img = cv2.imread(str(img_path))
    if img is None:
        return img_path.name, 0

    h, w = img.shape[:2]
    n = 0

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cid  = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts  = np.array([(coords[i] * w, coords[i+1] * h)
                         for i in range(0, 8, 2)], dtype=np.int32)
        color = PALETTE_BGR[cid % len(PALETTE_BGR)]

        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # class label near the top-left corner of the box
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        short  = CLASSES[cid].replace("_polybag", "")
        cv2.putText(img, short, (int(cx) - 18, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        n += 1

    cv2.imwrite(str(OUT_DIR / img_path.name), img)
    return img_path.name, n


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_images = sorted(TRAIN_DIR.glob("*.png"))
    print(f"Rendering overlays for {len(all_images)} images  "
          f"→  {OUT_DIR}")

    workers = max(1, cpu_count() - 1)
    total_boxes = 0
    skipped     = 0

    with Pool(workers) as pool:
        for _, n in tqdm(pool.imap_unordered(process, all_images),
                         total=len(all_images), unit="img"):
            if n == 0:
                skipped += 1
            total_boxes += n

    saved = len(all_images) - skipped
    print(f"\nDone — {saved} overlays written, {total_boxes} boxes drawn "
          f"({skipped} images had no labels).")
