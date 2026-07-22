"""
Generate overlay images for the annotated real dataset (train_v11_obb_final,
val_v11_obb_final). Reads YOLO OBB .txt labels, draws OBBs on each image,
saves to dataset/annotated/<split>/overlays/.
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/awthura/OVGU/AMS/real_polybags/dataset/annotated")
SPLITS   = ["train_v11_obb_final", "val_v11_obb_final"]

# Class semantics not documented by the supervisor yet — label by raw id
# until confirmed, then update CLASSES/PALETTE_BGR accordingly.
CLASSES = ["class_0", "class_1"]

# BGR colours, one per class
PALETTE_BGR = [
    ( 40,  40, 220),   # class_0 — red
    (200,  80,   0),   # class_1 — blue
]

# ── Worker ────────────────────────────────────────────────────────────────────

def process(args):
    img_path, labels_dir, out_dir = args
    lbl_path = labels_dir / (img_path.stem + ".txt")
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

        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        label  = CLASSES[cid] if cid < len(CLASSES) else str(cid)
        cv2.putText(img, label, (int(cx) - 18, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        n += 1

    cv2.imwrite(str(out_dir / img_path.name), img)
    return img_path.name, n


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    workers = max(1, cpu_count() - 1)

    for split in SPLITS:
        split_dir  = BASE / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        out_dir    = split_dir / "overlays"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_images = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        print(f"[{split}] Rendering overlays for {len(all_images)} images  →  {out_dir}")

        tasks = [(p, labels_dir, out_dir) for p in all_images]
        total_boxes = 0
        skipped     = 0

        with Pool(workers) as pool:
            for _, n in tqdm(pool.imap_unordered(process, tasks),
                             total=len(tasks), unit="img"):
                if n == 0:
                    skipped += 1
                total_boxes += n

        saved = len(all_images) - skipped
        print(f"[{split}] Done — {saved} overlays written, {total_boxes} boxes drawn "
              f"({skipped} images had no labels).\n")
