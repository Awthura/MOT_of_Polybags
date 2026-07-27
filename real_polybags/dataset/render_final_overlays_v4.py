"""
Render overlay images for a final recommended label set (cleaned + belt-
masked, +whole-belt-fix where applicable), across the full dataset, for
visual review. Local-only, no cluster/model needed -- just draws boxes from
existing label files onto existing images. Defaults to the v4 (real_v4_round2)
labels_final/; pass --labels-root/--out-root to point at another run's
output (e.g. unlabelled_autolabels_v4_tiled/labels_final/).

Usage:
    python render_final_overlays_v4.py [--camera CAMERA_NAME] [--limit N]
    python render_final_overlays_v4.py --labels-root unlabelled_autolabels_v4_tiled/labels_final \\
        --out-root unlabelled_autolabels_v4_tiled/overlays_final
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
DEFAULT_LABELS_ROOT = DATASET_ROOT / "unlabelled_autolabels_v4" / "labels_final"
DEFAULT_OUT_ROOT = DATASET_ROOT / "unlabelled_autolabels_v4" / "overlays_final"


def parse_label(path):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            boxes.append((cx, cy, bw, bh))
    return boxes


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None, help="only render this camera folder")
    ap.add_argument("--limit", type=int, default=None, help="cap number of images per camera")
    ap.add_argument("--labels-root", default=None,
                     help="path (relative to dataset/) to a labels_final-style dir; "
                          f"defaults to {DEFAULT_LABELS_ROOT.relative_to(DATASET_ROOT)}")
    ap.add_argument("--out-root", default=None,
                     help="output dir for overlays; defaults alongside --labels-root's run")
    args = ap.parse_args()

    LABELS_ROOT = DATASET_ROOT / args.labels_root if args.labels_root else DEFAULT_LABELS_ROOT
    OUT_ROOT = DATASET_ROOT / args.out_root if args.out_root else DEFAULT_OUT_ROOT

    cam_dirs = sorted(d for d in LABELS_ROOT.iterdir() if d.is_dir())
    if args.camera:
        cam_dirs = [d for d in cam_dirs if d.name == args.camera]

    total = 0
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        out_dir = OUT_ROOT / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)
        label_files = sorted(cam_dir.glob("*.txt"))
        if args.limit:
            label_files = label_files[:args.limit]

        n_boxes_cam = 0
        for label_path in label_files:
            stem = label_path.stem
            img_path = find_image(cam_name, stem)
            if img_path is None:
                continue
            boxes = parse_label(label_path)
            img = Image.open(img_path).convert("RGB")
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            draw_boxes(cv_img, boxes)
            cv2.imwrite(str(out_dir / (stem + ".jpg")), cv_img)
            total += 1
            n_boxes_cam += len(boxes)

        print(f"[{cam_name}] {len(label_files)} images -> {out_dir}")

    print(f"\nTotal overlays rendered: {total}")
    print(f"-> {OUT_ROOT}")


if __name__ == "__main__":
    main()
