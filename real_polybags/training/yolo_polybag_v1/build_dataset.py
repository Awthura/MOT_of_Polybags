#!/usr/bin/env python3
"""
Build a YOLO detect dataset from the tiled autolabels — polybag_v1.

Source images and their labels live in two separate trees:
  images: real_polybags/dataset/unlabelled/<cam_session>/<stem>.jpg
  labels: real_polybags/dataset/unlabelled_autolabels_v4_tiled/labels_final_v3/<cam_session>/<stem>.txt

Ultralytics finds a label by swapping `/images/` -> `/labels/` in the image
path, so this assembles the expected sibling layout under ./dataset/:
  dataset/images/{train,val}/<stem>.jpg   (symlinks — no image is copied)
  dataset/labels/{train,val}/<stem>.txt   (copies)

Split is a **contiguous per-camera tail** (last VAL_FRACTION of each camera's
frames go to val), not a random split: consecutive video frames are near
duplicates, so a random split would leak almost-identical frames into val and
flatter the metrics. One class, `polybag` (the autolabels are single-class,
axis-aligned).

Run this **on the machine that will train** (paths are baked into the symlinks
and the generated data.yaml). Also writes data.yaml with an absolute path, so
Ultralytics resolves the dataset regardless of its datasets_dir setting.

    python3 build_dataset.py            # all five cameras
    python3 build_dataset.py --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # real_polybags/
IMG_ROOT = REPO / "dataset" / "unlabelled"
LBL_ROOT = REPO / "dataset" / "unlabelled_autolabels_v4_tiled" / "labels_final_v3"

CAM_SESSIONS = [
    "basler_1_1280x720_20260528_092854",
    "basler_2_1280x720_20260528_092854",
    "lucid_1280x720_20260528_092854",
    "rgbd_1_color_1280x720_20260528_092854",
    "rgbd_2_color_1280x720_20260528_092854",
]


def build(out: Path, val_fraction: float) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "val": 0, "no_label": 0}
    per_cam = {}
    for cs in CAM_SESSIONS:
        img_dir, lbl_dir = IMG_ROOT / cs, LBL_ROOT / cs
        if not img_dir.exists():
            print(f"[skip] no images for {cs}")
            continue
        # Pair each frame with its label; keep only frames that have a label
        # file (an empty label = a valid negative / no bags in view).
        frames = sorted(img_dir.glob("*.jpg"))
        paired = [(im, lbl_dir / f"{im.stem}.txt") for im in frames
                  if (lbl_dir / f"{im.stem}.txt").exists()]
        totals["no_label"] += len(frames) - len(paired)
        n_val = int(round(len(paired) * val_fraction))
        split_at = len(paired) - n_val            # tail -> val
        tr = vl = 0
        for i, (im, lbl) in enumerate(paired):
            split = "train" if i < split_at else "val"
            (out / "images" / split / im.name).symlink_to(im.resolve())
            shutil.copy(lbl, out / "labels" / split / lbl.name)
            totals[split] += 1
            tr += split == "train"
            vl += split == "val"
        per_cam[cs] = (tr, vl)
        print(f"  {cs:38s} train {tr:4d}  val {vl:4d}")

    (out / "data.yaml").write_text(
        f"# polybag_v1 — YOLO detect, single class, tiled autolabels (labels_final_v3)\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['polybag']\n")
    print(f"\nTOTAL  train {totals['train']}  val {totals['val']}  "
          f"(skipped {totals['no_label']} frames with no label file)")
    print(f"wrote {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dataset"))
    ap.add_argument("--val-fraction", type=float, default=0.15)
    a = ap.parse_args()
    build(Path(a.out), a.val_fraction)


if __name__ == "__main__":
    main()
