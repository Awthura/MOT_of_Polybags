#!/usr/bin/env python3
"""
postprocess_new_frames.py
Run OUTSIDE Blender, after render_missing_frames.sh completes.

1. Copies new images from synth_dataset/cam_XX/images/ → synth_dataset/images/
2. Copies new labels from synth_dataset/cam_XX/labels/ → synth_dataset/labels/
3. Runs relabel_synth.py (colour-correct all labels via HSV sampling)
4. Generates class-coloured OBB overlays for every new image
"""

import shutil, subprocess, sys, cv2
import numpy as np
from pathlib import Path

BASE     = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
SD       = BASE / "synth_dataset"
IMG_OUT  = SD / "images"
LBL_OUT  = SD / "labels"
OVL_OUT  = SD / "class_overlays"

CAM_MAP = {
    "cam_01_front": "front",
    "cam_02_back":  "back",
    "cam_03_left":  "left",
    "cam_04_right": "right",
}

# 7-class colour palette (BGR) — matches relabel_synth.py class order
CLASS_COLORS = [
    (180,  80, 255),   # 0 pink
    (255, 150,  80),   # 1 blue (periwinkle)
    (  0, 220, 255),   # 2 yellow
    (180, 180, 180),   # 3 grey
    ( 60, 220,  60),   # 4 green
    ( 40,  40, 255),   # 5 red
    (200, 180,  50),   # 6 teal
]
CLASS_NAMES = ["pink","blue","yellow","grey","green","red","teal"]


def draw_obb_overlay(img_path: Path, lbl_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [coords[i] * w for i in range(0, 8, 2)]
        ys = [coords[i] * h for i in range(1, 8, 2)]
        pts = np.array(list(zip(xs, ys)), dtype=np.int32)
        color = CLASS_COLORS[cid % len(CLASS_COLORS)]
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, 2)
        label = CLASS_NAMES[cid % len(CLASS_NAMES)]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        cv2.putText(img, label, (cx - 20, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    OVL_OUT.mkdir(parents=True, exist_ok=True)

    # ── Step 1: copy new images and labels ────────────────────────────────────
    print("Copying new images and labels from per-camera folders...")
    new_images = []
    new_labels = []

    for cam_sub, cam_short in CAM_MAP.items():
        src_img_dir = SD / cam_sub / "images"
        src_lbl_dir = SD / cam_sub / "labels"
        if not src_img_dir.exists():
            continue
        for img_src in sorted(src_img_dir.glob("frame_*.png")):
            frame_num = int(img_src.stem.split("_")[1])
            dst_img = IMG_OUT / f"{cam_short}_frame_{frame_num:04d}.png"
            dst_lbl = LBL_OUT / f"{cam_short}_frame_{frame_num:04d}.txt"
            lbl_src = src_lbl_dir / f"frame_{frame_num:04d}.txt"

            if not dst_img.exists():
                shutil.copy2(img_src, dst_img)
                new_images.append(dst_img)
            if lbl_src.exists() and not dst_lbl.exists():
                shutil.copy2(lbl_src, dst_lbl)
                new_labels.append(dst_lbl)

    print(f"  {len(new_images)} new images, {len(new_labels)} new labels")

    if not new_images:
        print("  Nothing new — all frames already in dataset.")
        return

    # ── Step 2: colour-correct ALL labels via HSV sampling ────────────────────
    print("Running relabel_synth.py (full dataset colour correction)...")
    result = subprocess.run(
        [sys.executable, str(BASE / "relabel_synth.py")],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.strip():
            print(f"  {line}")
    if result.returncode != 0:
        print("  ERROR in relabel_synth.py:")
        print(result.stderr[-800:])

    # ── Step 3: generate overlays for new images ───────────────────────────────
    print(f"Generating overlays for {len(new_images)} new images...")
    skipped = 0
    for dst_img in sorted(new_images):
        stem = dst_img.stem                    # e.g. front_frame_0268
        dst_lbl = LBL_OUT / f"{stem}.txt"
        out_ovl = OVL_OUT / f"{stem}_overlay.jpg"
        if not dst_lbl.exists():
            skipped += 1
            continue
        draw_obb_overlay(dst_img, dst_lbl, out_ovl)

    print(f"  Overlays written: {len(new_images) - skipped}  skipped: {skipped}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_img = len(list(IMG_OUT.glob("*.png")))
    total_lbl = len(list(LBL_OUT.glob("*.txt")))
    total_ovl = len(list(OVL_OUT.glob("*.jpg")))
    print(f"\nDataset totals:")
    print(f"  images/   : {total_img}")
    print(f"  labels/   : {total_lbl}")
    print(f"  overlays/ : {total_ovl}")
    print()
    print("Next step — re-run MOT annotation for all frames:")
    print()
    print("  /Applications/Blender.app/Contents/MacOS/Blender \\")
    print(f"      {BASE}/convert_stl_to_animation_multi_camera.blend \\")
    print(f"      --background --python {BASE}/blender_mot_annotate.py \\")
    print(f"      -- --frames 100-1873")
    print()
    print("  Then:  python3 fix_mot_classes.py")


if __name__ == "__main__":
    main()
