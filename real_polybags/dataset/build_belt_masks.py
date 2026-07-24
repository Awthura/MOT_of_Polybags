"""
Build a per-camera "active region" mask to suppress detections on static
background fixtures (conveyor rollers, cables, monitors, keyboards, etc.)
that both LocateAnything and YOLO have been found to misclassify as
polybags -- see AUTOLABEL_REPORT.md section 5.

Since each camera is fixed for its whole capture session, the belt surface is
the only thing with real temporal variance frame-to-frame; static fixtures
are ~constant. Sample N frames, compute per-pixel max-min range, threshold,
and keep only the single largest connected component (the belt) -- clutter
consistently forms its own smaller, disjoint component.

One failure mode was found and hand-verified during testing, NOT handled
generically: an automatic "detection recurs often => must be real" rescue
rule was tried and rejected, because LA's own false-positive rate on
background clutter turned out to be highly camera-dependent -- as high as
66% of frames on basler_2's rollers, which a generic recurrence threshold
would have wrongly kept. Only rgbd_1_color has a genuine exception, a
slow-moving bag "buffer/queue zone" (bags sit ~still for long stretches, so
it reads as low-variance and gets wrongly excluded even though real
detections land there in ~92% of frames) -- confirmed for real by manually
inspecting the source frames, not inferred from statistics alone. That one
region is hardcoded below as a per-camera override rather than trusting a
blanket automatic rule.

Usage:
    python build_belt_masks.py            # build masks + QA overlays for all cameras
    python build_belt_masks.py --preview   # also dump the raw variation heatmap for inspection
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent
UNLABELLED_IMAGES = DATASET_ROOT / "unlabelled"
OUT_DIR = DATASET_ROOT / "belt_masks"

N_SAMPLE_FRAMES = 150
CLOSE_KERNEL = 25       # fill small holes/gaps within the belt region
OPEN_KERNEL = 9         # remove speckle noise from the variation threshold
DILATE_MARGIN = 20      # grow the mask outward so bag edges near the boundary aren't clipped

# Manually hand-verified extra regions to force-include, per camera, as
# (x1, y1, x2, y2) pixel rectangles at the camera's native resolution.
# Each entry was confirmed by viewing source frames, not inferred automatically.
MANUAL_INCLUDE_REGIONS = {
    "rgbd_1_color_1280x720_20260528_092854": [
        (875, 578, 1143, 720),  # bag buffer/queue zone, bottom-right (~92% of frames hit here)
    ],
}


def sample_frames(cam_dir, n):
    images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if len(images) <= n:
        return images
    idx = np.linspace(0, len(images) - 1, n).astype(int)
    return [images[i] for i in idx]


def sample_frames(cam_dir, n):
    images = sorted([p for p in cam_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if len(images) <= n:
        return images
    idx = np.linspace(0, len(images) - 1, n).astype(int)
    return [images[i] for i in idx]


def build_mask_for_camera(cam_dir, preview=False):
    frames = sample_frames(cam_dir, N_SAMPLE_FRAMES)
    print(f"[{cam_dir.name}] sampling {len(frames)} frames ...")

    stack = None
    for i, p in enumerate(frames):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if stack is None:
            h, w = img.shape
            stack = np.empty((len(frames), h, w), dtype=np.uint8)
        stack[i] = img

    frame_min = stack.min(axis=0)
    frame_max = stack.max(axis=0)
    variation = (frame_max.astype(np.int16) - frame_min.astype(np.int16)).astype(np.uint8)

    # Otsu threshold on the variation map: high-variation = belt/active region
    _, active = cv2.threshold(variation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((CLOSE_KERNEL, CLOSE_KERNEL), np.uint8))

    # Background clutter (rollers, cables, monitors) shows up as its own separate
    # connected component from the belt -- e.g. on rgbd_1_color the belt is one
    # ~274k-px blob while the left-side rollers and right-side monitor/cable
    # clutter each form their own disjoint ~25-100k-px blobs. Keep only the
    # single largest component (the belt) and drop everything else.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    if n_labels <= 1:
        active = np.zeros_like(active)
    else:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        active = np.where(labels == largest, np.uint8(255), np.uint8(0))

    active = cv2.dilate(active, np.ones((DILATE_MARGIN, DILATE_MARGIN), np.uint8))

    # Hand-verified per-camera exceptions -- see MANUAL_INCLUDE_REGIONS docstring note.
    for (x1, y1, x2, y2) in MANUAL_INCLUDE_REGIONS.get(cam_dir.name, []):
        active[y1:y2, x1:x2] = 255

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{cam_dir.name}.png"), active)
    if preview:
        cv2.imwrite(str(out_dir / f"{cam_dir.name}_variation.png"), variation)

    # QA overlay: mask boundary drawn in cyan over a sample frame
    sample_img = cv2.imread(str(frames[len(frames) // 2]))
    contours, _ = cv2.findContours(active, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(sample_img, contours, -1, (255, 255, 0), 3)
    cv2.imwrite(str(out_dir / f"{cam_dir.name}_overlay.jpg"), sample_img)

    coverage = active.mean() / 255
    print(f"  mask covers {coverage:.1%} of frame -> {out_dir / (cam_dir.name + '.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="also save the raw variation heatmap")
    args = ap.parse_args()

    cam_dirs = sorted(d for d in UNLABELLED_IMAGES.iterdir() if d.is_dir())
    for cam_dir in cam_dirs:
        build_mask_for_camera(cam_dir, preview=args.preview)


if __name__ == "__main__":
    main()
