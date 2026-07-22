"""
Auto-labelling pipeline for real-camera polybag images.

Background: bright red fabric
Classes:
  0 = white_polybag  (white/translucent bubble-wrap bags)
  1 = yellow_polybag (golden padded envelope)

Outputs YOLO OBB .txt labels and overlay PNGs.
"""

import cv2
import numpy as np
import os
import sys
import glob
from pathlib import Path

# ── Hyperparameters ────────────────────────────────────────────────────────────
# Red background detection
RED_H_LO1, RED_H_HI1 = 0, 10       # lower red hue range (OpenCV 0-180)
RED_H_LO2, RED_H_HI2 = 160, 180    # upper red hue range
RED_S_MIN = 90                       # saturation minimum for red
RED_V_MIN = 50                       # value minimum (exclude black)

# Dark pixel exclusion (table edges, equipment)
DARK_V_MAX = 40

# Yellow polybag (padded envelope)
YEL_H_LO, YEL_H_HI = 10, 45
YEL_S_MIN = 60
YEL_V_MIN = 80

# White bag brightness floor (room background tops out at V≈137; bags start at V≈140)
WHITE_V_MIN = 140

# Minimum region area (px²) to keep
MIN_AREA_WHITE = 1500
MIN_AREA_YELLOW = 500

# Watershed seed separation distance (px)
WATERSHED_DIST_THRESH = 0.3   # fraction of max distance for seeds
MIN_SEED_DIST_PX = 15

# OBB epsilon range for approxPolyDP
EPS_START = 0.02
EPS_END   = 0.15

# Nested duplicate overlap threshold
OVERLAP_THRESH = 0.50

# Class names and BGR colours for overlay
CLASS_NAMES = ["white_polybag", "yellow_polybag"]
OVERLAY_COLOURS = {
    0: (255, 255, 255),   # white
    1: (0, 200, 255),     # gold/orange
}


# ── Helper functions ──────────────────────────────────────────────────────────

def build_fg_mask(hsv):
    """Return foreground binary mask (not-red, not-dark)."""
    # red background
    red1 = cv2.inRange(hsv, (RED_H_LO1, RED_S_MIN, RED_V_MIN),
                            (RED_H_HI1, 255, 255))
    red2 = cv2.inRange(hsv, (RED_H_LO2, RED_S_MIN, RED_V_MIN),
                            (RED_H_HI2, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    # dark pixels
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_V_MAX))

    fg = cv2.bitwise_not(cv2.bitwise_or(red, dark))

    # morphological cleaning: close small gaps, open to remove specks
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k7, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3, iterations=1)

    # zero out image border (camera edge artifacts)
    border = 25
    fg[:border, :] = 0
    fg[-border:, :] = 0
    fg[:, :border] = 0
    fg[:, -border:] = 0
    return fg


def split_yellow(hsv, fg_mask):
    """Split foreground into yellow-bag and white-bag masks."""
    yellow = cv2.inRange(hsv, (YEL_H_LO, YEL_S_MIN, YEL_V_MIN),
                              (YEL_H_HI, 255, 255))
    yellow = cv2.bitwise_and(yellow, fg_mask)

    # dilate yellow slightly to capture edges
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k5, iterations=1)

    # white = foreground minus yellow
    white = cv2.bitwise_and(fg_mask, cv2.bitwise_not(yellow))

    # brightness floor for white bags
    bright = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, 255, 255))
    white = cv2.bitwise_and(white, bright)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k5, iterations=1)

    return white, yellow


def watershed_segment(binary_mask, min_area):
    """
    Use watershed to separate touching objects in binary_mask.
    Returns list of (mask_for_region) for each detected region.
    """
    if binary_mask.sum() == 0:
        return []

    # distance transform
    dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    # smooth to suppress local noise in the map
    dist_blur = cv2.GaussianBlur(dist, (5, 5), 1)

    # find local maxima as seeds
    from scipy.ndimage import maximum_filter
    # fallback if scipy unavailable: just threshold
    try:
        local_max = (dist_blur == maximum_filter(dist_blur, size=MIN_SEED_DIST_PX*2+1))
        local_max = local_max & (dist_blur > WATERSHED_DIST_THRESH * dist_blur.max())
        local_max = local_max.astype(np.uint8) * 255
    except ImportError:
        thresh_val = WATERSHED_DIST_THRESH * dist_blur.max()
        _, local_max = cv2.threshold(dist_blur, thresh_val, 255, cv2.THRESH_BINARY)
        local_max = local_max.astype(np.uint8)

    # connected components of seeds → marker image
    num_seeds, markers = cv2.connectedComponents(local_max)

    if num_seeds <= 1:
        # no seeds found; treat whole mask as one region
        num_seeds, markers = cv2.connectedComponents(binary_mask)

    # unknown region (foreground but not seed)
    sure_bg = cv2.dilate(binary_mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                         iterations=2)
    unknown = cv2.subtract(sure_bg, local_max)
    markers[unknown > 0] = 0

    # build 3-channel image for watershed
    # use grayscale of original (or just use binary_mask as luminance)
    rgb_input = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    markers = markers.astype(np.int32)
    cv2.watershed(rgb_input, markers)

    regions = []
    for label in range(1, num_seeds):
        region_mask = (markers == label).astype(np.uint8) * 255
        region_mask = cv2.bitwise_and(region_mask, binary_mask)
        area = cv2.countNonZero(region_mask)
        if area >= min_area:
            regions.append(region_mask)
    return regions


def fit_obb(region_mask, img_w, img_h):
    """
    Fit oriented bounding box to contour of region.
    Returns normalised [x1,y1,x2,y2,x3,y3,x4,y4] or None.
    """
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None

    arc = cv2.arcLength(cnt, True)
    pts4 = None
    for eps_f in np.linspace(EPS_START, EPS_END, 8):
        approx = cv2.approxPolyDP(cnt, eps_f * arc, True)
        if len(approx) == 4:
            pts4 = approx.reshape(4, 2)
            break

    if pts4 is None:
        rect = cv2.minAreaRect(cnt)
        pts4 = cv2.boxPoints(rect).astype(int)

    coords = []
    for (px, py) in pts4:
        coords.extend([px / img_w, py / img_h])
    return coords


def poly_area(pts):
    n = len(pts)
    return abs(sum(pts[i][0] * pts[(i+1) % n][1] -
                   pts[(i+1) % n][0] * pts[i][1]
                   for i in range(n))) / 2


def remove_nested(detections, img_w, img_h):
    """Remove smaller boxes that overlap ≥ OVERLAP_THRESH with a larger same-class box."""
    to_remove = set()
    n = len(detections)
    for i in range(n):
        for j in range(n):
            if i == j or i in to_remove:
                continue
            cid_i, coords_i = detections[i]
            cid_j, coords_j = detections[j]
            if cid_i != cid_j:
                continue
            pts_i = np.array([(coords_i[k] * img_w, coords_i[k + 1] * img_h)
                               for k in range(0, 8, 2)], dtype=np.float32)
            pts_j = np.array([(coords_j[k] * img_w, coords_j[k + 1] * img_h)
                               for k in range(0, 8, 2)], dtype=np.float32)
            area_i = poly_area(pts_i.tolist())
            area_j = poly_area(pts_j.tolist())
            if area_i >= area_j:
                continue
            ret, inter = cv2.intersectConvexConvex(pts_i, pts_j)
            if ret == 0 or inter is None or len(inter) < 3:
                continue
            inter_area = cv2.contourArea(inter.reshape(-1, 1, 2).astype(np.float32))
            if area_i > 0 and inter_area / area_i >= OVERLAP_THRESH:
                to_remove.add(i)
    return [d for k, d in enumerate(detections) if k not in to_remove]


def draw_overlay(img, detections, img_w, img_h):
    """Draw OBB detections onto a copy of img."""
    out = img.copy()
    for (cid, coords) in detections:
        pts = np.array([(int(coords[k] * img_w), int(coords[k + 1] * img_h))
                        for k in range(0, 8, 2)], dtype=np.int32)
        colour = OVERLAY_COLOURS.get(cid, (200, 200, 200))
        cv2.polylines(out, [pts], True, colour, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(out, CLASS_NAMES[cid], (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def autolabel_image(img_path, label_path, overlay_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  SKIP (unreadable): {img_path}")
        return 0

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    fg = build_fg_mask(hsv)
    white_mask, yellow_mask = split_yellow(hsv, fg)

    detections = []

    # ── White bags ────────────────────────────────────────────────
    white_regions = watershed_segment(white_mask, MIN_AREA_WHITE)
    for region in white_regions:
        coords = fit_obb(region, w, h)
        if coords:
            detections.append((0, coords))

    # ── Yellow bag ────────────────────────────────────────────────
    yellow_regions = watershed_segment(yellow_mask, MIN_AREA_YELLOW)
    for region in yellow_regions:
        coords = fit_obb(region, w, h)
        if coords:
            detections.append((1, coords))

    detections = remove_nested(detections, w, h)

    # Write label file
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        for (cid, coords) in detections:
            f.write(f"{cid} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

    # Write overlay
    if overlay_path:
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        overlay = draw_overlay(img, detections, w, h)
        cv2.imwrite(str(overlay_path), overlay)

    return len(detections)


def run(img_paths, out_dir):
    total = 0
    for img_path in img_paths:
        img_path = Path(img_path)
        subdir   = img_path.parent.name          # e.g. "0003"
        stem     = img_path.stem                 # e.g. "rgb_frame_001512"

        label_path   = Path(out_dir) / "labels"   / subdir / f"{stem}.txt"
        overlay_path = Path(out_dir) / "overlays"  / subdir / f"{stem}.png"

        n = autolabel_image(img_path, label_path, overlay_path)
        total += n
        print(f"  {subdir}/{stem}: {n} detections")

    print(f"\nDone. {len(img_paths)} frames, {total} total detections "
          f"({total/max(1,len(img_paths)):.1f} avg/frame)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/Users/awthura/OVGU/AMS/real_polybags/real_data",
                        help="Root dir or single subdir containing PNG frames")
    parser.add_argument("--subdirs", nargs="+", default=["0000"],
                        help="Subdirectories to process (default: 0000 for test)")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Max frames per subdir (0 = all)")
    parser.add_argument("--out_dir",
                        default="/Users/awthura/OVGU/AMS/real_polybags/real_data_labels",
                        help="Output directory for labels + overlays")
    args = parser.parse_args()

    img_paths = []
    for sd in args.subdirs:
        pattern = os.path.join(args.input_dir, sd, "rgb_frame_*.png")
        found = sorted(glob.glob(pattern))
        if args.max_frames > 0:
            found = found[:args.max_frames]
        img_paths.extend(found)

    print(f"Processing {len(img_paths)} frames → {args.out_dir}")
    run(img_paths, args.out_dir)
