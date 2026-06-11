"""
Polybag Detection Pipeline
- EDA with figures
- Annotation overlays on labelled images
- Auto-labelling experiment (color-based OBB)
- Subset copy for unlabelled images
- JSON (LabelMe) to YOLO OBB .txt conversion
"""

import json
import os
import shutil
import random
import math
from pathlib import Path
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path("/Users/awthura/OVGU/AMS")
TRAIN_DIR     = BASE / "YOLO11_OBB_training_data"
ANNOT_DIR     = BASE / "YOLO11_OBB_annotations"

OUT_ROOT      = BASE / "pipeline_output"
OUT_EDA       = OUT_ROOT / "eda"
OUT_OVERLAY   = OUT_ROOT / "overlays"
OUT_SUBSET    = OUT_ROOT / "unlabelled_subset"
OUT_AUTOLABEL = OUT_ROOT / "autolabel_experiment"
OUT_YOLO      = OUT_ROOT / "yolo_labels"

for d in [OUT_EDA, OUT_OVERLAY, OUT_SUBSET, OUT_AUTOLABEL, OUT_YOLO]:
    d.mkdir(parents=True, exist_ok=True)

# ── Class definitions ─────────────────────────────────────────────────────────
CLASSES = ["pink_polybag", "blue_polybag", "yellow_polybag",
           "grey_polybag", "green_polybag", "red_polybag"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# BGR palette for drawing
CLASS_COLOR_BGR = {
    "pink_polybag":   (180,  80, 200),
    "blue_polybag":   (200,  80,   0),
    "yellow_polybag": (  0, 200, 220),
    "grey_polybag":   (160, 160, 160),
    "green_polybag":  (  0, 180,  60),
    "red_polybag":    ( 40,  40, 220),
}

# HSV ranges for auto-labelling (lower, upper) — tuned for 3-D renders
HSV_RANGES = {
    "pink_polybag":   [([140,  40,  80], [175, 255, 255]),
                       ([  0,  30,  80], [ 10, 255, 255])],
    "blue_polybag":   [([100,  60,  60], [130, 255, 255])],
    "yellow_polybag": [([18,   80,  80], [ 38, 255, 255])],
    "grey_polybag":   [([  0,   0, 100], [179,  50, 220])],
    "green_polybag":  [([38,   60,  60], [ 85, 255, 255])],
    "red_polybag":    [([  0,  80,  80], [ 10, 255, 255]),
                       ([170,  80,  80], [179, 255, 255])],
}

AUTOLABEL_SUBSET = 50   # images to experiment with
UNLABELLED_COPY  = 100  # images to copy for the unlabelled subset folder

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    shapes = data.get("shapes", [])
    w = data.get("imageWidth", 1920)
    h = data.get("imageHeight", 1080)
    return shapes, w, h


def polygon_to_obb_yolo(points, img_w, img_h):
    """Convert 4-point polygon to YOLO OBB line (8 normalised coords)."""
    flat = []
    for x, y in points:
        flat.extend([x / img_w, y / img_h])
    return flat


def minrect_to_4pts(rect):
    """Convert cv2.minAreaRect to 4 corner points (float32)."""
    return cv2.boxPoints(rect).astype(float)


def mask_to_obbs(mask, label, img_w, img_h, min_area=400):
    """Return list of OBB dicts detected in a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obbs = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect  = cv2.minAreaRect(cnt)
        pts   = minrect_to_4pts(rect)
        yolo  = polygon_to_obb_yolo(pts, img_w, img_h)
        obbs.append({"label": label, "points": pts.tolist(), "yolo": yolo,
                     "area": cv2.contourArea(cnt)})
    return obbs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Inventory
# ─────────────────────────────────────────────────────────────────────────────

all_images  = sorted(TRAIN_DIR.glob("*.png"))
json_files  = {p.stem: p for p in TRAIN_DIR.glob("*.json")}
labelled    = [p for p in all_images if p.stem in json_files]
unlabelled  = [p for p in all_images if p.stem not in json_files]

print(f"Total images : {len(all_images)}")
print(f"Labelled     : {len(labelled)}")
print(f"Unlabelled   : {len(unlabelled)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EDA] Parsing annotations …")

class_counts      = Counter()
annots_per_image  = []
bbox_widths, bbox_heights, bbox_areas, bbox_angles = [], [], [], []

per_image_stats = []

for img_path in labelled:
    shapes, img_w, img_h = load_json_annotations(json_files[img_path.stem])
    n = len(shapes)
    annots_per_image.append(n)
    img_classes = Counter()
    for s in shapes:
        lbl = s["label"]
        class_counts[lbl] += 1
        img_classes[lbl] += 1
        pts = np.array(s["points"], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_areas.append(w * h)
        bbox_angles.append(angle % 90)
    per_image_stats.append({"file": img_path.name, "n_annots": n, **img_classes})

# ── Figure 1: dataset split pie ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([len(labelled), len(unlabelled)],
       labels=["Labelled", "Unlabelled"],
       autopct="%1.1f%%",
       colors=["#4CAF50", "#FF7043"],
       startangle=90, textprops={"fontsize": 13})
ax.set_title("Dataset Split", fontsize=15)
fig.tight_layout()
fig.savefig(OUT_EDA / "01_dataset_split.png", dpi=150)
plt.close(fig)

# ── Figure 2: class distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
classes_sorted = sorted(class_counts.keys(), key=lambda c: -class_counts[c])
counts_sorted  = [class_counts[c] for c in classes_sorted]
palette = [f"#{abs(hash(c)) % 0xFFFFFF:06X}" for c in classes_sorted]
bars = ax.bar(classes_sorted, counts_sorted, color=palette, edgecolor="black", linewidth=0.6)
ax.bar_label(bars, padding=3, fontsize=11)
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Annotation Count per Class", fontsize=14)
ax.set_xticklabels(classes_sorted, rotation=20, ha="right")
fig.tight_layout()
fig.savefig(OUT_EDA / "02_class_distribution.png", dpi=150)
plt.close(fig)

# ── Figure 3: annotations per image histogram ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(annots_per_image, bins=20, color="#5C6BC0", edgecolor="black", linewidth=0.6)
ax.set_xlabel("Annotations per image", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Annotations per Labelled Image", fontsize=14)
ax.axvline(np.mean(annots_per_image), color="red", linestyle="--",
           label=f"Mean = {np.mean(annots_per_image):.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_EDA / "03_annots_per_image.png", dpi=150)
plt.close(fig)

# ── Figure 4: bounding-box size scatter ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(bbox_widths, bbox_heights, alpha=0.35, s=10, c=bbox_areas,
                cmap="viridis")
fig.colorbar(sc, ax=ax, label="Area (px²)")
ax.set_xlabel("OBB width (px)", fontsize=12)
ax.set_ylabel("OBB height (px)", fontsize=12)
ax.set_title("OBB Width vs Height", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EDA / "04_bbox_size_scatter.png", dpi=150)
plt.close(fig)

# ── Figure 5: area distribution ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(bbox_areas, bins=40, color="#26A69A", edgecolor="black", linewidth=0.5)
ax.set_xlabel("OBB area (px²)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("OBB Area Distribution", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EDA / "05_bbox_area_distribution.png", dpi=150)
plt.close(fig)

# ── Figure 6: rotation angle distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(bbox_angles, bins=45, color="#EF5350", edgecolor="black", linewidth=0.5)
ax.set_xlabel("OBB angle mod 90° (degrees)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("OBB Rotation Angle Distribution", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EDA / "06_bbox_angle_distribution.png", dpi=150)
plt.close(fig)

# ── Figure 7: class co-occurrence heatmap ────────────────────────────────────
co_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
for img_path in labelled:
    shapes, _, _ = load_json_annotations(json_files[img_path.stem])
    img_cls = list({s["label"] for s in shapes})
    for i, a in enumerate(CLASSES):
        for j, b in enumerate(CLASSES):
            if a in img_cls and b in img_cls:
                co_matrix[i, j] += 1

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(co_matrix, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            linewidths=0.5)
ax.set_title("Class Co-occurrence per Image", fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(OUT_EDA / "07_class_cooccurrence.png", dpi=150)
plt.close(fig)

# ── Figure 8: spatial heatmap of annotation centres ──────────────────────────
img_w_ref, img_h_ref = 1920, 1080
cx_all, cy_all = [], []
for img_path in labelled:
    shapes, iw, ih = load_json_annotations(json_files[img_path.stem])
    for s in shapes:
        pts = np.array(s["points"])
        cx_all.append(pts[:, 0].mean() / iw)
        cy_all.append(pts[:, 1].mean() / ih)

fig, ax = plt.subplots(figsize=(10, 5))
h2d = ax.hist2d(cx_all, cy_all, bins=(40, 22), cmap="hot")
fig.colorbar(h2d[3], ax=ax, label="Count")
ax.set_xlabel("Normalised X", fontsize=12)
ax.set_ylabel("Normalised Y", fontsize=12)
ax.set_title("Spatial Heatmap of Annotation Centres", fontsize=14)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(OUT_EDA / "08_spatial_heatmap.png", dpi=150)
plt.close(fig)

# ── Figure 9: sample labelled images mosaic ──────────────────────────────────
sample_imgs = random.sample(labelled, min(9, len(labelled)))
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for ax, img_path in zip(axes.flat, sample_imgs):
    shapes, iw, ih = load_json_annotations(json_files[img_path.stem])
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for s in shapes:
        pts = np.array(s["points"], dtype=np.int32)
        color_bgr = CLASS_COLOR_BGR.get(s["label"], (128, 128, 128))
        color_rgb = color_bgr[::-1]
        poly = MplPolygon(pts, closed=True, fill=False,
                          edgecolor=[c / 255 for c in color_rgb], linewidth=1.5)
        ax.add_patch(poly)
    ax.imshow(img)
    ax.set_title(img_path.stem, fontsize=7)
    ax.axis("off")
for ax in axes.flat[len(sample_imgs):]:
    ax.axis("off")
fig.suptitle("Sample Labelled Images", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EDA / "09_sample_labelled_mosaic.png", dpi=120)
plt.close(fig)

print(f"[EDA] Figures saved to {OUT_EDA}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Annotation overlays
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Overlay] Drawing annotations on labelled images …")

for img_path in labelled:
    shapes, iw, ih = load_json_annotations(json_files[img_path.stem])
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    for s in shapes:
        pts  = np.array(s["points"], dtype=np.int32)
        lbl  = s["label"]
        color = CLASS_COLOR_BGR.get(lbl, (128, 128, 128))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # label text near first point
        tx, ty = int(pts[0][0]), max(int(pts[0][1]) - 5, 12)
        cv2.putText(img, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    out_path = OUT_OVERLAY / img_path.name
    cv2.imwrite(str(out_path), img)

print(f"[Overlay] {len(labelled)} images saved to {OUT_OVERLAY}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. JSON → YOLO OBB conversion
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Convert] JSON → YOLO OBB …")

converted = 0
for img_path in labelled:
    shapes, iw, ih = load_json_annotations(json_files[img_path.stem])
    lines = []
    for s in shapes:
        lbl = s["label"]
        if lbl not in CLASS_ID:
            continue
        pts   = s["points"]           # already 4 points from LabelMe polygon
        yolo  = polygon_to_obb_yolo(pts, iw, ih)
        line  = f"{CLASS_ID[lbl]} " + " ".join(f"{v:.6f}" for v in yolo)
        lines.append(line)

    if lines:
        out_txt = OUT_YOLO / (img_path.stem + ".txt")
        out_txt.write_text("\n".join(lines))
        converted += 1

# write classes.txt
(OUT_YOLO / "classes.txt").write_text("\n".join(CLASSES))
print(f"[Convert] {converted} label files saved to {OUT_YOLO}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Copy unlabelled subset
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Subset] Copying {UNLABELLED_COPY} unlabelled images …")

random.seed(42)
subset = random.sample(unlabelled, min(UNLABELLED_COPY, len(unlabelled)))
for p in subset:
    shutil.copy2(p, OUT_SUBSET / p.name)

print(f"[Subset] {len(subset)} images copied to {OUT_SUBSET}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Auto-labelling experiment (colour-based OBB)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[AutoLabel] Running colour-based OBB on {AUTOLABEL_SUBSET} images …")

autolabel_imgs = random.sample(unlabelled, min(AUTOLABEL_SUBSET, len(unlabelled)))
total_detections = 0
det_per_image = []

for img_path in autolabel_imgs:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    vis   = img.copy()
    yolo_lines = []
    n_det = 0

    for cls_name, ranges in HSV_RANGES.items():
        # combine all hue ranges for this class
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            combined_mask |= cv2.inRange(hsv, lo_arr, hi_arr)

        obbs = mask_to_obbs(combined_mask, cls_name, img_w, img_h, min_area=300)
        color = CLASS_COLOR_BGR.get(cls_name, (128, 128, 128))

        for obb in obbs:
            pts_draw = np.array(obb["points"], dtype=np.int32)
            cv2.drawContours(vis, [pts_draw], 0, color, 2)
            cx = int(pts_draw[:, 0].mean())
            cy = int(pts_draw[:, 1].mean())
            cv2.putText(vis, cls_name[:4], (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            yolo_lines.append(
                f"{CLASS_ID[cls_name]} " +
                " ".join(f"{v:.6f}" for v in obb["yolo"])
            )
            n_det += 1

    # save visualisation
    cv2.imwrite(str(OUT_AUTOLABEL / img_path.name), vis)

    # save YOLO label
    if yolo_lines:
        lbl_path = OUT_AUTOLABEL / (img_path.stem + ".txt")
        lbl_path.write_text("\n".join(yolo_lines))

    det_per_image.append(n_det)
    total_detections += n_det

print(f"[AutoLabel] {total_detections} total detections over {len(autolabel_imgs)} images")
print(f"            avg {np.mean(det_per_image):.1f} / image")

# ── Figure 10: autolabel detection count distribution ────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(det_per_image, bins=20, color="#AB47BC", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Detections per image", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Auto-label: Detections per Image (colour-based OBB)", fontsize=13)
ax.axvline(np.mean(det_per_image), color="red", linestyle="--",
           label=f"Mean = {np.mean(det_per_image):.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_EDA / "10_autolabel_detections.png", dpi=150)
plt.close(fig)

# ── Figure 11: sample auto-label results mosaic ───────────────────────────────
sample_al = [autolabel_imgs[i] for i in range(min(9, len(autolabel_imgs)))]
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
for ax, img_path in zip(axes.flat, sample_al):
    result = cv2.imread(str(OUT_AUTOLABEL / img_path.name))
    if result is not None:
        ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title(img_path.stem, fontsize=7)
    ax.axis("off")
for ax in axes.flat[len(sample_al):]:
    ax.axis("off")
fig.suptitle("Auto-label Results (colour-based OBB)", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_EDA / "11_autolabel_sample_mosaic.png", dpi=120)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  EDA figures      : {OUT_EDA}")
print(f"  Overlays         : {OUT_OVERLAY}  ({len(labelled)} images)")
print(f"  YOLO labels      : {OUT_YOLO}    ({converted} files)")
print(f"  Unlabelled subset: {OUT_SUBSET}   ({len(subset)} images)")
print(f"  Auto-label output: {OUT_AUTOLABEL} ({len(autolabel_imgs)} images + labels)")
print("=" * 60)
