"""
Full auto-labelling pipeline — v4 (watershed segmentation)

Pipeline per image:
  1. Threshold V > 120  →  bright foreground mask  (background V ≈ 59)
  2. MORPH_OPEN (3×3, 1 iter) to remove sub-pixel noise without merging
  3. Distance transform → local-maxima seeding → Watershed
     • Seeds found as local maxima of the distance map (min 12-px separation)
     • This correctly splits touching/adjacent polybags into separate instances
  4. For each watershed region, sample median interior HSV
  5. Nearest-centroid classifier (trained on GT, 3× weight on H channel)
  6. approxPolyDP → tight 4-corner OBB; minAreaRect fallback

Output layout:
  full_dataset/
  ├── images/          all 1157 images (symlinked)
  ├── labels_manual/   86  YOLO OBB .txt  (JSON → YOLO from hand annotations)
  ├── labels_auto/     1071 YOLO OBB .txt (watershed auto-label)
  ├── labels/          merged  (manual takes precedence)
  └── dataset.yaml
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path("/Users/awthura/OVGU/AMS/real_polybags")
TRAIN_DIR  = BASE / "YOLO11_OBB_training_data"
OUT_ROOT   = BASE / "full_dataset"

OUT_IMAGES        = OUT_ROOT / "images"
OUT_LABELS_MANUAL = OUT_ROOT / "labels_manual"
OUT_LABELS_AUTO   = OUT_ROOT / "labels_auto"
OUT_LABELS        = OUT_ROOT / "labels"

for d in [OUT_IMAGES, OUT_LABELS_MANUAL, OUT_LABELS_AUTO, OUT_LABELS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES  = ["pink_polybag", "blue_polybag", "yellow_polybag",
            "grey_polybag",  "green_polybag", "red_polybag"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# V threshold separating dark background (V≈59) from bright polybags (V≥170)
BG_V_THRESH = 120
# Minimum watershed region area (px²) — below this it's noise
MIN_AREA    = 150
# Local-maxima minimum separation (px) — tuned to ~half bag diameter
SEED_DIST   = 12

K3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K25 = np.ones((SEED_DIST * 2 + 1, SEED_DIST * 2 + 1), np.uint8)


# ── Nested-box removal ───────────────────────────────────────────────────────

def _pt_in_poly(px, py, poly):
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

def _poly_area(pts):
    n = len(pts)
    return abs(sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))) / 2

# Fraction of the smaller box that must be covered by the larger box to trigger removal
OVERLAP_THRESH = 0.5

def remove_nested(detections, img_w, img_h):
    """Remove the smaller box when two same-class boxes overlap by >= OVERLAP_THRESH."""
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
            pts_i = np.array([(coords_i[k]*img_w, coords_i[k+1]*img_h)
                               for k in range(0, 8, 2)], dtype=np.float32)
            pts_j = np.array([(coords_j[k]*img_w, coords_j[k+1]*img_h)
                               for k in range(0, 8, 2)], dtype=np.float32)
            area_i = _poly_area(pts_i.tolist())
            area_j = _poly_area(pts_j.tolist())
            if area_i >= area_j:          # j is not larger — skip
                continue
            # compute intersection area of the two convex polygons
            ret, inter = cv2.intersectConvexConvex(pts_i, pts_j)
            if ret == 0 or inter is None or len(inter) < 3:
                continue
            inter_area = cv2.contourArea(inter.reshape(-1, 1, 2).astype(np.float32))
            if area_i > 0 and inter_area / area_i >= OVERLAP_THRESH:
                to_remove.add(i)          # i is the smaller, heavily overlapped box
    return [d for k, d in enumerate(detections) if k not in to_remove]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)],
                     pts[np.argmin(diff)],
                     pts[np.argmax(s)],
                     pts[np.argmax(diff)]])


def contour_to_quad(cnt) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    for eps in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype(np.float64))
    return order_points(cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64))


def polygon_to_yolo(pts, img_w, img_h) -> list:
    return [v for x, y in pts for v in (x / img_w, y / img_h)]


# ── Colour classifier ─────────────────────────────────────────────────────────

def nearest_class(median_hsv: np.ndarray, centroids: dict) -> str:
    mh, ms, mv = median_hsv.astype(float)
    if ms < 25:
        return "grey_polybag"
    best_cls, best_d = None, float("inf")
    for cls, (rh, rs, rv) in centroids.items():
        if cls == "grey_polybag":
            continue
        dh   = min(abs(mh - rh), 180 - abs(mh - rh))
        dist = (3 * dh) ** 2 + (ms - rs) ** 2 + (mv - rv) ** 2
        if dist < best_d:
            best_d, best_cls = dist, cls
    return best_cls


# ── GT centroid learning ──────────────────────────────────────────────────────

def compute_gt_centroids(labelled, json_map) -> dict:
    accum = defaultdict(list)
    for img_path in labelled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        with open(json_map[img_path.stem]) as f:
            data = json.load(f)
        iw  = data.get("imageWidth",  img.shape[1])
        ih  = data.get("imageHeight", img.shape[0])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for s in data.get("shapes", []):
            lbl = s["label"]
            if lbl not in CLASS_ID:
                continue
            pts  = np.array(s["points"], dtype=np.int32)
            mask = np.zeros((ih, iw), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            mask = cv2.erode(mask, K3, iterations=2)
            vals = hsv[mask == 255]
            if len(vals) >= 5:
                accum[lbl].append(np.median(vals, axis=0))

    centroids = {}
    print("\n  GT HSV centroids (H, S, V):")
    for cls in CLASSES:
        if cls in accum:
            arr = np.array(accum[cls])
            c   = arr.mean(axis=0)
            centroids[cls] = tuple(c)
            print(f"    {cls:<22s}: H={c[0]:.1f}  S={c[1]:.1f}  V={c[2]:.1f}  (n={len(arr)})")
    return centroids


# ── JSON → YOLO ───────────────────────────────────────────────────────────────

def json_to_yolo(json_path: Path) -> list:
    with open(json_path) as f:
        data = json.load(f)
    iw = data.get("imageWidth", 1920)
    ih = data.get("imageHeight", 1080)
    lines = []
    for s in data.get("shapes", []):
        lbl = s["label"]
        if lbl not in CLASS_ID:
            continue
        yolo = polygon_to_yolo(s["points"], iw, ih)
        lines.append(f"{CLASS_ID[lbl]} " + " ".join(f"{v:.6f}" for v in yolo))
    return lines


def write_label(path: Path, lines: list):
    path.write_text("\n".join(lines) if lines else "")


# ── Auto-label worker ─────────────────────────────────────────────────────────

def autolabel_image(args):
    img_path, centroids = args
    img = cv2.imread(str(img_path))
    if img is None:
        return img_path.stem, []

    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Bright foreground mask
    fg = cv2.inRange(hsv,
                     np.array([0, 0, BG_V_THRESH], np.uint8),
                     np.array([179, 255, 255],      np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, K3, iterations=1)

    if cv2.countNonZero(fg) == 0:
        return img_path.stem, []

    # 2. Distance transform for seed extraction
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    # Blur suppresses shadow-induced sub-peaks inside a single bag
    dist = cv2.GaussianBlur(dist, (0, 0), 1)

    # 3. Local maxima (one seed per polybag, min SEED_DIST px apart)
    dilated = cv2.dilate(dist, K25)
    peaks   = np.uint8((dist >= dilated * 0.999) & (dist > 2.0)) * 255

    if cv2.countNonZero(peaks) == 0:
        return img_path.stem, []

    # 4. Watershed
    sure_bg  = cv2.dilate(fg, K3, iterations=2)
    unknown  = cv2.subtract(sure_bg, peaks)
    _, marks = cv2.connectedComponents(peaks)
    marks    = marks + 1
    marks[unknown == 255] = 0
    cv2.watershed(img.copy(), marks)

    # 5. Extract regions → OBB + colour class
    results = []
    for lid in np.unique(marks):
        if lid <= 1 or lid == -1:
            continue
        region = np.uint8(marks == lid) * 255
        if cv2.countNonZero(region) < MIN_AREA:
            continue

        # Sample interior colour
        interior = hsv[region == 255]
        if len(interior) == 0:
            continue
        cls = nearest_class(np.median(interior, axis=0), centroids)

        # Fit OBB
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt  = max(cnts, key=cv2.contourArea)
        quad = contour_to_quad(cnt)
        yolo = polygon_to_yolo(quad, w, h)
        results.append((CLASS_ID[cls], yolo))

    results = remove_nested(results, w, h)
    return img_path.stem, results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    all_images = sorted(TRAIN_DIR.glob("*.png"))
    json_map   = {p.stem: p for p in TRAIN_DIR.glob("*.json")}
    labelled   = [p for p in all_images if p.stem in json_map]
    unlabelled = [p for p in all_images if p.stem not in json_map]

    print(f"Total images : {len(all_images)}")
    print(f"  labelled   : {len(labelled)}")
    print(f"  unlabelled : {len(unlabelled)}")

    # 0. GT centroids
    print("\n[0/3] Computing GT colour centroids …")
    centroids = compute_gt_centroids(labelled, json_map)

    # 1. Manual JSON → YOLO
    print("\n[1/3] Converting manual JSON annotations …")
    for img_path in tqdm(labelled, unit="img"):
        lines = json_to_yolo(json_map[img_path.stem])
        write_label(OUT_LABELS_MANUAL / (img_path.stem + ".txt"), lines)

    # 2. Auto-label (parallel)
    print(f"\n[2/3] Auto-labelling {len(unlabelled)} images …")
    stats       = Counter()
    empty_count = 0

    with Pool(max(1, cpu_count() - 1)) as pool:
        for stem, detections in tqdm(
                pool.imap_unordered(autolabel_image, [(p, centroids) for p in unlabelled]),
                total=len(unlabelled), unit="img"):

            lines = [
                f"{cid} " + " ".join(f"{v:.6f}" for v in coords)
                for cid, coords in detections
            ]
            write_label(OUT_LABELS_AUTO / (stem + ".txt"), lines)
            if not detections:
                empty_count += 1
            for cid, _ in detections:
                stats[CLASSES[cid]] += 1

    total = sum(stats.values())
    print(f"\n  Total detections : {total}  ({total/max(len(unlabelled),1):.1f} avg/image)")
    print(f"  Empty label files: {empty_count}")
    for cls in CLASSES:
        print(f"    {cls:<22s}: {stats[cls]}")

    # 3. Merge + symlink
    print("\n[3/3] Building merged dataset …")
    for img_path in tqdm(all_images, unit="img"):
        link = OUT_IMAGES / img_path.name
        if not link.exists():
            link.symlink_to(img_path.resolve())

        stem = img_path.stem
        dst  = OUT_LABELS / (stem + ".txt")
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        src = (OUT_LABELS_MANUAL if stem in json_map else OUT_LABELS_AUTO) / (stem + ".txt")
        dst.symlink_to(src.resolve())

    (OUT_ROOT / "classes.txt").write_text("\n".join(CLASSES))
    (OUT_ROOT / "dataset.yaml").write_text(
        f"path: {OUT_ROOT}\ntrain: images\nval: images\n\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )

    print(f"""
{'='*60}
DONE
  {OUT_ROOT}/
  ├── images/          {len(all_images)} symlinks
  ├── labels_manual/   {len(labelled)} files  (hand-annotated)
  ├── labels_auto/     {len(unlabelled)} files  (watershed v4)
  ├── labels/          {len(all_images)} symlinks (merged)
  ├── classes.txt
  └── dataset.yaml
{'='*60}
""")
