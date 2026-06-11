"""
Generate a reference image: one row per track ID, showing
 - track ID number
 - a representative image crop around the bag centroid
 - the colour patch + HSV values
 - current class assignment

No Blender required — reads MOT gt_obb.txt + rendered images directly.
"""
import cv2, numpy as np, csv
from pathlib import Path
from collections import defaultdict

BASE       = Path("/Users/awthura/OVGU/AMS")
IMAGES_DIR = BASE / "synth_dataset" / "images"
MOT_DIR    = BASE / "synth_dataset" / "mot_obb"
OUT_CSV    = BASE / "synth_dataset" / "track_classes.csv"
OUT_IMG    = BASE / "track_color_reference.png"

CLASS_NAMES = ["pink_polybag","blue_polybag","yellow_polybag",
               "grey_polybag","green_polybag","red_polybag"]
CLASS_COLORS_BGR = [   # display color for label text
    (180,  60, 255),   # 0 pink
    (200,  80,   0),   # 1 blue
    (  0, 200, 220),   # 2 yellow
    (160, 160, 160),   # 3 grey
    ( 30, 180,  30),   # 4 green
    (  0,  30, 200),   # 5 red
]

CAMS = [("cam_01_front","front"), ("cam_02_back","back"),
        ("cam_03_left","left"),   ("cam_04_right","right")]

CROP_HALF = 60    # px around centroid in image space
N_SAMPLES  = 5   # crop samples per track

# ── Load known classes from CSV ───────────────────────────────────────────────
known = {}
if OUT_CSV.exists():
    with open(OUT_CSV) as f:
        for row in csv.DictReader(f):
            known[int(row["track_id"])] = int(row["class_id"])

# ── Read all MOT files: track_id → [(frame, cam_short, cx_img, cy_img)] ──────
track_appearances = defaultdict(list)   # tid → list of (frame, cam_short, cx, cy, rw, rh)

for cam_sub, cam_short in CAMS:
    mot_file = MOT_DIR / cam_sub / "gt" / "gt_obb.txt"
    if not mot_file.exists(): continue
    with open(mot_file) as f:
        for line in f:
            if line.startswith("#"): continue
            cols = line.strip().split(",")
            if len(cols) < 10: continue
            frame_idx = int(cols[0])   # 1-based
            tid       = int(cols[1])
            corners   = list(map(float, cols[2:10]))
            cx = np.mean(corners[0::2])
            cy = np.mean(corners[1::2])
            real_frame = 100 + frame_idx - 1
            track_appearances[tid].append((real_frame, cam_short, cx, cy))

all_tids = sorted(track_appearances.keys())
print(f"Found {len(all_tids)} tracks: {all_tids}")

# ── For each track, pick N_SAMPLES representative frames and crop ─────────────
CELL_W, CELL_H = 960, 220   # width per track row
PANEL_W = 220                # width of crop panel (N_SAMPLES crops side by side)
CROP_DISP = 100              # display size of each crop

def sample_crop(frame, cam_short, cx_img, cy_img):
    """Load image, crop around centroid, return resized patch or None."""
    img_path = IMAGES_DIR / f"{cam_short}_frame_{frame:04d}.png"
    if not img_path.exists(): return None
    img = cv2.imread(str(img_path))
    if img is None: return None
    h, w = img.shape[:2]
    cx, cy = int(cx_img), int(cy_img)
    x1=max(0,cx-CROP_HALF); x2=min(w,cx+CROP_HALF+1)
    y1=max(0,cy-CROP_HALF); y2=min(h,cy+CROP_HALF+1)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0: return None
    return cv2.resize(patch, (CROP_DISP, CROP_DISP))

rows = []
for tid in all_tids:
    apps = track_appearances[tid]
    # evenly sample N_SAMPLES frames from the track's lifetime
    step = max(1, len(apps) // N_SAMPLES)
    chosen = apps[::step][:N_SAMPLES]

    # ── Collect HSV from all appearances (no isolation filter) ───────────────
    all_hsv = []
    for frame, cam_short, cx_img, cy_img in apps:
        img_path = IMAGES_DIR / f"{cam_short}_frame_{frame:04d}.png"
        if not img_path.exists(): continue
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        cx, cy = int(cx_img), int(cy_img)
        x1=max(0,cx-6); x2=min(w,cx+7)
        y1=max(0,cy-6); y2=min(h,cy+7)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0: continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        med = np.median(hsv.reshape(-1,3), axis=0)
        if med[2] > 40:
            all_hsv.append(med)

    # ── Collect sample crops ──────────────────────────────────────────────────
    crops = []
    for frame, cam_short, cx_img, cy_img in chosen:
        c = sample_crop(frame, cam_short, int(cx_img), int(cy_img))
        if c is not None:
            crops.append(c)

    rows.append((tid, crops, all_hsv))

# ── Layout: one row per track ─────────────────────────────────────────────────
ROW_H = CROP_DISP + 20
LABEL_W = 200
SWATCH_W = 120
CROP_AREA_W = N_SAMPLES * (CROP_DISP + 4)
TOTAL_W = LABEL_W + SWATCH_W + CROP_AREA_W + 20
TOTAL_H = len(rows) * (ROW_H + 8) + 60

canvas = np.ones((TOTAL_H, TOTAL_W, 3), dtype=np.uint8) * 30  # dark bg

# header
cv2.putText(canvas, "Track ID  |  Median colour swatch  |  Image crops from rendered frames",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)

for idx, (tid, crops, all_hsv) in enumerate(rows):
    y0 = 50 + idx * (ROW_H + 8)

    cid = known.get(tid, -1)
    class_label = CLASS_NAMES[cid] if cid >= 0 else "UNKNOWN"
    label_color = CLASS_COLORS_BGR[cid] if cid >= 0 else (100,100,100)

    # ── Track ID label ────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, y0), (LABEL_W-4, y0+ROW_H), (50,50,50), -1)
    cv2.putText(canvas, f"Track {tid:2d}", (8, y0+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(canvas, f"cls {cid}: {class_label}", (8, y0+75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
    cv2.putText(canvas, f"{len(all_hsv)} hsv samples", (8, y0+95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

    # ── Colour swatch (median BGR) ────────────────────────────────────────────
    x0_swatch = LABEL_W
    if all_hsv:
        arr = np.array(all_hsv)
        med_hsv = np.median(arr, axis=0).astype(np.uint8)
        # also show 5 percentile bands sorted by hue for more info
        swatch_hsv = np.full((ROW_H, SWATCH_W, 3), med_hsv, dtype=np.uint8)
        swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
        canvas[y0:y0+ROW_H, x0_swatch:x0_swatch+SWATCH_W] = swatch_bgr
        h, s, v = float(med_hsv[0]), float(med_hsv[1]), float(med_hsv[2])
        # text on swatch
        text_col = (0,0,0) if v > 120 else (255,255,255)
        cv2.putText(canvas, f"H={h:.0f}", (x0_swatch+4, y0+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1)
        cv2.putText(canvas, f"S={s:.0f}", (x0_swatch+4, y0+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1)
        cv2.putText(canvas, f"V={v:.0f}", (x0_swatch+4, y0+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 1)
    else:
        cv2.rectangle(canvas, (x0_swatch, y0), (x0_swatch+SWATCH_W, y0+ROW_H), (60,60,60), -1)
        cv2.putText(canvas, "no data", (x0_swatch+4, y0+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

    # ── Crop thumbnails ───────────────────────────────────────────────────────
    x0_crops = LABEL_W + SWATCH_W + 4
    for ci, crop in enumerate(crops[:N_SAMPLES]):
        xc = x0_crops + ci * (CROP_DISP + 4)
        canvas[y0:y0+CROP_DISP, xc:xc+CROP_DISP] = crop

    # row separator
    cv2.line(canvas, (0, y0+ROW_H+3), (TOTAL_W, y0+ROW_H+3), (80,80,80), 1)

cv2.imwrite(str(OUT_IMG), canvas)
print(f"Saved: {OUT_IMG}  ({TOTAL_W}x{TOTAL_H})")
