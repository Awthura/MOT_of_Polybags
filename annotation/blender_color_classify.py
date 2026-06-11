"""
blender_color_classify.py
────────────────────────────────────────────────────────────────────────────────
Use Blender geometry + cameras to identify the true colour of each polybag
particle, then write a corrected class label for every YOLO OBB annotation.

How it works
────────────
1.  For every frame: import the per-frame STL, separate loose parts, and
    compute each part's 3-D world centroid exactly as the render script did.
2.  Run Hungarian matching on 3-D centroids → globally consistent track IDs
    (same algorithm as blender_mot_annotate.py).
3.  For every camera in every frame:
      • Project the 3-D centroid through Blender's camera matrix → pixel (cx, cy)
      • Skip if the particle is too close to another particle in 3-D
        (3-D isolation score < MIN_ISOLATION_3D) — overlapping in image space
        is guaranteed for nearby 3-D neighbours.
      • Sample a 13×13 patch in the ALREADY-RENDERED PNG at (cx, cy).
      • Record (track_id, H, S, V) for this observation.
4.  For each track_id compute the MEDIAN HSV across all clean observations.
5.  Map median HSV → one of 6 colour classes (same as polybag_pipeline.py).
6.  Write   synth_dataset/track_classes.csv      (track_id → class_id)
    Update  synth_dataset/labels/*.txt           (fix class in every OBB line)
    Update  synth_dataset/mot_obb/*/gt/gt_obb.txt (fix class_id column)

Run:
    /Applications/Blender.app/Contents/MacOS/Blender \
        /Users/awthura/OVGU/AMS/convert_stl_to_animation_multi_camera.blend \
        --background --python blender_color_classify.py \
        -- [--frames 100-1972]
"""

import sys, argparse, csv
from pathlib import Path
from collections import defaultdict

import bpy
import bpy_extras.object_utils
from mathutils import Vector
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path("/Users/awthura/OVGU/AMS")
STL_FOLDER  = BASE / "superquadrics_stl_files_100_2000_frames"
IMAGES_DIR  = BASE / "synth_dataset" / "images"
LABELS_DIR  = BASE / "synth_dataset" / "labels"
MOT_DIR     = BASE / "synth_dataset" / "mot_obb"
OUT_CSV     = BASE / "synth_dataset" / "track_classes.csv"

CAMERA_CONFIGS = [
    ("Cam_Front",  "cam_01_front",  "front"),
    ("Cam_Back",   "cam_02_back",   "back"),
    ("Cam_Left",   "cam_03_left",   "left"),
    ("Cam_Right",  "cam_04_right",  "right"),
]

CLASS_NAMES = [
    "pink_polybag",    # 0
    "blue_polybag",    # 1
    "yellow_polybag",  # 2
    "grey_polybag",    # 3
    "green_polybag",   # 4
    "red_polybag",     # 5
]

# 3-D isolation threshold: skip sampling if nearest neighbour < this (Blender units)
MIN_ISOLATION_3D = 0.12

# Colour classification thresholds (OpenCV HSV: H [0-180], S/V [0-255])
# Priority order, first match wins.
#   (class_id, H_min, H_max, S_min, S_max, V_min, V_max)
COLOUR_RULES = [
    (3,   0, 180,   0,  30,  80, 255),   # grey / white  – very low saturation
    (5,   0,  12,  40, 255,  80, 255),   # red  (low side)
    (5, 165, 180,  40, 255,  80, 255),   # red  (high side, wraps)
    (2,  13,  40,  30, 255,  80, 255),   # yellow / orange
    (4,  41, 100,  25, 255,  50, 255),   # green / lime / teal
    (1, 101, 138,  25, 255,  50, 255),   # blue / cool purple
    (0, 139, 167,  25, 255,  50, 255),   # pink / warm purple / magenta
]
FALLBACK_CLASS = 3

# Tracking
MAX_MATCH_DIST = 0.30   # Blender units
PATCH_HALF     = 6      # 13×13 px patch


# ── Colour helper ─────────────────────────────────────────────────────────────
def hsv_to_class(h, s, v):
    for cid, h0, h1, s0, s1, v0, v1 in COLOUR_RULES:
        if h0 <= h <= h1 and s0 <= s <= s1 and v0 <= v <= v1:
            return cid
    return FALLBACK_CLASS


# ── STL helpers (mirrors blender_mot_annotate.py) ────────────────────────────
_mat_cache = {}

def _get_mat(index):
    name = f"mat_cc_{index:03d}"
    if name in _mat_cache:
        return _mat_cache[name]
    COLS = [(0.2,0,0,1),(0,0.2,0,1),(0,0,0.2,1),(0.2,0.2,0,1),
            (0.2,0,0.2,1),(0,0.2,0.2,1),(0.1,0.1,0.1,1),(0.15,0.1,0,1)]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = COLS[index % len(COLS)]
    _mat_cache[name] = mat
    return mat

def clean_parts():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("part_cc_"):
            bpy.data.objects.remove(obj, do_unlink=True)

def stl_path(frame):
    for pat in [f"ExtractSurface1_frame_{frame:04d}.stl",
                f"dump_plane1stl_frame_{frame:04d}.stl",
                f"Triangulate1_frame_{frame:04d}.stl"]:
        p = STL_FOLDER / pat
        if p.exists(): return p
    return STL_FOLDER / f"ExtractSurface1_frame_{frame:04d}.stl"

def import_parts(frame):
    path = stl_path(frame)
    if not path.exists(): return []
    try:
        bpy.ops.wm.stl_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(path))
    imp = bpy.context.selected_objects[0]
    imp.name = f"STL_cc_{frame}"
    bpy.ops.object.select_all(action="DESELECT")
    imp.select_set(True)
    bpy.context.view_layer.objects.active = imp
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(imp, do_unlink=True)
    result = []
    for i, obj in enumerate(bpy.context.selected_objects):
        obj.name = f"part_cc_{frame}_{i:03d}"
        m = _get_mat(i)
        if obj.data.materials: obj.data.materials[0] = m
        else: obj.data.materials.append(m)
        mw = obj.matrix_world
        vw = [mw @ v.co for v in obj.data.vertices]
        c = np.mean([[v.x,v.y,v.z] for v in vw], axis=0) if vw else np.zeros(3)
        result.append((obj, i, c))
    return result


# ── Hungarian ID tracking ─────────────────────────────────────────────────────
_next_id      = 1
_prev_cents   = None
_prev_ids     = None

def assign_ids(cents):
    global _next_id, _prev_cents, _prev_ids
    n = len(cents)
    if n == 0: return np.array([], dtype=int)
    curr = np.array(cents)
    if _prev_cents is None or len(_prev_cents) == 0:
        ids = np.arange(_next_id, _next_id+n, dtype=int)
        _next_id += n
        _prev_cents, _prev_ids = curr, ids
        return ids
    diff = curr[:,None,:] - _prev_cents[None,:,:]
    cost = np.sqrt((diff**2).sum(axis=2))
    ri, ci = linear_sum_assignment(cost)
    ids = np.full(n, -1, dtype=int)
    for r,c in zip(ri,ci):
        if cost[r,c] <= MAX_MATCH_DIST:
            ids[r] = _prev_ids[c]
    for i in range(n):
        if ids[i] == -1:
            ids[i] = _next_id; _next_id += 1
    _prev_cents, _prev_ids = curr, ids
    return ids


# ── Projection helper ─────────────────────────────────────────────────────────
def project_centroid(scene, cam_obj, centroid_3d, rw, rh):
    wp = Vector(centroid_3d)
    co = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, wp)
    if co.z < 0: return None, None
    cx = co.x * rw
    cy = (1.0 - co.y) * rh
    if not (0 <= cx < rw and 0 <= cy < rh): return None, None
    return int(cx), int(cy)

def sample_patch(img, cx, cy, rw, rh):
    x1=max(0,cx-PATCH_HALF); x2=min(rw,cx+PATCH_HALF+1)
    y1=max(0,cy-PATCH_HALF); y2=min(rh,cy+PATCH_HALF+1)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0: return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1,3), axis=0)   # (H, S, V)


# ── Args ──────────────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--frames", default=None)
args = parser.parse_args(argv)

scene  = bpy.context.scene
render = scene.render
rw, rh = render.resolution_x, render.resolution_y

fs = scene.frame_start; fe = scene.frame_end
if args.frames:
    p = args.frames.split("-")
    fs, fe = int(p[0]), (int(p[1]) if len(p)>1 else int(p[0]))

cameras = {name: bpy.data.objects.get(name) for name,_,_ in CAMERA_CONFIGS}
cameras = {k:v for k,v in cameras.items() if v}

print(f"\n{'='*65}")
print(f"  blender_color_classify.py")
print(f"  Frames: {fs} – {fe}   Cameras: {list(cameras.keys())}")
print(f"{'='*65}\n")

# colour accumulator: track_id → list of (H, S, V) medians
colour_obs = defaultdict(list)

for frame in range(fs, fe+1):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    clean_parts()
    parts = import_parts(frame)
    if not parts: continue

    cents = [c for _,_,c in parts]
    ids   = assign_ids(cents)

    # 3-D isolation: min distance to any other particle
    n = len(cents)
    iso = np.full(n, np.inf)
    if n > 1:
        arr = np.array(cents)
        for i in range(n):
            dists = np.linalg.norm(arr - arr[i], axis=1)
            dists[i] = np.inf
            iso[i] = dists.min()

    for cam_name, cam_sub, cam_short in CAMERA_CONFIGS:
        cam_obj = cameras.get(cam_name)
        if cam_obj is None: continue

        img_path = IMAGES_DIR / f"{cam_short}_frame_{frame:04d}.png"
        if not img_path.exists(): continue
        img = cv2.imread(str(img_path))
        if img is None: continue

        for i, (obj, mat_idx, cent3d) in enumerate(parts):
            if iso[i] < MIN_ISOLATION_3D: continue   # too crowded in 3-D

            cx, cy = project_centroid(scene, cam_obj, cent3d, rw, rh)
            if cx is None: continue

            hsv_med = sample_patch(img, cx, cy, rw, rh)
            if hsv_med is None: continue

            h, s, v = float(hsv_med[0]), float(hsv_med[1]), float(hsv_med[2])
            if v < 60: continue   # background pixel

            track_id = int(ids[i])
            colour_obs[track_id].append((h, s, v))

    if frame % 100 == 0:
        print(f"  Frame {frame:04d}: {len(parts)} parts, "
              f"{sum(len(v) for v in colour_obs.values())} total colour samples")

# ── Build track_id → class mapping ───────────────────────────────────────────
print(f"\nColour classification:")
track_class = {}
for tid, obs in sorted(colour_obs.items()):
    arr = np.array(obs)
    med_h = np.median(arr[:,0]); med_s = np.median(arr[:,1]); med_v = np.median(arr[:,2])
    cid = hsv_to_class(med_h, med_s, med_v)
    track_class[tid] = cid
    print(f"  track {tid:2d}: {len(obs):4d} samples  "
          f"H={med_h:.0f} S={med_s:.0f} V={med_v:.0f}  → "
          f"class {cid} ({CLASS_NAMES[cid]})")

# Write CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track_id", "class_id", "class_name",
                "n_samples", "median_H", "median_S", "median_V"])
    for tid, obs in sorted(colour_obs.items()):
        arr = np.array(obs)
        cid = track_class[tid]
        w.writerow([tid, cid, CLASS_NAMES[cid], len(obs),
                    round(np.median(arr[:,0]),1),
                    round(np.median(arr[:,1]),1),
                    round(np.median(arr[:,2]),1)])
print(f"\nTrack→class map written: {OUT_CSV}")

# ── Apply classes to YOLO label files ────────────────────────────────────────
# Strategy: for each label file, re-read gt_obb.txt to get track_id per centroid,
# then update the YOLO label's class accordingly.
# We use the front camera MOT data; matching is done by pixel centroid proximity.
print("\nApplying corrected classes to YOLO labels …")

# Load front-camera MOT data: frame → list of (track_id, cx_px, cy_px)
mot_front = defaultdict(list)  # frame_number → [(track_id, cx, cy)]
mot_obb_file = MOT_DIR / "cam_01_front" / "gt" / "gt_obb.txt"
if mot_obb_file.exists():
    with open(mot_obb_file) as f:
        for line in f:
            if line.startswith("#"): continue
            cols = line.strip().split(",")
            if len(cols) < 10: continue
            seq_idx  = int(cols[0])
            tid      = int(cols[1])
            corners  = list(map(float, cols[2:10]))
            cx = np.mean(corners[0::2])
            cy = np.mean(corners[1::2])
            frame_num = 100 + seq_idx - 1
            mot_front[frame_num].append((tid, cx, cy))

# Relabel YOLO files for all cameras
n_fixed = 0
for cam_name, cam_sub, cam_short in CAMERA_CONFIGS:
    # Load this camera's MOT data
    mot_cam = defaultdict(list)
    mot_file = MOT_DIR / cam_sub / "gt" / "gt_obb.txt"
    if mot_file.exists():
        with open(mot_file) as f:
            for line in f:
                if line.startswith("#"): continue
                cols = line.strip().split(",")
                if len(cols) < 10: continue
                seq_idx = int(cols[0])
                tid     = int(cols[1])
                corners = list(map(float, cols[2:10]))
                cx = np.mean(corners[0::2]); cy = np.mean(corners[1::2])
                mot_cam[100+seq_idx-1].append((tid, cx, cy))

    for lbl_path in sorted(LABELS_DIR.glob(f"{cam_short}_frame_*.txt")):
        frame_num = int(lbl_path.stem.split("_frame_")[1])
        mot_entries = mot_cam.get(frame_num, [])
        if not mot_entries: continue

        lines = lbl_path.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) != 9:
                new_lines.append(line); continue
            coords = list(map(float, parts[1:]))
            cx = np.mean([coords[i] * rw for i in range(0,8,2)])
            cy = np.mean([coords[i] * rh for i in range(1,8,2)])
            # find closest MOT entry
            best_tid, best_dist = -1, np.inf
            for tid, mx, my in mot_entries:
                d = ((cx-mx)**2 + (cy-my)**2)**0.5
                if d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid in track_class and best_dist < 50:
                new_cid = track_class[best_tid]
            else:
                new_cid = int(parts[0])   # keep old if no match
            new_lines.append(f"{new_cid} " + " ".join(f"{v:.6f}" for v in coords))
            n_fixed += 1
        lbl_path.write_text("\n".join(new_lines))

print(f"  Updated {n_fixed} YOLO annotation class labels")

# ── Update MOT gt_obb.txt files ───────────────────────────────────────────────
print("Updating MOT gt_obb.txt files …")
for cam_name, cam_sub, cam_short in CAMERA_CONFIGS:
    mot_file = MOT_DIR / cam_sub / "gt" / "gt_obb.txt"
    if not mot_file.exists(): continue
    lines = mot_file.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("#"):
            new_lines.append(line); continue
        cols = line.split(",")
        if len(cols) < 12:
            new_lines.append(line); continue
        tid = int(cols[1])
        if tid in track_class:
            cols[11] = str(track_class[tid] + 1)  # 1-based in MOT format
        new_lines.append(",".join(cols))
    mot_file.write_text("\n".join(new_lines))
    print(f"  Updated {mot_file}")

print(f"\n{'='*65}")
print(f"  Done. Track→class map: {OUT_CSV}")
print(f"  YOLO labels fixed: {n_fixed}")
print(f"{'='*65}")
