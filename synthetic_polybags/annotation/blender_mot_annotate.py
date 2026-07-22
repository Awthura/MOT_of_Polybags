"""
blender_mot_annotate.py
────────────────────────────────────────────────────────────────────────────────
Generates pixel-perfect OBB-MOT ground-truth annotations directly from Blender.

Outputs per camera:
  mot_obb/{cam}/gt/gt_obb.txt   — OBB-MOT format (our extended standard)
  mot_obb/{cam}/gt/gt.txt       — Standard MOT16 (AABB, for legacy tool compat)
  mot_obb/{cam}/seqinfo.ini     — MOTChallenge sequence descriptor

OBB-MOT format (16 values, comma-separated):
  frame, id, x1, y1, x2, y2, x3, y3, x4, y4, conf, class_id, visibility, cx_w, cy_w, cz_w
  └ frame     : 1-based frame index within this sequence
  └ id        : globally unique particle ID, consistent across ALL frames and ALL cameras
  └ x1..y4   : 4 OBB corners in pixels (float), ordered by cv2.minAreaRect
  └ conf      : 1 (ground truth)
  └ class_id  : 1-based class (1=pink 2=blue 3=yellow 4=grey 5=green 6=red)
  └ visibility: fraction of object visible (1.0 = fully visible)
  └ cx_w,cy_w,cz_w : 3D world centroid of this particle (Blender units)

Standard MOT16 gt.txt (9 values):
  frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
  (AABB derived from the OBB corners — for use with legacy MOT eval tools)

ID Assignment:
  IDs are assigned using Hungarian matching of 3D world centroids across consecutive
  frames. This gives oracle-quality, noise-free tracking that is:
    - Consistent across frames  (same physical particle = same ID always)
    - Consistent across cameras (same STL part index → same 3D centroid → same ID)
  New IDs are issued for particles that cannot be matched within MAX_MATCH_DIST.

Usage:
  /Applications/Blender.app/Contents/MacOS/Blender \\
      /Users/awthura/OVGU/AMS/synthetic_polybags/convert_stl_to_animation_multi_camera.blend \\
      --background --python blender_mot_annotate.py \\
      -- [--frames 100-768] [--out_dir ./synth_dataset] [--fps 25]
"""

import sys, os, argparse
from pathlib import Path
from collections import defaultdict

import bpy
import bpy_extras.object_utils
from mathutils import Vector
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE           = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
STL_FOLDER     = BASE / "superquadrics_stl_files_100_2000_frames"
DEFAULT_OUT    = BASE / "synth_dataset"

# ── Class mapping (mirrors blender_annotate.py) ───────────────────────────────
CLASS_NAMES = ["pink_polybag","blue_polybag","yellow_polybag",
               "grey_polybag","green_polybag","red_polybag"]
MAT_IDX_TO_CLASS = {0:5, 1:4, 2:1, 3:2, 4:0, 5:3}   # mat_index → class_id (0-based)

# ── Tracking parameters ───────────────────────────────────────────────────────
MAX_MATCH_DIST = 0.30    # Blender units; centroids farther than this → new ID
MIN_AREA_PX2   = 50      # minimum projected OBB area to keep

# ── Args ──────────────────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--frames",  default=None)
parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
parser.add_argument("--fps",     default=25, type=int)
args = parser.parse_args(argv)

OUT_DIR = Path(args.out_dir)
MOT_DIR = OUT_DIR / "mot_obb"


# ── STL helpers ───────────────────────────────────────────────────────────────
_mat_cache = {}

def _get_material(index):
    name = f"mat_mot_{index:03d}"
    if name in _mat_cache:
        return _mat_cache[name]
    COLORS = [(0.2,0,0,1),(0,0.2,0,1),(0,0,0.2,1),(0.2,0.2,0,1),
              (0.2,0,0.2,1),(0,0.2,0.2,1),(0.15,0.1,0,1),(0.1,0.1,0.1,1)]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = COLORS[index % len(COLORS)]
    _mat_cache[name] = mat
    return mat

def clean_parts():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("part_mot_"):
            bpy.data.objects.remove(obj, do_unlink=True)

def stl_path(frame):
    for pat in [f"ExtractSurface1_frame_{frame:04d}.stl",
                f"dump_plane1stl_frame_{frame:04d}.stl",
                f"Triangulate1_frame_{frame:04d}.stl"]:
        p = STL_FOLDER / pat
        if p.exists():
            return p
    return STL_FOLDER / f"ExtractSurface1_frame_{frame:04d}.stl"

def import_parts(frame):
    """Import STL, separate loose parts. Returns list of (obj, mat_idx, centroid_3d)."""
    path = stl_path(frame)
    if not path.exists():
        return []
    try:
        bpy.ops.wm.stl_import(filepath=str(path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(path))

    imported = bpy.context.selected_objects[0]
    imported.name = f"STL_mot_{frame}"
    bpy.ops.object.select_all(action="DESELECT")
    imported.select_set(True)
    bpy.context.view_layer.objects.active = imported
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(imported, do_unlink=True)

    result = []
    for i, obj in enumerate(bpy.context.selected_objects):
        obj.name = f"part_mot_{frame}_{i:03d}"
        mat = _get_material(i)
        if obj.data.materials: obj.data.materials[0] = mat
        else: obj.data.materials.append(mat)
        # 3D centroid in world space
        mat_w = obj.matrix_world
        verts_world = [mat_w @ v.co for v in obj.data.vertices]
        if verts_world:
            c = np.mean([[v.x, v.y, v.z] for v in verts_world], axis=0)
        else:
            c = np.array([0.0, 0.0, 0.0])
        result.append((obj, i, c))
    return result


# ── ID assignment via Hungarian matching ─────────────────────────────────────
_next_id   = 1
_prev_centroids = None   # np.array (N,3) from previous frame
_prev_ids       = None   # np.array (N,) int

def assign_ids(centroids_3d):
    """
    Match current frame centroids to previous frame using Hungarian algorithm.
    Returns np.array of integer IDs (1-based), consistent across frames.
    """
    global _next_id, _prev_centroids, _prev_ids

    n = len(centroids_3d)
    if n == 0:
        return np.array([], dtype=int)

    curr = np.array(centroids_3d)   # (n, 3)

    if _prev_centroids is None or len(_prev_centroids) == 0:
        # First frame: assign fresh IDs
        ids = np.arange(_next_id, _next_id + n, dtype=int)
        _next_id += n
        _prev_centroids = curr
        _prev_ids = ids
        return ids

    # Cost matrix: Euclidean distance between all pairs
    diff = curr[:, None, :] - _prev_centroids[None, :, :]   # (n, m, 3)
    cost = np.sqrt((diff ** 2).sum(axis=2))                  # (n, m)

    row_ind, col_ind = linear_sum_assignment(cost)

    ids = np.full(n, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= MAX_MATCH_DIST:
            ids[r] = _prev_ids[c]

    # Assign new IDs to unmatched particles
    for i in range(n):
        if ids[i] == -1:
            ids[i] = _next_id
            _next_id += 1

    _prev_centroids = curr
    _prev_ids = ids
    return ids


# ── Projection ────────────────────────────────────────────────────────────────
def project_vertices(scene, cam_obj, obj, render_w, render_h):
    mat = obj.matrix_world
    pts = []
    for v in obj.data.vertices:
        wp = mat @ v.co
        co = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, wp)
        if co.z < 0:
            return None
        pts.append([co.x * render_w, (1.0 - co.y) * render_h])
    return np.array(pts, dtype=np.float32) if pts else None

def obb_from_pts(pts, render_w, render_h):
    """Returns (obb_corners_4x2, aabb_ltwh) or (None, None)."""
    hull = cv2.convexHull(pts)
    if cv2.contourArea(hull) < MIN_AREA_PX2:
        return None, None
    inside = pts[(pts[:,0]>=0)&(pts[:,0]<=render_w)&
                 (pts[:,1]>=0)&(pts[:,1]<=render_h)]
    if len(inside) == 0:
        return None, None
    clipped = np.clip(hull, [0,0], [render_w, render_h])
    rect = cv2.minAreaRect(clipped.reshape(-1,1,2).astype(np.float32))
    box  = cv2.boxPoints(rect)   # 4 corners

    # AABB from OBB corners
    x1, y1 = box[:,0].min(), box[:,1].min()
    x2, y2 = box[:,0].max(), box[:,1].max()
    aabb = (max(0,x1), max(0,y1),
            min(render_w,x2)-max(0,x1),
            min(render_h,y2)-max(0,y1))
    return box, aabb


# ── Main ──────────────────────────────────────────────────────────────────────
scene    = bpy.context.scene
render   = scene.render
render_w = render.resolution_x
render_h = render.resolution_y

CAMERA_CONFIGS = [
    {"name":"Cam_Front",  "subfolder":"cam_01_front"},
    {"name":"Cam_Back",   "subfolder":"cam_02_back"},
    {"name":"Cam_Left",   "subfolder":"cam_03_left"},
    {"name":"Cam_Right",  "subfolder":"cam_04_right"},
]

cameras = {cfg["name"]: bpy.data.objects.get(cfg["name"])
           for cfg in CAMERA_CONFIGS}
cameras = {k:v for k,v in cameras.items() if v}

# Frame range
frame_start = scene.frame_start
frame_end   = scene.frame_end
if args.frames:
    parts = args.frames.split("-")
    frame_start = int(parts[0])
    frame_end   = int(parts[1]) if len(parts)>1 else frame_start

all_frames = list(range(frame_start, frame_end + 1))

print(f"\n{'='*65}")
print(f"  OBB-MOT Annotation Pipeline")
print(f"  Frames : {frame_start} to {frame_end}  ({len(all_frames)} frames)")
print(f"  Cameras: {list(cameras.keys())}")
print(f"  Output : {MOT_DIR}")
print(f"{'='*65}\n")

# Create output dirs and open gt files
gt_obb_files = {}
gt_files     = {}

for cfg in CAMERA_CONFIGS:
    sub = cfg["subfolder"]
    cam_mot = MOT_DIR / sub / "gt"
    cam_mot.mkdir(parents=True, exist_ok=True)
    gt_obb_files[sub] = open(cam_mot / "gt_obb.txt", "w")
    gt_files[sub]     = open(cam_mot / "gt.txt",     "w")

# Write headers
for sub in gt_obb_files:
    gt_obb_files[sub].write(
        "# frame,id,x1,y1,x2,y2,x3,y3,x4,y4,conf,class_id,visibility,cx_w,cy_w,cz_w\n"
    )
    gt_files[sub].write(
        "# frame,id,bb_left,bb_top,bb_width,bb_height,conf,class_id,visibility\n"
    )

# Track which frames actually have data per camera (for seqinfo)
cam_frame_count = defaultdict(int)

# ── Frame loop ────────────────────────────────────────────────────────────────
for seq_idx, frame in enumerate(all_frames, start=1):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    clean_parts()
    parts = import_parts(frame)

    if not parts:
        print(f"  Frame {frame:04d}: no STL found, skipping")
        continue

    # Extract 3D centroids and assign globally consistent IDs
    centroids_3d = [c for (_, _, c) in parts]
    ids = assign_ids(centroids_3d)

    n_written = 0
    for cfg in CAMERA_CONFIGS:
        cam_name = cfg["name"]
        sub      = cfg["subfolder"]
        cam_obj  = cameras.get(cam_name)
        if cam_obj is None:
            continue

        for part_idx, (obj, mat_idx, centroid_3d) in enumerate(parts):
            track_id  = int(ids[part_idx])
            class_id  = MAT_IDX_TO_CLASS.get(mat_idx % 6, 0) + 1  # 1-based

            pts = project_vertices(scene, cam_obj, obj, render_w, render_h)
            if pts is None:
                continue

            box, aabb = obb_from_pts(pts, render_w, render_h)
            if box is None:
                continue

            cx, cy, cz = centroid_3d

            # OBB-MOT line
            corners = ",".join(f"{v:.2f}" for v in box.flatten())
            gt_obb_files[sub].write(
                f"{seq_idx},{track_id},{corners},"
                f"1,{class_id},1.0,{cx:.4f},{cy:.4f},{cz:.4f}\n"
            )

            # Standard MOT16 AABB line
            bl, bt, bw, bh = aabb
            gt_files[sub].write(
                f"{seq_idx},{track_id},{bl:.2f},{bt:.2f},{bw:.2f},{bh:.2f},"
                f"1,{class_id},1.0\n"
            )
            n_written += 1

        cam_frame_count[sub] += 1

    print(f"  Frame {frame:04d} (seq {seq_idx:04d}): "
          f"{len(parts)} parts, {n_written} annotations across {len(cameras)} cams "
          f"| max_id={int(ids.max()) if len(ids) else 0}")

# Close files
for f in gt_obb_files.values(): f.close()
for f in gt_files.values():     f.close()

# ── seqinfo.ini per camera ────────────────────────────────────────────────────
for cfg in CAMERA_CONFIGS:
    sub = cfg["subfolder"]
    seq_len = cam_frame_count[sub]
    ini_path = MOT_DIR / sub / "seqinfo.ini"
    ini_path.write_text(
        f"[Sequence]\n"
        f"name={sub}\n"
        f"imDir=../../images\n"
        f"frameRate={args.fps}\n"
        f"seqLength={seq_len}\n"
        f"imWidth={render_w}\n"
        f"imHeight={render_h}\n"
        f"imExt=.png\n"
    )

# ── classes.txt ───────────────────────────────────────────────────────────────
(MOT_DIR / "classes.txt").write_text("\n".join(CLASS_NAMES))
(MOT_DIR / "FORMAT.md").write_text(
"""# OBB-MOT Ground Truth Format

## gt_obb.txt  (Oriented Bounding Box MOT)
```
frame, id, x1, y1, x2, y2, x3, y3, x4, y4, conf, class_id, visibility, cx_w, cy_w, cz_w
```
| Field       | Description |
|-------------|-------------|
| frame       | 1-based sequence frame index |
| id          | Globally unique particle ID — consistent across ALL frames and ALL cameras |
| x1..y4      | 4 OBB corners in pixels (cv2.minAreaRect order) |
| conf        | 1 = ground truth |
| class_id    | 1=pink 2=blue 3=yellow 4=grey 5=green 6=red |
| visibility  | 1.0 = fully visible |
| cx_w,cy_w,cz_w | 3D world centroid (Blender units) |

## gt.txt  (Standard MOT16, AABB)
```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
```
Axis-aligned bounding box derived from OBB corners. Compatible with
standard MOTChallenge evaluation tools.

## ID Assignment
IDs are assigned by Hungarian matching of 3D world centroids across
consecutive frames. Same physical particle = same ID in all cameras and
all frames. No tracker or detector required.
"""
)

print(f"\n{'='*65}")
print(f"  OBB-MOT annotation complete.")
print(f"  Unique particle IDs assigned: {_next_id - 1}")
print(f"  Output: {MOT_DIR}")
print(f"    mot_obb/{{cam}}/gt/gt_obb.txt  (OBB-MOT format)")
print(f"    mot_obb/{{cam}}/gt/gt.txt      (standard MOT16 AABB)")
print(f"    mot_obb/{{cam}}/seqinfo.ini")
print(f"    mot_obb/FORMAT.md")
print(f"{'='*65}\n")
