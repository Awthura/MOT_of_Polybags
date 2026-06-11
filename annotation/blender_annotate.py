"""
blender_annotate.py
────────────────────────────────────────────────────────────────────────────────
Drop-in annotation companion to the existing Blender render pipeline.
For every frame × camera it:
  1. Imports the per-frame STL and separates loose polybag parts
     (exactly as the existing render script does).
  2. Projects each part's 3-D bounding box through the camera matrix
     to get a pixel-perfect 2-D oriented bounding box.
  3. Writes a YOLO OBB .txt label file alongside the rendered PNG.

Run (background, no window):
    /Applications/Blender.app/Contents/MacOS/Blender \
        /Users/awthura/OVGU/AMS/convert_stl_to_animation_multi_camera.blend \
        --background --python blender_annotate.py \
        -- [--frames 100-768] [--render] [--out_dir ./synth_dataset] [--dry_run]

Flags:
    --frames    100-768   Frame range (default: reads from scene)
    --render              Also render PNGs (default: labels only, images assumed
                          to already exist in out_dir/{cam}/images/)
    --out_dir PATH        Root output dir  (default: same as existing renders)
    --dry_run             Print objects found on frame 100, then exit
"""

import sys, os, math, argparse
from pathlib import Path

import bpy
import bpy_extras.object_utils
from mathutils import Vector
import numpy as np
import cv2

# ── Local paths  (Mac) ────────────────────────────────────────────────────────
BASE = Path("/Users/awthura/OVGU/AMS")

# STL files live in one folder but use two filename patterns depending on frame.
# The lookup function below handles both automatically.
STL_FOLDER   = BASE / "superquadrics_stl_files_100_2000_frames"

def stl_path_for_frame(frame):
    """Return the STL path for a given frame, trying both known filename patterns."""
    candidates = [
        STL_FOLDER / f"ExtractSurface1_frame_{frame:04d}.stl",
        STL_FOLDER / f"dump_plane1stl_frame_{frame:04d}.stl",
        STL_FOLDER / f"Triangulate1_frame_{frame:04d}.stl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]   # return first anyway; import_frame_parts will warn

# Default output dir
DEFAULT_OUT  = BASE / "synth_dataset"

# ── Class mapping ─────────────────────────────────────────────────────────────
# Each polybag part inherits a material assigned by its index inside the STL.
# Material index → class ID mirrors the DARK_COLORS order in the render script.
# Set SINGLE_CLASS = True to label every part as class 0 "polybag" instead.
SINGLE_CLASS = False

CLASS_NAMES = [
    "pink_polybag",    # mat index 4 (dark magenta)
    "blue_polybag",    # mat index 2 (dark blue)
    "yellow_polybag",  # mat index 3 (dark yellow/olive)
    "grey_polybag",    # mat index 5 (dark cyan)
    "green_polybag",   # mat index 1 (dark green)
    "red_polybag",     # mat index 0 (dark red)
]

# Maps DARK_COLORS index → CLASS_NAMES index
#  0=dark red → red(5), 1=dark green → green(4), 2=dark blue → blue(1),
#  3=dark yellow → yellow(2), 4=dark magenta → pink(0), 5=dark cyan → grey(3)
MAT_IDX_TO_CLASS = {0: 5, 1: 4, 2: 1, 3: 2, 4: 0, 5: 3}

# ── Camera config (must match the render script) ───────────────────────────────
CAMERA_CONFIGS = [
    {"name": "Cam_Front",  "subfolder": "cam_01_front"},
    {"name": "Cam_Back",   "subfolder": "cam_02_back"},
    {"name": "Cam_Left",   "subfolder": "cam_03_left"},
    {"name": "Cam_Right",  "subfolder": "cam_04_right"},
]

# Minimum projected 2-D area (px²) — parts smaller than this are skipped.
MIN_AREA_PX2 = 50


# ── Argument parsing ───────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--frames",     default=None, help="e.g. 100-768")
parser.add_argument("--frame_list", default=None, help="File with one frame number per line")
parser.add_argument("--render",     action="store_true", help="Also render PNGs")
parser.add_argument("--out_dir",    default=str(DEFAULT_OUT))
parser.add_argument("--dry_run",    action="store_true")
args = parser.parse_args(argv)

OUT_DIR = Path(args.out_dir)


# ── Blender helpers (mirrors the render script) ────────────────────────────────

_material_cache = {}

def _get_or_create_material(index):
    name = f"mat_f1_{index:03d}"
    if name in _material_cache:
        return _material_cache[name]
    DARK_COLORS = [
        (0.2, 0.0, 0.0, 1), (0.0, 0.2, 0.0, 1), (0.0, 0.0, 0.2, 1),
        (0.2, 0.2, 0.0, 1), (0.2, 0.0, 0.2, 1), (0.0, 0.2, 0.2, 1),
        (0.15,0.1, 0.0, 1), (0.1, 0.1, 0.1, 1), (0.2, 0.1, 0.0, 1),
        (0.1, 0.0, 0.15,1), (0.0, 0.15,0.1, 1), (0.15,0.0, 0.1, 1),
        (0.1, 0.15,0.0, 1), (0.0, 0.1, 0.15,1), (0.15,0.15,0.0, 1),
        (0.1, 0.05,0.15,1),
    ]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = DARK_COLORS[index % len(DARK_COLORS)]
    bsdf.inputs["Roughness"].default_value  = 0.8
    _material_cache[name] = mat
    return mat


def clean_parts():
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name.startswith("part_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def import_frame_parts(frame):
    """Import STL for this frame, separate loose parts, assign materials.
    Returns list of (mesh_object, material_index) tuples."""
    stl_path = stl_path_for_frame(frame)
    if not stl_path.exists():
        print(f"  SKIP: STL not found: {stl_path}")
        return []

    # Blender 4+ uses wm.stl_import; older versions used import_mesh.stl
    try:
        bpy.ops.wm.stl_import(filepath=str(stl_path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(stl_path))
    imported = bpy.context.selected_objects[0]
    imported.name = f"STL_import_{frame}"

    # Separate loose parts
    bpy.ops.object.select_all(action="DESELECT")
    imported.select_set(True)
    bpy.context.view_layer.objects.active = imported
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(imported, do_unlink=True)

    parts = bpy.context.selected_objects
    result = []
    for i, part in enumerate(parts):
        part.name = f"part_f1_{frame}_{i:03d}"
        mat = _get_or_create_material(i)
        if part.data.materials:
            part.data.materials[0] = mat
        else:
            part.data.materials.append(mat)
        result.append((part, i))
    return result


# ── Projection helpers ─────────────────────────────────────────────────────────

def project_part(scene, cam_obj, part, render_w, render_h):
    """Project all mesh vertices → 2-D pixel coords for a tight OBB.
    Falls back to bounding-box corners if the mesh has no vertices.
    Returns np.array (N,2) float32, or None if object is behind camera."""
    mat  = part.matrix_world
    mesh = part.data

    # Use actual vertices for accurate rotated OBB
    verts = mesh.vertices if mesh.vertices else []
    sources = [mat @ v.co for v in verts] if verts else \
              [mat @ Vector(c) for c in part.bound_box]

    if not sources:
        return None

    pts_px = []
    for world_pt in sources:
        co = bpy_extras.object_utils.world_to_camera_view(scene, cam_obj, world_pt)
        if co.z < 0:
            return None           # any vertex behind camera → skip whole object
        pts_px.append([co.x * render_w, (1.0 - co.y) * render_h])

    return np.array(pts_px, dtype=np.float32)


def corners_to_yolo_obb(corners_px, render_w, render_h):
    """Fit OBB to projected corners. Returns normalised 8-coord list or None."""
    hull = cv2.convexHull(corners_px)
    area = cv2.contourArea(hull)
    if area < MIN_AREA_PX2:
        return None

    # Check at least some corners are inside the frame
    inside = corners_px[
        (corners_px[:, 0] >= 0) & (corners_px[:, 0] <= render_w) &
        (corners_px[:, 1] >= 0) & (corners_px[:, 1] <= render_h)
    ]
    if len(inside) == 0:
        return None

    clipped = np.clip(hull, [0, 0], [render_w, render_h])
    rect    = cv2.minAreaRect(clipped.reshape(-1, 1, 2).astype(np.float32))
    box     = cv2.boxPoints(rect)

    coords = []
    for (px, py) in box:
        coords.extend([
            max(0.0, min(1.0, px / render_w)),
            max(0.0, min(1.0, py / render_h)),
        ])
    return coords


# ── Main ───────────────────────────────────────────────────────────────────────

scene    = bpy.context.scene
render   = scene.render
render_w = render.resolution_x
render_h = render.resolution_y

# Build the frame list to process
if args.frame_list:
    with open(args.frame_list) as _f:
        frame_list = [int(l.strip()) for l in _f if l.strip().isdigit()]
    frame_start = frame_list[0] if frame_list else scene.frame_start
    frame_end   = frame_list[-1] if frame_list else scene.frame_end
elif args.frames:
    _p = args.frames.split("-")
    frame_start = int(_p[0])
    frame_end   = int(_p[1]) if len(_p) > 1 else frame_start
    frame_list  = list(range(frame_start, frame_end + 1))
else:
    frame_start = scene.frame_start
    frame_end   = scene.frame_end
    frame_list  = list(range(frame_start, frame_end + 1))

# Cameras present in the scene
cameras = {cfg["name"]: bpy.data.objects.get(cfg["name"])
           for cfg in CAMERA_CONFIGS}
cameras = {k: v for k, v in cameras.items() if v is not None}

print(f"\n{'='*65}")
print(f"  Blender annotation pipeline")
print(f"  STL folder  : {STL_FOLDER}")
print(f"  Out dir     : {OUT_DIR}")
print(f"  Frames      : {len(frame_list)} (first={frame_start}, last={frame_end})")
print(f"  Source      : {'--frame_list' if args.frame_list else '--frames'}")
print(f"  Cameras     : {list(cameras.keys())}")
print(f"  Render PNGs : {args.render}")
print(f"  Single class: {SINGLE_CLASS}")
print(f"{'='*65}\n")

# Create output dirs
(OUT_DIR / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "classes.txt").write_text("\n".join(
    ["polybag"] if SINGLE_CLASS else CLASS_NAMES
))
for cfg in CAMERA_CONFIGS:
    (OUT_DIR / cfg["subfolder"] / "labels").mkdir(parents=True, exist_ok=True)
    if args.render:
        (OUT_DIR / cfg["subfolder"] / "images").mkdir(parents=True, exist_ok=True)

# Dry run: just show what the first frame looks like
if args.dry_run:
    scene.frame_set(frame_list[0])
    bpy.context.view_layer.update()
    clean_parts()
    parts = import_frame_parts(frame_list[0])
    print(f"Dry run — frame {frame_list[0]}: {len(parts)} parts found")
    for obj, mat_idx in parts[:10]:
        cid = 0 if SINGLE_CLASS else MAT_IDX_TO_CLASS.get(mat_idx % 6, 0)
        print(f"  {obj.name:35s}  mat_idx={mat_idx}  → class {cid} ({CLASS_NAMES[cid]})")
    sys.exit(0)

# Main loop
original_camera = scene.camera
total_labels = 0

for frame in frame_list:
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    clean_parts()
    parts = import_frame_parts(frame)

    if not parts:
        continue

    for cfg in CAMERA_CONFIGS:
        cam_name = cfg["name"]
        cam_obj  = cameras.get(cam_name)
        if cam_obj is None:
            continue

        subfolder  = cfg["subfolder"]
        label_path = OUT_DIR / subfolder / "labels" / f"frame_{frame:04d}.txt"
        img_path   = OUT_DIR / subfolder / "images"  / f"frame_{frame:04d}.png"

        # Skip this camera/frame if both image and label already exist
        if args.render and img_path.exists() and label_path.exists():
            continue

        # Render if requested and not already present
        if args.render:
            scene.camera = cam_obj
            render.filepath = str(img_path)
            render.image_settings.file_format = "PNG"
            bpy.ops.render.render(write_still=True)

        # Annotate
        lines = []
        for obj, mat_idx in parts:
            cid = 0 if SINGLE_CLASS else MAT_IDX_TO_CLASS.get(mat_idx % 6, 0)

            corners = project_part(scene, cam_obj, obj, render_w, render_h)
            if corners is None:
                continue

            coords = corners_to_yolo_obb(corners, render_w, render_h)
            if coords is None:
                continue

            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in coords))

        label_path.write_text("\n".join(lines))
        total_labels += len(lines)

    print(f"  Frame {frame:04d}: {len(parts)} parts, "
          f"{len(lines)} labels (last cam)")

scene.camera = original_camera
print(f"\nDone. {len(frame_list)} frames x "
      f"{len(cameras)} cameras = {total_labels} total annotations")
print(f"Labels → {OUT_DIR}/<cam>/labels/")
