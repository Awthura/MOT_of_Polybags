"""
Full re-annotation using the ORIGINAL blend file's animated object positions.

Strategy:
  Phase 1 (frames 769-1972): blend objects have keyframed positions that exactly
    match the simulation STL centroids.  We match each STL part to one of the 6
    blend objects, recording part → material_name.
  Phase 2 (all frames 100-1972): Hungarian tracker → track_id per STL part.
  Combine: track_id → material_name (majority vote over matched frames).
  Phase 3: sample rendered-image HSV per MATERIAL (aggregating across all frames
    and cameras where that material's part is visible).  Any track that shares a
    material with another isolated track gets classified automatically.
  Output: track_classes.csv  +  re-labelled YOLO labels  +  MOT gt_obb.txt.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
import bpy, bpy_extras.object_utils
from mathutils import Vector
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

BASE        = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
ORIG_BLEND  = BASE / "convert_stl_to_animation_multi_camera_original.blend"
STL_FOLDER  = BASE / "superquadrics_stl_files_100_2000_frames"
IMAGES_DIR  = BASE / "synth_dataset" / "images"
LABELS_DIR  = BASE / "synth_dataset" / "labels"
MOT_DIR     = BASE / "synth_dataset" / "mot_obb"
OUT_CSV     = BASE / "synth_dataset" / "track_classes.csv"

CLASS_NAMES = ["pink_polybag","blue_polybag","yellow_polybag",
               "grey_polybag","green_polybag","red_polybag"]
COLOUR_RULES = [
    (3,  0,180,  0, 30, 80,255),   # grey / white — very low saturation
    (5,  0, 12, 40,255, 80,255),   # red (low side)
    (5,165,180, 40,255, 80,255),   # red (high side, wraps)
    (2, 13, 40, 30,255, 80,255),   # yellow / orange
    (4, 41,100, 25,255, 50,255),   # green / lime / teal
    (1,101,138, 25,255, 50,255),   # blue / cool purple
    (0,139,167, 25,255, 50,255),   # pink / warm purple / magenta
]
def hsv_to_class(h, s, v):
    for cid, h0, h1, s0, s1, v0, v1 in COLOUR_RULES:
        if h0 <= h <= h1 and s0 <= s <= s1 and v0 <= v <= v1:
            return cid
    return 3

MAX_MATCH_DIST = 0.30   # Hungarian track linking distance
BLEND_MATCH_D  = 0.50   # max distance to match STL part → blend object
MIN_ISO        = 0.08   # 3-D isolation threshold for colour sampling
PATCH          = 6      # half-size of pixel patch (13×13)

CAMS = [("Cam_Front","cam_01_front","front"),
        ("Cam_Back", "cam_02_back", "back"),
        ("Cam_Left", "cam_03_left", "left"),
        ("Cam_Right","cam_04_right","right")]

# ── helpers ──────────────────────────────────────────────────────────────────

def stl_path(frame):
    for p in [f"ExtractSurface1_frame_{frame:04d}.stl",
              f"dump_plane1stl_frame_{frame:04d}.stl",
              f"Triangulate1_frame_{frame:04d}.stl"]:
        q = STL_FOLDER / p
        if q.exists(): return q
    return STL_FOLDER / f"ExtractSurface1_frame_{frame:04d}.stl"

_mat_cache = {}
def _tmp_mat(i):
    n = f"mat_cls_{i:03d}"
    if n in _mat_cache: return _mat_cache[n]
    COLS=[(0.2,0,0,1),(0,0.2,0,1),(0,0,0.2,1),(0.2,0.2,0,1),
          (0.2,0,0.2,1),(0,0.2,0.2,1),(0.1,0.1,0.1,1),(0.15,0.1,0,1)]
    m = bpy.data.materials.new(n); m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = COLS[i % len(COLS)]
    _mat_cache[n] = m; return m

def clean():
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.name.startswith("part_cls_"):
            bpy.data.objects.remove(o, do_unlink=True)

def import_stl(frame):
    """Import STL for frame, separate into loose parts, return list of (obj, centroid_3d)."""
    path = stl_path(frame)
    if not path.exists(): return []
    try: bpy.ops.wm.stl_import(filepath=str(path))
    except: bpy.ops.import_mesh.stl(filepath=str(path))
    imp = bpy.context.selected_objects[0]; imp.name = f"STL_cls_{frame}"
    bpy.ops.object.select_all(action="DESELECT")
    imp.select_set(True); bpy.context.view_layer.objects.active = imp
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(imp, do_unlink=True)
    res = []
    for i, o in enumerate(bpy.context.selected_objects):
        o.name = f"part_cls_{frame}_{i:03d}"
        m = _tmp_mat(i)
        if o.data.materials: o.data.materials[0] = m
        else: o.data.materials.append(m)
        mw = o.matrix_world
        vw = [mw @ v.co for v in o.data.vertices]
        c = np.mean([[v.x,v.y,v.z] for v in vw], axis=0) if vw else np.zeros(3)
        res.append((o, c))
    return res

# ── Hungarian tracker ─────────────────────────────────────────────────────────
_nid = 1; _prev_cents = None; _prev_ids = None
def assign_ids(cents):
    global _nid, _prev_cents, _prev_ids
    n = len(cents)
    if n == 0: return np.array([], dtype=int)
    curr = np.array(cents)
    if _prev_cents is None or len(_prev_cents) == 0:
        ids = np.arange(_nid, _nid+n, dtype=int); _nid += n
        _prev_cents, _prev_ids = curr, ids; return ids
    diff = curr[:,None,:] - _prev_cents[None,:,:]; cost = np.sqrt((diff**2).sum(2))
    ri, ci = linear_sum_assignment(cost)
    ids = np.full(n, -1, dtype=int)
    for r, c in zip(ri, ci):
        if cost[r,c] <= MAX_MATCH_DIST: ids[r] = _prev_ids[c]
    for i in range(n):
        if ids[i] == -1: ids[i] = _nid; _nid += 1
    _prev_cents, _prev_ids = curr, ids; return ids

# ── Phase 1+2: track frame range ──────────────────────────────────────────────
scene = bpy.context.scene
render = scene.render
rw, rh = render.resolution_x, render.resolution_y
cameras = {n: bpy.data.objects.get(n) for n,_,_ in CAMS}
cameras = {k: v for k,v in cameras.items() if v}

# Blend file objects (6 animated polybags, frames 769-2000)
BLEND_OBJS = [o for o in bpy.data.objects
              if o.type == "MESH" and o.name.startswith("part_f1_")]
print(f"\nBlend objects ({len(BLEND_OBJS)}): {[o.name for o in BLEND_OBJS]}")
print(f"Their materials: {[o.material_slots[0].material.name if o.material_slots else 'none' for o in BLEND_OBJS]}")

# Accumulate per-material and per-track colour samples
mat_colour_obs  = defaultdict(list)   # mat_name → [(H,S,V)]
track_mat_votes = defaultdict(lambda: defaultdict(int))   # track_id → {mat_name: count}
track_colour_obs= defaultdict(list)   # track_id → [(H,S,V)]  (direct, no material lookup)

BLEND_FRAME_START = 769

print(f"\nProcessing frames 100-1972 ...")
print(f"  Blend-object matching active for frames {BLEND_FRAME_START}-1972")

for frame in range(100, 1973):
    scene.frame_set(frame); bpy.context.view_layer.update()
    clean()
    parts = import_stl(frame)
    if not parts: continue
    cents = [c for _, c in parts]
    ids   = assign_ids(cents)

    # ── blend-object position matching (only when blend objects are animated) ──
    part_to_mat = {}   # part_index → mat_name
    if frame >= BLEND_FRAME_START and BLEND_OBJS:
        bcents = []
        for bo in BLEND_OBJS:
            loc = bo.matrix_world.translation
            bcents.append(np.array([loc.x, loc.y, loc.z]))
        if bcents:
            bcents_arr = np.array(bcents)
            scents_arr = np.array(cents)
            # Build cost matrix: STL part → blend object
            diff = scents_arr[:,None,:] - bcents_arr[None,:,:]
            dist = np.sqrt((diff**2).sum(2))
            ri, ci = linear_sum_assignment(dist)
            for r, c in zip(ri, ci):
                if dist[r,c] <= BLEND_MATCH_D:
                    mat_name = (BLEND_OBJS[c].material_slots[0].material.name
                                if BLEND_OBJS[c].material_slots else "unknown")
                    part_to_mat[r] = mat_name
                    track_mat_votes[int(ids[r])][mat_name] += 1

    # ── 3-D isolation score ────────────────────────────────────────────────────
    n = len(cents); iso = np.full(n, np.inf)
    if n > 1:
        arr = np.array(cents)
        for i in range(n):
            d = np.linalg.norm(arr - arr[i], axis=1); d[i] = np.inf; iso[i] = d.min()

    # ── colour sampling from rendered images ───────────────────────────────────
    for cn, cs, cam_short in CAMS:
        cam = cameras.get(cn)
        if not cam: continue
        img_path = IMAGES_DIR / f"{cam_short}_frame_{frame:04d}.png"
        if not img_path.exists(): continue
        img = cv2.imread(str(img_path))
        if img is None: continue
        for i in range(n):
            if iso[i] < MIN_ISO: continue
            wp = Vector(cents[i])
            co = bpy_extras.object_utils.world_to_camera_view(scene, cam, wp)
            if co.z < 0: continue
            cx, cy = int(co.x * rw), int((1 - co.y) * rh)
            if not (0 <= cx < rw and 0 <= cy < rh): continue
            x1=max(0,cx-PATCH); x2=min(rw,cx+PATCH+1)
            y1=max(0,cy-PATCH); y2=min(rh,cy+PATCH+1)
            patch = img[y1:y2, x1:x2]
            if patch.size == 0: continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            med = np.median(hsv.reshape(-1,3), axis=0)
            if float(med[2]) < 60: continue
            sample = tuple(med)
            track_colour_obs[int(ids[i])].append(sample)
            if i in part_to_mat:
                mat_colour_obs[part_to_mat[i]].append(sample)

    if frame % 200 == 0:
        n_track_samples = sum(len(v) for v in track_colour_obs.values())
        n_mat_samples   = sum(len(v) for v in mat_colour_obs.values())
        print(f"  Frame {frame:04d}: {n} parts | "
              f"track samples={n_track_samples} | mat samples={n_mat_samples} | "
              f"mat-matched parts={len(part_to_mat)}")

# ── Phase 3: classify materials, then tracks ──────────────────────────────────
print("\n--- Material colour classification ---")
mat_class = {}
for mat_name, obs in sorted(mat_colour_obs.items()):
    if not obs:
        print(f"  {mat_name}: 0 samples → grey (fallback)")
        mat_class[mat_name] = 3; continue
    arr = np.array(obs)
    h, s, v = np.median(arr[:,0]), np.median(arr[:,1]), np.median(arr[:,2])
    cid = hsv_to_class(h, s, v)
    mat_class[mat_name] = cid
    print(f"  {mat_name}: {len(obs):4d} samples  H={h:.0f} S={s:.0f} V={v:.0f} "
          f"→ class {cid} ({CLASS_NAMES[cid]})")

print("\n--- Track → material vote map ---")
track_mat_assignment = {}
for tid, votes in sorted(track_mat_votes.items()):
    best_mat = max(votes, key=votes.get)
    total    = sum(votes.values())
    print(f"  track {tid:2d}: {dict(votes)}  → {best_mat} ({votes[best_mat]}/{total} frames)")
    track_mat_assignment[tid] = best_mat

# ── Classify every track ──────────────────────────────────────────────────────
print("\n--- Final track classification ---")
all_track_ids = sorted(set(list(track_mat_votes.keys()) + list(track_colour_obs.keys())))
track_class = {}
track_stats = {}  # for CSV

for tid in all_track_ids:
    # Method 1: via material assignment (most reliable — material aggregates all instances)
    if tid in track_mat_assignment:
        mat_name = track_mat_assignment[tid]
        if mat_name in mat_class:
            cid  = mat_class[mat_name]
            obs  = mat_colour_obs.get(mat_name, [])
            arr  = np.array(obs) if obs else np.zeros((1,3))
            h,s,v = (np.median(arr[:,0]), np.median(arr[:,1]), np.median(arr[:,2]))
            track_class[tid] = cid
            track_stats[tid] = (len(obs), h, s, v, f"via {mat_name}")
            print(f"  track {tid:2d}: {len(obs):4d} samples  H={h:.0f} S={s:.0f} V={v:.0f} "
                  f"→ class {cid} ({CLASS_NAMES[cid]})  [mat={mat_name}]")
            continue

    # Method 2: direct track pixel samples (fallback)
    obs = track_colour_obs.get(tid, [])
    if obs:
        arr  = np.array(obs)
        h,s,v= np.median(arr[:,0]), np.median(arr[:,1]), np.median(arr[:,2])
        cid  = hsv_to_class(h, s, v)
        track_class[tid] = cid
        track_stats[tid] = (len(obs), h, s, v, "direct")
        print(f"  track {tid:2d}: {len(obs):4d} samples  H={h:.0f} S={s:.0f} V={v:.0f} "
              f"→ class {cid} ({CLASS_NAMES[cid]})  [direct]")
    else:
        track_class[tid] = 3  # grey fallback
        track_stats[tid] = (0, 0, 0, 0, "fallback")
        print(f"  track {tid:2d}: 0 samples → class 3 (grey_polybag)  [fallback]")

# ── Write track_classes.csv ───────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track_id","class_id","class_name","n_samples",
                "median_H","median_S","median_V","method"])
    for tid in sorted(track_class.keys()):
        cid = track_class[tid]
        n, h, s, v, mth = track_stats[tid]
        w.writerow([tid, cid, CLASS_NAMES[cid], n,
                    round(float(h),1), round(float(s),1), round(float(v),1), mth])
print(f"\nWrote {OUT_CSV}")

# ── Apply classes to YOLO labels and MOT files ────────────────────────────────
print("\nApplying classes to YOLO labels and MOT files …")
for cn, cam_sub, cam_short in CAMS:
    mot_cam = defaultdict(list)
    mot_file = MOT_DIR / cam_sub / "gt" / "gt_obb.txt"
    if mot_file.exists():
        with open(mot_file) as f:
            for line in f:
                if line.startswith("#"): continue
                cols = line.strip().split(",")
                if len(cols) < 10: continue
                tid = int(cols[1]); corners = list(map(float, cols[2:10]))
                mot_cam[100+int(cols[0])-1].append(
                    (tid, np.mean(corners[0::2]), np.mean(corners[1::2])))

    n = 0
    for lf in sorted(LABELS_DIR.glob(f"{cam_short}_frame_*.txt")):
        fn = int(lf.stem.split("_frame_")[1])
        entries = mot_cam.get(fn, [])
        if not entries: continue
        lines = lf.read_text().strip().splitlines(); new_lines = []
        for line in lines:
            pts = line.split()
            if len(pts) != 9: new_lines.append(line); continue
            coords = list(map(float, pts[1:]))
            cx = np.mean([coords[i]*rw for i in range(0,8,2)])
            cy = np.mean([coords[i]*rh for i in range(1,8,2)])
            bd, bt = np.inf, -1
            for tid, mx, my in entries:
                d = ((cx-mx)**2+(cy-my)**2)**0.5
                if d < bd: bd, bt = d, tid
            nc = track_class[bt] if bt in track_class and bd < 50 else int(pts[0])
            new_lines.append(f"{nc} " + " ".join(f"{v:.6f}" for v in coords)); n += 1
        lf.write_text("\n".join(new_lines))

    if mot_file.exists():
        lines = mot_file.read_text().splitlines(); new_lines = []
        for line in lines:
            if line.startswith("#"): new_lines.append(line); continue
            cols = line.split(",")
            if len(cols) < 12: new_lines.append(line); continue
            tid = int(cols[1])
            if tid in track_class: cols[11] = str(track_class[tid]+1)
            new_lines.append(",".join(cols))
        mot_file.write_text("\n".join(new_lines))

    print(f"  {cam_short}: {n} YOLO labels updated")

print("\n" + "="*60)
print(f"  Done.  Track→class: {dict(sorted(track_class.items()))}")
print("="*60)
