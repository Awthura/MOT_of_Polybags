"""
Fallback: classify tracks 1, 6, 11 that were too clustered for the first pass.
Uses a relaxed 3-D isolation threshold (0.05 units instead of 0.12).
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
import bpy, bpy_extras.object_utils
from mathutils import Vector
import numpy as np, cv2
from scipy.optimize import linear_sum_assignment

BASE       = Path("/Users/awthura/OVGU/AMS")
STL_FOLDER = BASE / "superquadrics_stl_files_100_2000_frames"
IMAGES_DIR = BASE / "synth_dataset" / "images"
LABELS_DIR = BASE / "synth_dataset" / "labels"
MOT_DIR    = BASE / "synth_dataset" / "mot_obb"
OUT_CSV    = BASE / "synth_dataset" / "track_classes.csv"

CLASS_NAMES = ["pink_polybag","blue_polybag","yellow_polybag",
               "grey_polybag","green_polybag","red_polybag"]
COLOUR_RULES = [
    (3,  0,180,  0, 30, 80,255),
    (5,  0, 12, 40,255, 80,255),
    (5,165,180, 40,255, 80,255),
    (2, 13, 40, 30,255, 80,255),
    (4, 41,100, 25,255, 50,255),
    (1,101,138, 25,255, 50,255),
    (0,139,167, 25,255, 50,255),
]
def hsv_to_class(h,s,v):
    for cid,h0,h1,s0,s1,v0,v1 in COLOUR_RULES:
        if h0<=h<=h1 and s0<=s<=s1 and v0<=v<=v1: return cid
    return 3

MAX_MATCH_DIST=0.30; MIN_ISO=0.05; PATCH=6

# Read existing classifications
existing = {}
with open(OUT_CSV) as f:
    for row in csv.DictReader(f):
        existing[int(row["track_id"])] = int(row["class_id"])

MISSING = [tid for tid in [1,6,11] if tid not in existing]
print(f"Tracks to classify with relaxed threshold: {MISSING}")
if not MISSING:
    print("All tracks already classified."); sys.exit(0)

_mat_cache={}
def _get_mat(i):
    n=f"mat_fb_{i:03d}"
    if n in _mat_cache: return _mat_cache[n]
    COLS=[(0.2,0,0,1),(0,0.2,0,1),(0,0,0.2,1),(0.2,0.2,0,1),
          (0.2,0,0.2,1),(0,0.2,0.2,1),(0.1,0.1,0.1,1),(0.15,0.1,0,1)]
    m=bpy.data.materials.new(n); m.use_nodes=True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value=COLS[i%len(COLS)]
    _mat_cache[n]=m; return m

def clean():
    for o in list(bpy.data.objects):
        if o.type=="MESH" and o.name.startswith("part_fb_"):
            bpy.data.objects.remove(o,do_unlink=True)

def stl(frame):
    for p in [f"ExtractSurface1_frame_{frame:04d}.stl",
              f"dump_plane1stl_frame_{frame:04d}.stl",
              f"Triangulate1_frame_{frame:04d}.stl"]:
        q=STL_FOLDER/p
        if q.exists(): return q
    return STL_FOLDER/f"ExtractSurface1_frame_{frame:04d}.stl"

def import_p(frame):
    path=stl(frame)
    if not path.exists(): return []
    try: bpy.ops.wm.stl_import(filepath=str(path))
    except: bpy.ops.import_mesh.stl(filepath=str(path))
    imp=bpy.context.selected_objects[0]; imp.name=f"STL_fb_{frame}"
    bpy.ops.object.select_all(action="DESELECT")
    imp.select_set(True); bpy.context.view_layer.objects.active=imp
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(imp,do_unlink=True)
    res=[]
    for i,o in enumerate(bpy.context.selected_objects):
        o.name=f"part_fb_{frame}_{i:03d}"
        m=_get_mat(i)
        if o.data.materials: o.data.materials[0]=m
        else: o.data.materials.append(m)
        mw=o.matrix_world
        vw=[mw@v.co for v in o.data.vertices]
        c=np.mean([[v.x,v.y,v.z] for v in vw],axis=0) if vw else np.zeros(3)
        res.append((o,i,c))
    return res

_nid=1; _pc=None; _pi=None
def assign(cents):
    global _nid,_pc,_pi
    n=len(cents)
    if n==0: return np.array([],dtype=int)
    curr=np.array(cents)
    if _pc is None or len(_pc)==0:
        ids=np.arange(_nid,_nid+n,dtype=int); _nid+=n; _pc,_pi=curr,ids; return ids
    diff=curr[:,None,:]-_pc[None,:,:]; cost=np.sqrt((diff**2).sum(axis=2))
    ri,ci=linear_sum_assignment(cost)
    ids=np.full(n,-1,dtype=int)
    for r,c in zip(ri,ci):
        if cost[r,c]<=MAX_MATCH_DIST: ids[r]=_pi[c]
    for i in range(n):
        if ids[i]==-1: ids[i]=_nid; _nid+=1
    _pc,_pi=curr,ids; return ids

scene=bpy.context.scene; render=scene.render
rw,rh=render.resolution_x,render.resolution_y
CAMS=[("Cam_Front","cam_01_front","front"),("Cam_Back","cam_02_back","back"),
      ("Cam_Left","cam_03_left","left"),("Cam_Right","cam_04_right","right")]
cameras={n:bpy.data.objects.get(n) for n,_,_ in CAMS}
cameras={k:v for k,v in cameras.items() if v}

colour_obs=defaultdict(list)
for frame in range(100,1973):
    scene.frame_set(frame); bpy.context.view_layer.update()
    clean(); parts=import_p(frame)
    if not parts: continue
    cents=[c for _,_,c in parts]; ids=assign(cents)
    target=[i for i,tid in enumerate(ids) if int(tid) in MISSING]
    if not target: continue
    n=len(cents); iso=np.full(n,np.inf)
    if n>1:
        arr=np.array(cents)
        for i in range(n):
            d=np.linalg.norm(arr-arr[i],axis=1); d[i]=np.inf; iso[i]=d.min()
    for cn,cs,cam_short in CAMS:
        cam=cameras.get(cn)
        if not cam: continue
        img_path=IMAGES_DIR/f"{cam_short}_frame_{frame:04d}.png"
        if not img_path.exists(): continue
        img=cv2.imread(str(img_path))
        if img is None: continue
        for i in target:
            if iso[i]<MIN_ISO: continue
            wp=Vector(cents[i])
            co=bpy_extras.object_utils.world_to_camera_view(scene,cam,wp)
            if co.z<0: continue
            cx,cy=int(co.x*rw),int((1-co.y)*rh)
            if not(0<=cx<rw and 0<=cy<rh): continue
            x1=max(0,cx-PATCH); x2=min(rw,cx+PATCH+1)
            y1=max(0,cy-PATCH); y2=min(rh,cy+PATCH+1)
            patch=img[y1:y2,x1:x2]
            if patch.size==0: continue
            hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
            med=np.median(hsv.reshape(-1,3),axis=0)
            if float(med[2])<60: continue
            colour_obs[int(ids[i])].append(tuple(med))

print("\nFallback results:")
new_classes={}
for tid in MISSING:
    obs=colour_obs.get(tid,[])
    if not obs:
        print(f"  track {tid}: 0 samples → grey (fallback)")
        new_classes[tid]=3
    else:
        arr=np.array(obs)
        h,s,v=np.median(arr[:,0]),np.median(arr[:,1]),np.median(arr[:,2])
        cid=hsv_to_class(h,s,v)
        new_classes[tid]=cid
        print(f"  track {tid}: {len(obs)} samples H={h:.0f} S={s:.0f} V={v:.0f} → class {cid} ({CLASS_NAMES[cid]})")

with open(OUT_CSV,"a",newline="") as f:
    w=csv.writer(f)
    for tid,cid in new_classes.items():
        obs=colour_obs.get(tid,[])
        arr=np.array(obs) if obs else np.zeros((1,3))
        w.writerow([tid,cid,CLASS_NAMES[cid],len(obs),
                    round(float(np.median(arr[:,0])),1),
                    round(float(np.median(arr[:,1])),1),
                    round(float(np.median(arr[:,2])),1)])

all_classes={**existing,**new_classes}
print(f"\nFull mapping: {dict(sorted(all_classes.items()))}")

# Apply to all cameras
for cn,cam_sub,cam_short in CAMS:
    mot_cam=defaultdict(list)
    mot_file=MOT_DIR/cam_sub/"gt"/"gt_obb.txt"
    if mot_file.exists():
        with open(mot_file) as f:
            for line in f:
                if line.startswith("#"): continue
                cols=line.strip().split(",")
                if len(cols)<10: continue
                tid=int(cols[1]); corners=list(map(float,cols[2:10]))
                mot_cam[100+int(cols[0])-1].append((tid,np.mean(corners[0::2]),np.mean(corners[1::2])))

    n=0
    for lf in sorted(LABELS_DIR.glob(f"{cam_short}_frame_*.txt")):
        fn=int(lf.stem.split("_frame_")[1])
        entries=mot_cam.get(fn,[])
        if not entries: continue
        lines=lf.read_text().strip().splitlines(); new_lines=[]
        for line in lines:
            parts=line.split()
            if len(parts)!=9: new_lines.append(line); continue
            coords=list(map(float,parts[1:]))
            cx=np.mean([coords[i]*rw for i in range(0,8,2)])
            cy=np.mean([coords[i]*rh for i in range(1,8,2)])
            bd,bt=np.inf,-1
            for tid,mx,my in entries:
                d=((cx-mx)**2+(cy-my)**2)**0.5
                if d<bd: bd,bt=d,tid
            nc=all_classes[bt] if bt in all_classes and bd<50 else int(parts[0])
            new_lines.append(f"{nc} "+" ".join(f"{v:.6f}" for v in coords)); n+=1
        lf.write_text("\n".join(new_lines))

    if mot_file.exists():
        lines=mot_file.read_text().splitlines(); new_lines=[]
        for line in lines:
            if line.startswith("#"): new_lines.append(line); continue
            cols=line.split(",")
            if len(cols)<12: new_lines.append(line); continue
            tid=int(cols[1])
            if tid in all_classes: cols[11]=str(all_classes[tid]+1)
            new_lines.append(",".join(cols))
        mot_file.write_text("\n".join(new_lines))
    print(f"  {cam_short}: {n} labels updated")

print("\nDone.")
