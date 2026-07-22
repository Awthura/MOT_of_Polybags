"""
build_synth_dataset.py
──────────────────────────────────────────────────────────────────────────────
Run OUTSIDE Blender (normal Python).

1. Scans all existing render folders for valid frame_XXXX.png images.
2. Symlinks them into a flat  synth_dataset/images/  folder, naming each
   file  {cam}_{frame}.png  to avoid cross-camera collisions.
3. Calls Blender (subprocess) to generate the matching YOLO OBB labels.

Usage:
    python3 build_synth_dataset.py [--out_dir ./synth_dataset] [--dry_run]
"""

import os, sys, re, subprocess, argparse
from pathlib import Path

BASE    = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
OUT_DIR = BASE / "synth_dataset"

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
BLEND   = BASE / "convert_stl_to_animation_multi_camera.blend"
ANNOTATE_SCRIPT = BASE / "blender_annotate.py"

# Render source folders (in priority order — first match for a given cam+frame wins)
RENDER_ROOTS = [
    BASE / "superquadrics_render_100_768_frames_multi_camera",
    BASE / "superquadrics_render_769_2000_frames_multi_camera",
]

# Camera subfolder name → short prefix used in flat filenames
CAM_PREFIX = {
    "cam_01_front": "front",
    "cam_02_back":  "back",
    "cam_03_left":  "left",
    "cam_04_right": "right",
}

FRAME_RE = re.compile(r"frame_(\d{4})\.png$")

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",  default=str(OUT_DIR))
parser.add_argument("--dry_run",  action="store_true")
args = parser.parse_args()

OUT_DIR   = Path(args.out_dir)
IMG_DIR   = OUT_DIR / "images"
LBL_DIR   = OUT_DIR / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: collect valid images ───────────────────────────────────────────────
print("Scanning existing render folders …")
seen = {}          # (cam_short, frame_num) → source path   (first-wins)
frame_set = set()  # all unique frame numbers found

for root in RENDER_ROOTS:
    if not root.exists():
        continue
    for cam_sub, cam_short in CAM_PREFIX.items():
        cam_dir = root / cam_sub
        if not cam_dir.exists():
            continue
        for f in sorted(cam_dir.glob("frame_*.png")):
            m = FRAME_RE.search(f.name)
            if not m:
                continue
            frame_num = int(m.group(1))
            key = (cam_short, frame_num)
            if key not in seen:
                seen[key] = f
                frame_set.add(frame_num)

print(f"  Found {len(seen)} valid images across {len(frame_set)} unique frames")
print(f"  Frame range: {min(frame_set)} – {max(frame_set)}")
print(f"  Output dir : {OUT_DIR}\n")

if args.dry_run:
    for (cam, frm), src in list(seen.items())[:10]:
        print(f"  {cam}_frame_{frm:04d}.png  ←  {src}")
    print("  … dry run, exiting.")
    sys.exit(0)

# ── Step 2: symlink images into flat folder ────────────────────────────────────
print("Symlinking images …")
for (cam_short, frame_num), src in seen.items():
    dst = IMG_DIR / f"{cam_short}_frame_{frame_num:04d}.png"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())

print(f"  {len(seen)} symlinks → {IMG_DIR}\n")

# ── Step 3: generate labels via Blender ───────────────────────────────────────
# Group frames by contiguous range to make fewer Blender calls
sorted_frames = sorted(frame_set)
print(f"Generating labels for {len(sorted_frames)} frames via Blender …")
print("  (This may take a few minutes — no rendering, just projection)\n")

# Find contiguous ranges
ranges = []
start = prev = sorted_frames[0]
for f in sorted_frames[1:]:
    if f == prev + 1:
        prev = f
    else:
        ranges.append((start, prev))
        start = prev = f
ranges.append((start, prev))

print(f"  Frame ranges: {ranges[:5]}{'…' if len(ranges)>5 else ''}")

for rng_start, rng_end in ranges:
    cmd = [
        BLENDER, str(BLEND),
        "--background",
        "--python", str(ANNOTATE_SCRIPT),
        "--",
        "--frames", f"{rng_start}-{rng_end}",
        "--out_dir", str(OUT_DIR),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Print only meaningful lines
    for line in result.stdout.splitlines():
        if any(x in line for x in ["Frame", "Done.", "ERROR", "SKIP", "Warning"]):
            print(f"  {line.strip()}")
    if result.returncode != 0:
        print(f"  ERROR on frames {rng_start}-{rng_end}")
        print(result.stderr[-500:])

# ── Step 4: move labels into flat structure ────────────────────────────────────
# blender_annotate.py writes to out_dir/{cam_subfolder}/labels/frame_XXXX.txt
# We want out_dir/labels/{cam_short}_frame_XXXX.txt
print("\nFlattening label files …")
moved = 0
cam_map = {v: k for k, v in CAM_PREFIX.items()}   # short → subfolder

for cam_short, cam_sub in [("front","cam_01_front"),("back","cam_02_back"),
                             ("left","cam_03_left"),("right","cam_04_right")]:
    src_lbl_dir = OUT_DIR / cam_sub / "labels"
    if not src_lbl_dir.exists():
        continue
    for lbl in src_lbl_dir.glob("frame_*.txt"):
        m = re.search(r"frame_(\d{4})\.txt$", lbl.name)
        if not m:
            continue
        frame_num = int(m.group(1))
        # Only keep label if we have a matching image
        if (cam_short, frame_num) not in seen:
            continue
        dst = LBL_DIR / f"{cam_short}_frame_{frame_num:04d}.txt"
        dst.write_text(lbl.read_text())
        moved += 1

print(f"  {moved} label files → {LBL_DIR}")

# ── Step 5: write classes.txt ──────────────────────────────────────────────────
classes = [
    "pink_polybag", "blue_polybag", "yellow_polybag",
    "grey_polybag", "green_polybag", "red_polybag",
]
(OUT_DIR / "classes.txt").write_text("\n".join(classes))

# ── Summary ────────────────────────────────────────────────────────────────────
img_count = len(list(IMG_DIR.glob("*.png")))
lbl_count = len(list(LBL_DIR.glob("*.txt")))
print(f"\n{'='*55}")
print(f"  synth_dataset/")
print(f"    images/    {img_count} PNGs  (symlinked)")
print(f"    labels/    {lbl_count} YOLO OBB .txt files")
print(f"    classes.txt")
print(f"{'='*55}")
