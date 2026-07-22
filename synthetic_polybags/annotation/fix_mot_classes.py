#!/usr/bin/env python3
"""
fix_mot_classes.py
Run OUTSIDE Blender, after blender_mot_annotate.py finishes.

blender_mot_annotate.py assigns class_id from the raw material index
(which is arbitrary and wrong). This script replaces every class_id field
in gt_obb.txt and gt.txt with the visually confirmed track→class mapping
from track_classes.csv (class_id is 1-based in MOT files).
"""

import csv
from pathlib import Path

BASE    = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
MOT_DIR = BASE / "synth_dataset" / "mot_obb"
CSV     = BASE / "synth_dataset" / "track_classes.csv"

# Load track_id → 1-based class_id from track_classes.csv
track_to_class = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        tid = int(row["track_id"])
        cid = int(row["class_id"]) + 1   # 0-based → 1-based
        track_to_class[tid] = cid

print(f"Track→class mapping (1-based): {track_to_class}")

CAM_SUBDIRS = ["cam_01_front", "cam_02_back", "cam_03_left", "cam_04_right"]

for cam_sub in CAM_SUBDIRS:
    gt_dir = MOT_DIR / cam_sub / "gt"

    # ── gt_obb.txt: frame,id,x1,y1,x2,y2,x3,y3,x4,y4,conf,class_id,vis,cx,cy,cz
    obb_path = gt_dir / "gt_obb.txt"
    if obb_path.exists():
        lines_in  = obb_path.read_text().splitlines()
        lines_out = []
        changed   = 0
        for line in lines_in:
            if line.startswith("#") or not line.strip():
                lines_out.append(line)
                continue
            parts = line.split(",")
            if len(parts) >= 12:
                track_id = int(parts[1])
                old_cid  = int(parts[11])
                new_cid  = track_to_class.get(track_id, old_cid)
                if new_cid != old_cid:
                    parts[11] = str(new_cid)
                    changed += 1
            lines_out.append(",".join(parts))
        obb_path.write_text("\n".join(lines_out))
        print(f"  {cam_sub}/gt/gt_obb.txt : {len(lines_in)-1} rows, {changed} class_id fixes")

    # ── gt.txt: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class_id,vis
    gt_path = gt_dir / "gt.txt"
    if gt_path.exists():
        lines_in  = gt_path.read_text().splitlines()
        lines_out = []
        changed   = 0
        for line in lines_in:
            if line.startswith("#") or not line.strip():
                lines_out.append(line)
                continue
            parts = line.split(",")
            if len(parts) >= 8:
                track_id = int(parts[1])
                old_cid  = int(parts[7])
                new_cid  = track_to_class.get(track_id, old_cid)
                if new_cid != old_cid:
                    parts[7] = str(new_cid)
                    changed += 1
            lines_out.append(",".join(parts))
        gt_path.write_text("\n".join(lines_out))
        print(f"  {cam_sub}/gt/gt.txt     : {len(lines_in)-1} rows, {changed} class_id fixes")

print("\nDone. All MOT class_ids now match visual ground truth.")
