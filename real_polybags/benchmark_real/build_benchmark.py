#!/usr/bin/env python3
"""Build a detection + tracking benchmark from the supervisor's hand-annotated
real validation set (Single_Polybags_Iteration_1_fps_5, session 20260602_092655,
1280x720, LabelMe polygons with group_id track IDs).

Per annotated camera it writes, under this folder:
  <cam>/images/frame_XXXXXX.jpg    (copied annotated frames)
  <cam>/labels/frame_XXXXXX.txt    (YOLO detect: class 0 + aabb of the polygon)
  <cam>/gt.txt                     (MOT16 ground truth: frame,id,x,y,w,h,1,-1,-1,-1)
  data_<cam>.yaml                  (ultralytics val config)
lucid has no annotations and is skipped.
"""
import json, glob, re, shutil
from pathlib import Path

SRC = Path("/Users/awthura/OVGU/Single_Polybags_Iteration_1_fps_5")
OUT = Path("/Users/awthura/OVGU/AMS/real_polybags/benchmark_real")
CAMS = ["basler_1", "basler_2", "rgbd_1", "rgbd_2"]

def poly_to_aabb(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

summary = []
for cam in CAMS:
    js = sorted(glob.glob(str(SRC / cam / "images" / "*.json")))
    if not js:
        continue
    (OUT / cam / "images").mkdir(parents=True, exist_ok=True)
    (OUT / cam / "labels").mkdir(parents=True, exist_ok=True)
    gt_rows = []
    n_box = 0
    for jf in js:
        d = json.load(open(jf))
        W, H = d["imageWidth"], d["imageHeight"]
        stem = Path(jf).stem
        frame_no = int(re.search(r"(\d+)", stem).group(1))
        img_src = SRC / cam / "images" / f"{stem}.jpg"
        if not img_src.exists():
            continue
        shutil.copy2(img_src, OUT / cam / "images" / f"{stem}.jpg")
        yolo_lines = []
        for s in d["shapes"]:
            if s.get("label") != "polybag":
                continue
            x0, y0, x1, y1 = poly_to_aabb(s["points"])
            x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
            y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
            w = x1 - x0; h = y1 - y0
            if w <= 1 or h <= 1:
                continue
            cx = (x0 + x1) / 2 / W; cy = (y0 + y1) / 2 / H
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            gid = s.get("group_id")
            if gid is not None:
                # MOT16: frame is 1-indexed
                gt_rows.append((frame_no + 1, int(gid), x0, y0, w, h))
            n_box += 1
        (OUT / cam / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")
    # MOT ground truth
    gt_rows.sort()
    with open(OUT / cam / "gt.txt", "w") as fh:
        for f, i, x, y, w, h in gt_rows:
            fh.write(f"{f},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    # per-camera detection yaml
    (OUT / f"data_{cam}.yaml").write_text(
        f"path: {OUT/cam}\ntrain: images\nval: images\nnc: 1\nnames: ['polybag']\n")
    summary.append((cam, len(js), n_box, len(gt_rows)))

print("built benchmark (annotated frames | det boxes | MOT gt rows):")
for cam, nf, nb, ng in summary:
    print(f"  {cam:9s}: {nf:4d} frames, {nb:4d} boxes, {ng:4d} gt rows")
print(f"\nwritten under {OUT}")
