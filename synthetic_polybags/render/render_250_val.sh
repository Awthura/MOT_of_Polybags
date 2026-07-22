#!/usr/bin/env bash
# render_250_val.sh
#
# Renders frames 1000-1250 (251 frames) for all 4 cameras into a fresh
# synth_dataset_val/ directory.  Splits into 4 parallel Blender instances.
#
# After rendering, runs:
#   relabel_synth.py      -- HSV colour correction on all YOLO labels
#   blender_mot_annotate  -- OBB-MOT ground truth for all 251 frames
#   fix_mot_classes.py    -- apply confirmed track->class mapping to MOT files
#
# Usage:
#   chmod +x render_250_val.sh
#   ./render_250_val.sh

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
BLEND="$BASE/convert_stl_to_animation_multi_camera.blend"
ANNOTATE="$BASE/blender_annotate.py"
MOT_ANNOTATE="$BASE/blender_mot_annotate.py"
RELABEL="$BASE/relabel_synth.py"

OUT_DIR="$BASE/synth_dataset_val"
LOG_DIR="$BASE/render_logs_val"

START=1000
END=1250
N=4   # parallel instances
TOTAL_IMGS=$(( (END - START + 1) * 4 ))   # 251 frames x 4 cameras

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── Split 1000-1250 into N batches ────────────────────────────────────────────
TOTAL=$(( END - START + 1 ))
BATCH=$(( (TOTAL + N - 1) / N ))

echo "[1/4] Splitting frames $START-$END into $N batches of ~$BATCH frames each"

BATCH_FILES=()
for i in $(seq 0 $(( N - 1 ))); do
    BS=$(( START + i * BATCH ))
    BE=$(( BS + BATCH - 1 ))
    [ "$BE" -gt "$END" ] && BE=$END
    [ "$BS" -gt "$END" ] && break
    FILE="$LOG_DIR/batch_val_$(printf '%02d' $i).txt"
    seq "$BS" "$BE" > "$FILE"
    BATCH_FILES+=("$FILE")
    echo "   Batch $i: frames $BS-$BE ($(wc -l < "$FILE" | tr -d ' ') frames)"
done

# ── Launch parallel Blender instances ─────────────────────────────────────────
echo ""
echo "[2/4] Launching ${#BATCH_FILES[@]} Blender instances..."
echo "      Output -> $OUT_DIR"
echo "      Logs   -> $LOG_DIR/render_val_*.log"
echo ""

START_TS=$(date +%s)
PIDS=()

for i in "${!BATCH_FILES[@]}"; do
    LOG="$LOG_DIR/render_val_$(printf '%02d' $i).log"
    echo "   Starting batch $i -> $(basename "${BATCH_FILES[$i]}")"
    "$BLENDER" "$BLEND" \
        --background \
        --python "$ANNOTATE" \
        -- \
        --frame_list "${BATCH_FILES[$i]}" \
        --render \
        --out_dir "$OUT_DIR" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "   PID: ${PIDS[$i]}"
done

echo ""
echo "   Monitoring render progress (updates every 10 s)..."

# tqdm progress monitor: polls the output directory while Blender runs
python3 - "$OUT_DIR" "$TOTAL_IMGS" <<'PYEOF' &
import sys, time, pathlib
out_dir = pathlib.Path(sys.argv[1])
total   = int(sys.argv[2])
try:
    from tqdm import tqdm
    pbar = tqdm(total=total, desc="  Rendering", unit="img", ncols=72, dynamic_ncols=False)
    prev = 0
    while True:
        curr = sum(1 for _ in out_dir.rglob("frame_*.png"))
        if curr > prev:
            pbar.update(curr - prev)
            prev = curr
        if prev >= total:
            break
        time.sleep(10)
    pbar.close()
except ImportError:
    while True:
        curr = sum(1 for _ in out_dir.rglob("frame_*.png"))
        print(f"\r  Rendered {curr}/{total} images", end="", flush=True)
        if curr >= total:
            break
        time.sleep(10)
    print()
PYEOF
MONITOR_PID=$!

ALL_OK=true
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "   Batch $i done."
    else
        echo "   Batch $i FAILED. Check $LOG_DIR/render_val_$(printf '%02d' $i).log"
        ALL_OK=false
    fi
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "   Render complete: ${ELAPSED}s (~$(( ELAPSED / 60 )) min)"

if [ "$ALL_OK" = false ]; then
    echo "   WARNING: some batches failed. Re-running the script will skip already-rendered frames."
    exit 1
fi

# ── Flatten per-camera outputs into synth_dataset_val/images + labels ─────────
echo ""
echo "[3/4] Flattening images and labels..."

python3 - <<'PYEOF'
import shutil
from pathlib import Path
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it

BASE    = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
OUT     = BASE / "synth_dataset_val"
IMG_OUT = OUT / "images"
LBL_OUT = OUT / "labels"
IMG_OUT.mkdir(exist_ok=True)
LBL_OUT.mkdir(exist_ok=True)

CAM_MAP = {
    "cam_01_front": "front",
    "cam_02_back":  "back",
    "cam_03_left":  "left",
    "cam_04_right": "right",
}

pairs = []
for cam_sub, cam_short in CAM_MAP.items():
    for img in sorted((OUT / cam_sub / "images").glob("frame_*.png")):
        frm = int(img.stem.split("_")[1])
        lbl = OUT / cam_sub / "labels" / f"frame_{frm:04d}.txt"
        pairs.append((img, lbl, cam_short, frm))

imgs = lbls = 0
for img, lbl, cam_short, frm in _tqdm(pairs, desc="  Flattening", unit="frame", ncols=72):
    shutil.copy2(img, IMG_OUT / f"{cam_short}_frame_{frm:04d}.png")
    imgs += 1
    if lbl.exists():
        shutil.copy2(lbl, LBL_OUT / f"{cam_short}_frame_{frm:04d}.txt")
        lbls += 1

print(f"   {imgs} images, {lbls} labels -> {OUT}")
PYEOF

# ── Colour-correct YOLO labels ─────────────────────────────────────────────────
echo ""
echo "   Running relabel_synth.py (HSV colour correction)..."

python3 - <<'PYEOF'
import sys
sys.path.insert(0, "/Users/awthura/OVGU/AMS/synthetic_polybags")
from pathlib import Path
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it

import relabel_synth
relabel_synth.LABELS_DIR  = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset_val/labels")
relabel_synth.IMAGES_DIR  = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset_val/images")
relabel_synth.CLASSES_TXT = Path("/Users/awthura/OVGU/AMS/synthetic_polybags/synth_dataset_val/classes.txt")
relabel_synth.main()
PYEOF

# Copy classes.txt
printf 'pink_polybag\nblue_polybag\nyellow_polybag\ngrey_polybag\ngreen_polybag\nred_polybag\nteal_polybag\n' > "$OUT_DIR/classes.txt"

# ── OBB-MOT ground truth ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Generating OBB-MOT ground truth (frames $START-$END)..."
"$BLENDER" "$BLEND" \
    --background \
    --python "$MOT_ANNOTATE" \
    -- \
    --frames "$START-$END" \
    --out_dir "$OUT_DIR" \
    >> "$LOG_DIR/mot_annotate.log" 2>&1
echo "   MOT annotation done."

echo ""
echo "   Applying confirmed track->class mapping..."
python3 - <<'PYEOF'
from pathlib import Path
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw): return it

# Confirmed track->class mapping (1-based class_id for MOT files)
# track:  1=red  2=green  3=blue  4=yellow  5=pink  6=teal  7=yellow  8=grey  9=yellow  10=pink  11=teal
track_to_class = {1:6, 2:5, 3:2, 4:3, 5:1, 6:7, 7:3, 8:4, 9:3, 10:1, 11:7}

BASE    = Path("/Users/awthura/OVGU/AMS/synthetic_polybags")
MOT_DIR = BASE / "synth_dataset_val" / "mot_obb"

targets = [
    ("cam_01_front", "gt_obb.txt", 11),
    ("cam_01_front", "gt.txt",      7),
    ("cam_02_back",  "gt_obb.txt", 11),
    ("cam_02_back",  "gt.txt",      7),
    ("cam_03_left",  "gt_obb.txt", 11),
    ("cam_03_left",  "gt.txt",      7),
    ("cam_04_right", "gt_obb.txt", 11),
    ("cam_04_right", "gt.txt",      7),
]

for cam_sub, fname, col in _tqdm(targets, desc="  Fix classes", unit="file", ncols=72):
    p = MOT_DIR / cam_sub / "gt" / fname
    if not p.exists():
        continue
    lines = p.read_text().splitlines()
    out, changed = [], 0
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line); continue
        parts = line.split(",")
        if len(parts) > col:
            tid = int(parts[1])
            new = track_to_class.get(tid, int(parts[col]))
            if new != int(parts[col]):
                parts[col] = str(new); changed += 1
        out.append(",".join(parts))
    p.write_text("\n".join(out))
    print(f"   {cam_sub}/{fname}: {changed} class_id fixes")
PYEOF

END_TS=$(date +%s)
TOTAL_TIME=$(( END_TS - START_TS ))
echo ""
echo "======================================================"
echo "  Val dataset done in $(( TOTAL_TIME / 60 )) min $(( TOTAL_TIME % 60 )) s"
echo "  Dataset: $OUT_DIR"
echo "    images/   $(ls "$OUT_DIR/images" | wc -l | tr -d ' ') PNGs"
echo "    labels/   $(ls "$OUT_DIR/labels" | wc -l | tr -d ' ') YOLO OBB txts"
echo "    mot_obb/  OBB-MOT ground truth (4 cameras)"
echo "======================================================"
