#!/usr/bin/env bash
# render_500_mcmot.sh
#
# Renders frames 100-599 (500 consecutive) for all 4 cameras into a fresh
# synth_dataset_mcmot/ directory. Splits into 4 parallel Blender instances
# (~67 min total at 8 s/frame on this MacBook).
#
# After rendering, runs:
#   relabel_synth.py      -- HSV colour correction on all YOLO labels
#   blender_mot_annotate  -- OBB-MOT ground truth for all 500 frames
#   fix_mot_classes.py    -- apply confirmed track->class mapping to MOT files
#
# Usage:
#   chmod +x render_500_mcmot.sh
#   ./render_500_mcmot.sh

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
BLEND="$BASE/convert_stl_to_animation_multi_camera.blend"
ANNOTATE="$BASE/blender_annotate.py"
MOT_ANNOTATE="$BASE/blender_mot_annotate.py"
FIX_MOT="$BASE/fix_mot_classes.py"
RELABEL="$BASE/relabel_synth.py"

OUT_DIR="$BASE/synth_dataset_mcmot"
LOG_DIR="$BASE/render_logs_mcmot"

START=100
END=599
N=4   # parallel instances

mkdir -p "$LOG_DIR" "$OUT_DIR"

# ── Split 100-599 into N batches ───────────────────────────────────────────────
TOTAL=$(( END - START + 1 ))
BATCH=$(( (TOTAL + N - 1) / N ))

echo "[1/4] Splitting frames $START-$END into $N batches of ~$BATCH frames each"

BATCH_FILES=()
for i in $(seq 0 $(( N - 1 ))); do
    BS=$(( START + i * BATCH ))
    BE=$(( BS + BATCH - 1 ))
    [ "$BE" -gt "$END" ] && BE=$END
    [ "$BS" -gt "$END" ] && break
    FILE="$LOG_DIR/batch_mcmot_$(printf '%02d' $i).txt"
    seq "$BS" "$BE" > "$FILE"
    BATCH_FILES+=("$FILE")
    echo "   Batch $i: frames $BS-$BE ($(wc -l < "$FILE" | tr -d ' ') frames)"
done

# ── Launch parallel Blender instances ─────────────────────────────────────────
echo ""
echo "[2/4] Launching ${#BATCH_FILES[@]} Blender instances..."
echo "      Output -> $OUT_DIR"
echo "      Logs   -> $LOG_DIR/render_mcmot_*.log"
echo "      Estimated time: ~$(( BATCH * 32 / 60 )) min (4 cams x 8 s/frame per batch)"
echo ""

START_TS=$(date +%s)
PIDS=()

for i in "${!BATCH_FILES[@]}"; do
    LOG="$LOG_DIR/render_mcmot_$(printf '%02d' $i).log"
    echo "   Starting batch $i (PID will be shown) -> $(basename "${BATCH_FILES[$i]}")"
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
echo "   Waiting for all instances to finish..."

ALL_OK=true
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "   Batch $i done."
    else
        echo "   Batch $i FAILED. Check $LOG_DIR/render_mcmot_$(printf '%02d' $i).log"
        ALL_OK=false
    fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "   Render complete: ${ELAPSED}s (~$(( ELAPSED / 60 )) min)"

if [ "$ALL_OK" = false ]; then
    echo "   WARNING: some batches failed. Re-running the script will skip already-rendered frames."
    exit 1
fi

# ── Flatten per-camera outputs into synth_dataset_mcmot/images + labels ───────
echo ""
echo "[3/4] Flattening images and labels..."

python3 - <<'PYEOF'
import shutil
from pathlib import Path

BASE    = Path(__file__).resolve().parent if "__file__" in dir() else Path("/Users/awthura/OVGU/AMS")
BASE    = Path("/Users/awthura/OVGU/AMS")
OUT     = BASE / "synth_dataset_mcmot"
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

imgs = lbls = 0
for cam_sub, cam_short in CAM_MAP.items():
    for img in sorted((OUT / cam_sub / "images").glob("frame_*.png")):
        frm = int(img.stem.split("_")[1])
        dst = IMG_OUT / f"{cam_short}_frame_{frm:04d}.png"
        shutil.copy2(img, dst)
        imgs += 1
    for lbl in sorted((OUT / cam_sub / "labels").glob("frame_*.txt")):
        frm = int(lbl.stem.split("_")[1])
        dst = LBL_OUT / f"{cam_short}_frame_{frm:04d}.txt"
        shutil.copy2(lbl, dst)
        lbls += 1

print(f"   {imgs} images, {lbls} labels -> {OUT}")
PYEOF

# ── Colour-correct YOLO labels ─────────────────────────────────────────────────
echo ""
echo "   Running relabel_synth.py (HSV colour correction)..."
# Temporarily point LABELS_DIR and IMAGES_DIR at the mcmot dataset
python3 -c "
import sys, importlib.util, types
from pathlib import Path

# Patch paths before importing
import relabel_synth
relabel_synth.LABELS_DIR  = Path('/Users/awthura/OVGU/AMS/synth_dataset_mcmot/labels')
relabel_synth.IMAGES_DIR  = Path('/Users/awthura/OVGU/AMS/synth_dataset_mcmot/images')
relabel_synth.CLASSES_TXT = Path('/Users/awthura/OVGU/AMS/synth_dataset_mcmot/classes.txt')
relabel_synth.main()
" 2>/dev/null || python3 "$RELABEL"

# Copy classes.txt
cp "$BASE/synth_dataset/classes.txt" "$OUT_DIR/classes.txt" 2>/dev/null || true

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
python3 -c "
from pathlib import Path
BASE = Path('/Users/awthura/OVGU/AMS')
MOT_DIR = Path('/Users/awthura/OVGU/AMS/synth_dataset_mcmot/mot_obb')

# Confirmed track->class mapping (1-based class_id for MOT files)
track_to_class = {1:6, 2:5, 3:2, 4:3, 5:1, 6:7, 7:3, 8:4, 9:3, 10:1, 11:7}

for cam_sub in ['cam_01_front','cam_02_back','cam_03_left','cam_04_right']:
    for fname, col in [('gt_obb.txt',11),('gt.txt',7)]:
        p = MOT_DIR / cam_sub / 'gt' / fname
        if not p.exists(): continue
        lines = p.read_text().splitlines()
        out = []
        changed = 0
        for line in lines:
            if line.startswith('#') or not line.strip():
                out.append(line); continue
            parts = line.split(',')
            if len(parts) > col:
                tid = int(parts[1])
                new = track_to_class.get(tid, int(parts[col]))
                if new != int(parts[col]):
                    parts[col] = str(new); changed += 1
            out.append(','.join(parts))
        p.write_text('\n'.join(out))
        print(f'   {cam_sub}/{fname}: {changed} class_id fixes')
"

END_TS=$(date +%s)
TOTAL_TIME=$(( END_TS - START_TS ))
echo ""
echo "======================================================"
echo "  Done in $(( TOTAL_TIME / 60 )) min $(( TOTAL_TIME % 60 )) s"
echo "  Dataset: $OUT_DIR"
echo "    images/   $(ls "$OUT_DIR/images" | wc -l | tr -d ' ') PNGs"
echo "    labels/   $(ls "$OUT_DIR/labels" | wc -l | tr -d ' ') YOLO OBB txts"
echo "    mot_obb/  OBB-MOT ground truth (4 cameras)"
echo "======================================================"
