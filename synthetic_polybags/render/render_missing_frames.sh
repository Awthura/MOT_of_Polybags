#!/usr/bin/env bash
# render_missing_frames.sh
#
# Renders all frames missing from any camera, then runs the full annotation
# post-processing pipeline. Run this overnight from the AMS directory.
#
# Usage:
#   chmod +x render_missing_frames.sh
#   ./render_missing_frames.sh
#
# After it finishes, all cameras will have complete coverage for the same
# frame set, and synth_dataset/ will be updated with new images, YOLO
# labels (colour-corrected), MOT annotations, and overlays.

set -euo pipefail

BASE="$(cd "$(dirname "$0")" && pwd)"
BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
BLEND="$BASE/convert_stl_to_animation_multi_camera.blend"
ANNOTATE_SCRIPT="$BASE/blender_annotate.py"
POST_SCRIPT="$BASE/postprocess_new_frames.py"
OUT_DIR="$BASE/synth_dataset"
LOG_DIR="$BASE/render_logs"
TMPDIR_BATCHES="$BASE/render_batches"

N_PARALLEL=4   # Number of parallel Blender instances (one per CPU core)

mkdir -p "$LOG_DIR" "$TMPDIR_BATCHES"

# ── Step 1: compute union of all missing frames ────────────────────────────────
echo "[1/4] Computing union of missing frames..."
ALL_MISSING="$BASE/all_missing_frames.txt"
cat "$BASE/missing_frames_front.txt" \
    "$BASE/missing_frames_back.txt" \
    "$BASE/missing_frames_left.txt" \
    "$BASE/missing_frames_right.txt" \
    | sort -nu > "$ALL_MISSING"

TOTAL=$(wc -l < "$ALL_MISSING" | tr -d ' ')
echo "      $TOTAL unique frames to render"

if [ "$TOTAL" -eq 0 ]; then
    echo "      Nothing to render — all cameras already complete."
    exit 0
fi

# ── Step 2: split into N_PARALLEL batches ─────────────────────────────────────
echo "[2/4] Splitting into $N_PARALLEL batches..."
rm -f "$TMPDIR_BATCHES"/batch_*
BATCH_SIZE=$(( (TOTAL + N_PARALLEL - 1) / N_PARALLEL ))
split -l "$BATCH_SIZE" "$ALL_MISSING" "$TMPDIR_BATCHES/batch_"

BATCHES=("$TMPDIR_BATCHES"/batch_*)
echo "      Batches: ${#BATCHES[@]} (up to $BATCH_SIZE frames each)"
for b in "${BATCHES[@]}"; do
    echo "        $(basename "$b"): $(wc -l < "$b" | tr -d ' ') frames"
done

# ── Step 3: launch parallel Blender instances ──────────────────────────────────
echo "[3/4] Launching $N_PARALLEL Blender instances in parallel..."
echo "      Logs: $LOG_DIR/render_batch_*.log"
echo "      (Each instance renders all 4 cameras for its frame batch)"
echo ""

START_TS=$(date +%s)
PIDS=()

for i in "${!BATCHES[@]}"; do
    BATCH="${BATCHES[$i]}"
    LOG="$LOG_DIR/render_batch_$(printf '%02d' $i).log"
    echo "      Batch $i  → $(basename "$BATCH")  | log: $(basename "$LOG")"
    "$BLENDER" "$BLEND" \
        --background \
        --python "$ANNOTATE_SCRIPT" \
        -- \
        --frame_list "$BATCH" \
        --render \
        --out_dir "$OUT_DIR" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "      Waiting for all Blender instances to finish..."
echo "      Estimated time: $(( BATCH_SIZE * 32 / 60 )) min per instance (4 cams × 8 s/frame)"

ALL_OK=true
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo "      Batch $i (PID $PID): DONE"
    else
        echo "      Batch $i (PID $PID): FAILED (exit code $?)"
        ALL_OK=false
    fi
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo ""
echo "      Render phase: ${ELAPSED}s  (~$(( ELAPSED / 60 )) min)"

if [ "$ALL_OK" = false ]; then
    echo ""
    echo "WARNING: One or more batches failed. Check logs in $LOG_DIR"
    echo "         You can re-run to render only the remaining missing frames"
    echo "         (already-rendered frames are skipped automatically)."
    echo ""
fi

# ── Step 4: post-processing ────────────────────────────────────────────────────
echo "[4/4] Running post-processing (copy files, relabel classes, overlays)..."
python3 "$POST_SCRIPT"

echo ""
echo "All done. Run blender_mot_annotate.py manually for MOT re-annotation:"
echo ""
echo "  $BLENDER $BLEND \\"
echo "      --background --python $BASE/blender_mot_annotate.py \\"
echo "      -- --frames 100-1873"
echo ""
echo "Then apply track->class mapping with:"
echo "  python3 $BASE/fix_mot_classes.py"
