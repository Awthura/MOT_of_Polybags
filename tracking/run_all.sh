#!/bin/bash
# tracking/run_all.sh
#
# Run YOLO tracking on val + test with both ByteTrack and BoT-SORT,
# then evaluate and print a comparison table.
#
# Usage:
#   cd repo/tracking
#   ./run_all.sh                                       # hires model (default)
#   ./run_all.sh ../training/weights_synth_640.pt      # 640 model
#   ./run_all.sh ../training/weights_synth_hires.pt val bytetrack  # single run
#
# Dependencies: pip install ultralytics opencv-python motmetrics

set -e

MODEL="${1:-../training/weights_synth_hires.pt}"
DATASET="${2:-both}"
TRACKER="${3:-both}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================================="
echo "  Model:   $MODEL"
echo "  Dataset: $DATASET"
echo "  Tracker: $TRACKER"
echo "=============================================="

cd "$SCRIPT_DIR"

echo ""
echo ">>> Step 1: Running tracking..."
python run_tracking.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --tracker "$TRACKER" \
    --imgsz 1920 \
    --conf 0.25

echo ""
echo ">>> Step 2: Evaluating metrics..."
python evaluate_mot.py --all

echo ""
echo "Done. Results in: $(realpath ../tracking_results)"
