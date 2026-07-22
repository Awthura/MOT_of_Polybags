#!/bin/bash
# tracking/run_all.sh
#
# Full pipeline: detection → intra-camera tracking → inter-camera association
# → MOT metrics. Offers two modes:
#
#   SEQUENTIAL MODE (default, existing pipeline)
#   ── processes cameras one-by-one, then associates tracklets offline
#   ./run_all.sh
#   ./run_all.sh val bytetrack
#
#   REALTIME MODE (new architecture)
#   ── all 4 cameras inferred simultaneously per frame, online association
#   ./run_all.sh --realtime
#   ./run_all.sh --realtime val class_rank
#   ./run_all.sh --realtime both all      # all datasets × all methods
#
# Arguments (sequential):  [model] [dataset] [tracker]
# Arguments (realtime):    --realtime [dataset] [method] [tracker]
#
# Dependencies: pip install ultralytics opencv-python motmetrics scipy

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-../training/weights_synth_hires.pt}"

if [[ "$1" == "--realtime" ]]; then
    # ── Real-time parallel mode ────────────────────────────────────────────────
    DATASET="${2:-val}"
    METHOD="${3:-all}"
    TRACKER="${4:-bytetrack}"

    echo "=============================================="
    echo "  MODE:    REALTIME (parallel 4-cam + online association)"
    echo "  Model:   $MODEL"
    echo "  Dataset: $DATASET"
    echo "  Method:  $METHOD"
    echo "  Tracker: $TRACKER"
    echo "=============================================="

    cd "$SCRIPT_DIR"
    python run_realtime_mcmot.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --method  "$METHOD" \
        --tracker "$TRACKER" \
        --imgsz 1920 \
        --conf 0.25

else
    # ── Sequential mode (original pipeline) ───────────────────────────────────
    DATASET="${2:-both}"
    TRACKER="${3:-both}"

    echo "=============================================="
    echo "  MODE:    SEQUENTIAL (single-cam then offline association)"
    echo "  Model:   $MODEL"
    echo "  Dataset: $DATASET"
    echo "  Tracker: $TRACKER"
    echo "=============================================="

    cd "$SCRIPT_DIR"

    echo ""
    echo ">>> Step 1: Intra-camera tracking..."
    python run_tracking.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --tracker "$TRACKER" \
        --imgsz 1920 \
        --conf 0.25

    echo ""
    echo ">>> Step 2: Per-camera MOT metrics..."
    python evaluate_mot.py --all

    echo ""
    echo ">>> Step 3: Inter-camera association benchmark (5 methods)..."
    python associate_cameras.py \
        --dataset "$DATASET" \
        --tracker "$TRACKER" \
        --method all

    echo ""
    echo "Done. Results in: $(realpath ../tracking_results)"
fi
