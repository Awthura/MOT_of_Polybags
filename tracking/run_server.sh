#!/bin/bash
#SBATCH --job-name=mcmot_benchmark
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ── MCMOT GPU Benchmark — OVGU cluster (ams partition) ─────────────────────
# Usage (interactive):   bash tracking/run_server.sh
# Usage (SLURM):         sbatch tracking/run_server.sh
#
# Run from repo root or from tracking/ — script auto-detects.
# After it finishes, copy the JSON locally:
#   scp diky85bu@ants:~/MOT_of_Polybags/tracking/results/benchmark_*.json .
#   python generate_report.py --from-json benchmark_YYYYMMDD_HHMMSS.json

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_DIR="$HOME/MOT_of_Polybags"
DATA_ROOT="$HOME/data"           # override: DATA_ROOT=/other/path bash run_server.sh
MODEL_PATH="$REPO_DIR/training/weights_synth_hires.pt"

# ── Python environment ────────────────────────────────────────────────────────
# Uses venv at ~/venv — activate only if not already in a venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV="$HOME/venv"
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "ERROR: venv not found at $VENV. Activate your environment first."
        exit 1
    fi
    source "$VENV/bin/activate"
fi

# ── Output paths ─────────────────────────────────────────────────────────────
cd "$REPO_DIR/tracking"
mkdir -p results logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JSON_OUT="results/benchmark_${TIMESTAMP}.json"

# ── Info ─────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  MCMOT Benchmark"
echo "  Job:       ${SLURM_JOB_ID:-interactive}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Python:    $(python --version)"
echo "  Data root: $DATA_ROOT"
echo "  Model:     $MODEL_PATH"
echo "  Output:    $JSON_OUT"
echo "========================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi — CPU mode)"

# ── Verify paths ─────────────────────────────────────────────────────────────
if [[ ! -f "$MODEL_PATH" ]]; then
    echo ""
    echo "WARNING: model not found at $MODEL_PATH"
    echo "  Provide path via: MODEL_PATH=/path/to/weights.pt bash run_server.sh"
    echo "  Or copy the weights file to $MODEL_PATH"
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo ""
    echo "WARNING: data root not found: $DATA_ROOT"
    echo "  Provide path via: DATA_ROOT=/path/to/data bash run_server.sh"
    exit 1
fi

echo ""
echo "Datasets found in $DATA_ROOT:"
ls "$DATA_ROOT" 2>/dev/null | grep -E "synth_dataset" || echo "  (no synth_dataset* dirs found)"
echo ""

# ── Run benchmark ─────────────────────────────────────────────────────────────
python server_benchmark.py \
    --data-root   "$DATA_ROOT" \
    --model       "$MODEL_PATH" \
    --device      auto \
    --imgsz       1920 \
    --conf        0.25 \
    --datasets    val test \
    --trackers    bytetrack botsort \
    --online      class_rank class_iou class_smooth \
    --offline     no_assoc class_only trk_temporal trk_spatial trk_combined \
    --out         "$JSON_OUT"

echo ""
echo "========================================================"
echo "  Done!"
echo ""
echo "  Copy results to local machine:"
echo "    scp diky85bu@ants:$REPO_DIR/tracking/$JSON_OUT ."
echo ""
echo "  Generate PDF report:"
echo "    python generate_report.py --from-json $(basename $JSON_OUT)"
echo "========================================================"
