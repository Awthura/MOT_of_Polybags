#!/bin/bash
#SBATCH --job-name=mcmot_benchmark
#SBATCH --partition=ams
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# ── MCMOT GPU Benchmark — OVGU cluster (ams partition) ─────────────────────
# Usage:
#   sbatch run_server.sh
# Or interactively:
#   bash run_server.sh  (will use CPU if no GPU allocated)
#
# Results will be saved to:
#   results/benchmark_YYYYMMDD_HHMMSS.json
#
# After copying the JSON locally:
#   python generate_report.py --from-json benchmark_YYYYMMDD_HHMMSS.json

set -euo pipefail

# ── Paths — edit these to match your server layout ──────────────────────────
REPO_DIR="$HOME/repo"
DATA_ROOT="$HOME/AMS"                        # contains synth_dataset_val/, etc.
MODEL_PATH="$REPO_DIR/training/weights_synth_hires.pt"

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SCRIPT" ]]; then
    CONDA_SCRIPT="/opt/conda/etc/profile.d/conda.sh"
fi
if [[ ! -f "$CONDA_SCRIPT" ]]; then
    echo "ERROR: conda not found. Edit CONDA_SCRIPT in run_server.sh"
    exit 1
fi
source "$CONDA_SCRIPT"
conda activate ams

# ── Output paths ─────────────────────────────────────────────────────────────
cd "$REPO_DIR/tracking"
mkdir -p results logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JSON_OUT="results/benchmark_${TIMESTAMP}.json"

# ── Info ─────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  MCMOT Benchmark"
echo "  Job:       ${SLURM_JOB_ID:-local}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Data root: $DATA_ROOT"
echo "  Model:     $MODEL_PATH"
echo "  Output:    $JSON_OUT"
echo "========================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

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
echo "  Done! Copy results from server:"
echo ""
echo "  scp diky85bu@<server>:$REPO_DIR/tracking/$JSON_OUT ."
echo "  python generate_report.py --from-json $(basename $JSON_OUT)"
echo "========================================================"
