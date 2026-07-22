#!/bin/bash
#SBATCH --job-name=locany_pilot
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err

# ── LocateAnything-3B zero-shot pilot — OVGU cluster (gpu-stud partition) ───
# Usage:  sbatch run_pilot.slurm
# Run from real_polybags/training/locate_anything/ (or repo root; auto-detects).
#
# After it finishes, pull results locally:
#   scp -r diky85bu@ants:~/real_polybags/training/locate_anything/results .

set -euo pipefail

EAGLE_DIR="${EAGLE_DIR:-$HOME/Eagle}"
MODEL="${MODEL:-nvidia/LocateAnything-3B}"

# ── Python environment ────────────────────────────────────────────────────────
# Reuses the venv already set up for the synthetic-pipeline cluster training.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV="$HOME/venv"
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "ERROR: venv not found at $VENV. Activate your environment first."
        exit 1
    fi
    source "$VENV/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p results logs

echo "========================================================"
echo "  LocateAnything-3B zero-shot pilot"
echo "  Job:        ${SLURM_JOB_ID:-interactive}"
echo "  Node:       $(hostname)"
echo "  Date:       $(date)"
echo "  Python:     $(python --version)"
echo "  Eagle repo: $EAGLE_DIR"
echo "  Model:      $MODEL"
echo "========================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi — CPU mode)"

if [[ ! -d "$EAGLE_DIR/Embodied" ]]; then
    echo ""
    echo "ERROR: Eagle repo not found at $EAGLE_DIR (expected $EAGLE_DIR/Embodied)"
    echo "  Clone it first:  git clone https://github.com/NVlabs/Eagle.git $EAGLE_DIR"
    echo "  Or point at a different path: EAGLE_DIR=/path/to/Eagle sbatch run_pilot.slurm"
    exit 1
fi

# ── Ensure required packages are installed ───────────────────────────────────
pip install -q "transformers==4.57.1" timm accelerate peft opencv-python-headless pillow numpy 2>/dev/null || true

# ── Run pilot ─────────────────────────────────────────────────────────────────
python run_pilot.py \
    --eagle-dir "$EAGLE_DIR" \
    --model     "$MODEL" \
    --device    cuda \
    --out-dir   results

echo ""
echo "========================================================"
echo "  Done!"
echo ""
echo "  Pull results to local machine:"
echo "    scp -r diky85bu@ants:$SCRIPT_DIR/results ."
echo "========================================================"
