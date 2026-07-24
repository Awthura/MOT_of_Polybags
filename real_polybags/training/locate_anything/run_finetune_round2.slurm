#!/bin/bash
#SBATCH --job-name=locany_round2
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/round2_%j.out
#SBATCH --error=logs/round2_%j.err

# ── LocateAnything-3B round-2 domain-adaptive LoRA fine-tune ────────────────
# Combines the 569-image clean annotated set with cross-validated,
# belt-masked pseudo-labels from the 2,926-image real conveyor dataset.
# See prepare_jsonl_round2.py and AUTOLABEL_REPORT.md "Next steps".
#
# Usage:  sbatch run_finetune_round2.slurm
# Monitor: squeue -u $USER ; tail -f logs/round2_<jobid>.out
# After it finishes, pull the checkpoint locally:
#   scp -r diky85bu@ants:~/runs_locateanything_lora/real_v4_round2/checkpoint-* .

set -euo pipefail

EAGLE_DIR="${EAGLE_DIR:-$HOME/Eagle}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    VENV="$HOME/venv"
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "ERROR: venv not found at $VENV." >&2
        exit 1
    fi
    source "$VENV/bin/activate"
fi

# Under sbatch (not interactive srun), BASH_SOURCE[0] resolves to a node-local
# slurm spool path (/var/spool/slurmd/job<id>/slurm_script), not the real
# script location -- cd'ing there and mkdir'ing fails with permission denied.
# SLURM_SUBMIT_DIR holds the actual submission directory and is reliable here.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "========================================================"
echo "  LocateAnything-3B round-2 domain-adaptive fine-tune"
echo "  Job:    ${SLURM_JOB_ID:-interactive}"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Python: $(python --version)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

if [[ ! -d "$EAGLE_DIR/Embodied" ]]; then
    echo "ERROR: Eagle repo not found at $EAGLE_DIR (expected $EAGLE_DIR/Embodied)" >&2
    exit 1
fi

# 4,224 effective images/epoch (1,707 from the repeated 569-image annotated
# set + 2,517 pseudo-labelled). Same batch=1 * grad_acc=4 -> effective batch 4
# as the successful real_v3_multibox_sdpa run. 2 epochs to give the model
# real exposure to the noisier conveyor domain without overfitting the
# pseudo-labels in a single pass.
export META_PATH="$SCRIPT_DIR/jsonl_data_round2/recipe.json"
export OUTPUT_DIR="$HOME/runs_locateanything_lora/real_v4_round2"
export ATTN_IMPL=sdpa
export N_TRAIN=4224
export EPOCHS=2
export PER_DEVICE_BATCH_SIZE=1
export GRADIENT_ACC=4

cd "$EAGLE_DIR/Embodied"
# eaglevl isn't pip-installed (no editable install, no site .pth) -- torch's
# elastic launch doesn't add cwd to sys.path for a relative script path, so
# eaglevl/train/locany_finetune_magi_stream.py can't find the eaglevl package
# without this explicit on PYTHONPATH.
export PYTHONPATH="$EAGLE_DIR/Embodied${PYTHONPATH:+:$PYTHONPATH}"
# locany_finetune_magi_stream.py defaults to LAUNCHER=slurm, which requires
# SLURM_NTASKS/SLURM_PROCID -- only set by an inner `srun`, not by a plain
# `sbatch` batch script. We're launching via torchrun (torch.distributed.run),
# which sets RANK/WORLD_SIZE/MASTER_ADDR itself, so force the matching launcher.
export LAUNCHER=pytorch
bash "$SCRIPT_DIR/finetune_lora.sh"

echo ""
echo "========================================================"
echo "  Done! Pull the checkpoint locally:"
echo "    scp -r diky85bu@ants:$OUTPUT_DIR ."
echo "========================================================"
