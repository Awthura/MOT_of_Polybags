#!/bin/bash
# First-time cluster setup — run once on the login node (NOT in a slurm job)
# Usage: bash ~/MOT_of_Polybags/training/cluster_setup.sh
set -e

source /opt/spack/main/env.sh

echo "[1/3] Setting up Python venv at ~/venv ..."
if [ -d ~/venv ]; then
    echo "  ~/venv already exists — skipping creation, updating packages."
    source ~/venv/bin/activate
else
    python3 -m venv ~/venv
    source ~/venv/bin/activate
fi

pip install --upgrade pip
pip install ultralytics

echo "[2/3] Creating logs directory ..."
mkdir -p ~/MOT_of_Polybags/training/logs

echo "[3/3] Verifying torch + CUDA ..."
python3 - <<'EOF'
import torch
print(f"  torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "Submit training job:"
echo "  cd ~/MOT_of_Polybags/training"
echo "  sbatch train_synth.slurm"
echo ""
echo "Check job status:"
echo "  squeue --me"
echo ""
echo "Watch live log (replace JOBID):"
echo "  tail -f ~/MOT_of_Polybags/training/logs/train_synth_JOBID.out"
echo ""
echo "If job runs out of time (1-day limit), resume it:"
echo "  sbatch resume_synth.slurm"
