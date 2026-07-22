#!/bin/bash
#SBATCH --job-name=yolo_resume
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-20:00:00
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err

# Submit from ~/MOT_of_Polybags/training/ so relative log paths resolve correctly
cd ~/MOT_of_Polybags/training
mkdir -p logs

source ~/venv/bin/activate

LAST_PT=~/runs/synth_round_0/weights/last.pt

if [ ! -f "$LAST_PT" ]; then
    echo "ERROR: checkpoint not found at $LAST_PT"
    echo "Make sure the first training job (train_synth.slurm) has started before resuming."
    exit 1
fi

echo "Resuming from: $LAST_PT"
yolo obb train resume model="$LAST_PT"
