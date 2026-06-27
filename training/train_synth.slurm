#!/bin/bash
#SBATCH --job-name=yolo_synth
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-20:00:00
#SBATCH --output=logs/train_synth_%j.out
#SBATCH --error=logs/train_synth_%j.err

# Submit from ~/MOT_of_Polybags/training/ so relative log paths resolve correctly
cd ~/MOT_of_Polybags/training
mkdir -p logs

source /opt/spack/main/env.sh
source ~/venv/bin/activate

yolo obb train \
    data=~/MOT_of_Polybags/training/synth_data.yaml \
    model=yolo11n-obb.pt \
    epochs=100 \
    imgsz=640 \
    batch=32 \
    amp=False \
    name=synth_round_0 \
    project=~/runs
