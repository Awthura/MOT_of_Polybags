#!/bin/bash
#SBATCH --job-name=yolo_hires
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-20:00:00
#SBATCH --output=logs/train_hires_%j.out
#SBATCH --error=logs/train_hires_%j.err

# Submit from ~/MOT_of_Polybags/training/
cd ~/MOT_of_Polybags/training
mkdir -p logs

source ~/venv/bin/activate

yolo obb train \
    data=~/MOT_of_Polybags/training/synth_data.yaml \
    model=yolo11n-obb.pt \
    epochs=100 \
    imgsz=1920 \
    batch=4 \
    amp=False \
    name=synth_hires \
    project=~/runs
