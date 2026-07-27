#!/bin/bash
# YOLO11n-OBB training on the real annotated dataset (train_v11_obb_final /
# val_v11_obb_final). Mirrors the conventions of synthetic_polybags/training/
# train_synth_hires.slurm (amp=False to avoid the known AMP-validation hang
# on cluster GPU nodes).
#
# Run via srun (sbatch hit an unresolved permission issue on this cluster for
# the LocateAnything pilot job — srun works reliably, so sticking with it):
#   srun -p gpu-stud --gres=gpu:1 -c 8 --mem=32G -w ant2 bash train.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

source ~/venv/bin/activate

yolo obb train \
    data=data.yaml \
    model=yolo11n-obb.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    amp=False \
    name=real_v1 \
    project=~/runs_real
