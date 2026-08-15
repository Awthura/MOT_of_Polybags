#!/bin/bash
# YOLO11 detect training — polybag_v1, single class, from the tiled autolabels.
#
# Mirrors training/yolo_real/train.sh conventions for this cluster:
#   - launch via srun (sbatch had an unresolved permission issue here)
#   - amp=False to dodge the known AMP-validation hang on the GPU nodes
#
# On the cluster (repo at ~/MOT_of_Polybags):
#   cd ~/MOT_of_Polybags/real_polybags/training/yolo_polybag_v1
#   python3 build_dataset.py          # rebuild dataset/ so symlinks are cluster-local
#   srun -p gpu-stud --gres=gpu:1 -c 8 --mem=32G -w ant2 bash train.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

source ~/venv/bin/activate

# yolo11s = better detection (user wants better results); swap to yolo11n.pt for
# faster inference on the MacBook streamer if real-time matters more than recall.
MODEL=${MODEL:-yolo11s.pt}
IMGSZ=${IMGSZ:-960}          # small bags in lucid/rgbd views benefit from 960
BATCH=${BATCH:-8}
EPOCHS=${EPOCHS:-100}

yolo detect train \
    data=dataset/data.yaml \
    model="$MODEL" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    amp=False \
    name=polybag_v1 \
    project=~/runs_polybag
