#!/usr/bin/env bash
# LoRA fine-tune LocateAnything-3B on the real annotated dataset (single GPU).
# Adapted from NVlabs/Eagle's Embodied/shell/locate-anything-lora-visual-prompt.sh
# for: 1 GPU (their default is 8), a small ~569-image dataset (their default
# MAX_STEPS=5000 targets a 138M-query pretraining mixture, not us), and no
# W&B account.
#
# Run from ~/Eagle/Embodied (the training script's imports are relative to there):
#   META_PATH=~/MOT_of_Polybags/real_polybags/training/locate_anything/jsonl_data/recipe.json \
#   bash ~/MOT_of_Polybags/real_polybags/training/locate_anything/finetune_lora.sh
set -euo pipefail

export WANDB_DISABLED=true
export HF_TOKEN="${HF_TOKEN:-not_required_model_is_public}"

GPUS=${GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/runs_locateanything_lora/real_v1"}
MODEL_PATH=${MODEL_PATH:-"nvidia/LocateAnything-3B"}

if [[ -z "${META_PATH:-}" ]]; then
  echo "Please set META_PATH to the recipe.json produced by prepare_jsonl.py" >&2
  exit 1
fi
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config.json"}

# 8044 train examples (one per box, not per image — see prepare_jsonl.py's
# docstring for why: NVlabs' packing code has a real bug triggered by >1
# supervised detection per example). batch=1 * grad_acc=4 -> effective batch
# 4 -> ~2011 steps/epoch. 1 epoch is already ~14x the image-exposure of the
# original 6-epoch/569-image plan, so default EPOCHS is much lower now.
# Override EPOCHS/PER_DEVICE_BATCH_SIZE/GRADIENT_ACC/N_TRAIN to retune if the
# dataset size or batch size changes.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=${GRADIENT_ACC:-4}
EPOCHS=${EPOCHS:-1}
N_TRAIN=${N_TRAIN:-8044}
STEPS_PER_EPOCH=$(( (N_TRAIN + PER_DEVICE_BATCH_SIZE * GRADIENT_ACC - 1) / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACC) ))
MAX_STEPS=${MAX_STEPS:-$(( STEPS_PER_EPOCH * EPOCHS ))}
SAVE_STEPS=${SAVE_STEPS:-$STEPS_PER_EPOCH}
LR=${LR:-2e-5}
WARMUP_STEPS=${WARMUP_STEPS:-$(( STEPS_PER_EPOCH / 2 ))}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
ATTN_IMPL=${ATTN_IMPL:-magi}

USE_LLM_LORA=${USE_LLM_LORA:-64}
USE_BACKBONE_LORA=${USE_BACKBONE_LORA:-0}

mkdir -p "$OUTPUT_DIR"
echo "steps/epoch=$STEPS_PER_EPOCH  max_steps=$MAX_STEPS  save_steps=$SAVE_STEPS  attn=$ATTN_IMPL"

python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node="$GPUS" --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$MODEL_PATH" \
  --max_steps "$MAX_STEPS" \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --overwrite_output_dir False \
  --block_size "${BLOCK_SIZE:-6}" \
  --attn_implementation "$ATTN_IMPL" \
  --causal_attn False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora "$USE_LLM_LORA" \
  --use_backbone_lora "$USE_BACKBONE_LORA" \
  --vision_select_layer -1 \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --bf16 True \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACC" \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 3 \
  --learning_rate "$LR" \
  --weight_decay 0.01 \
  --warmup_steps "$WARMUP_STEPS" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --sample_log_interval 20 \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_num_tokens_per_sample "$MAX_SEQ_LENGTH" \
  --max_num_tokens "$MAX_SEQ_LENGTH" \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --report_to "tensorboard" \
  --run_name "real_polybags_lora_v1" \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
