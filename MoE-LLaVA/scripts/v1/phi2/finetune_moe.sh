#!/bin/bash
set -euo pipefail

JSON_FOLDER="data/preprocess/tb_v1/v1"
IMAGE_FOLDER="data/datasets/tb_v1"

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_${USER}_ds
export TORCH_CUDA_ARCH_LIST="8.6"

export DS_BUILD_OPS=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1
export DEEPSPEED_DISABLE_FUSED_LAMB=1
export DS_DISABLE_FUSED_ADAM=1
export DS_DISABLE_FUSED_LAMB=1

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=29501

# MoE settings (start memory-safe)
moe_mode="sparse"
num_experts=8
top_k_experts=2
capacity_factor=1.25
use_residual=False
router_aux_loss_coef=0.01
train_modules=("wg")




DATA_JSON="${JSON_FOLDER}/preprocess_manifest_finetune.jsonl"
DS_JSON="MoE-LLaVA/scripts/zero1-moe.json"

MODEL_NAME_OR_PATH="checkpoints/MoE-LLaVA-Phi2-finetune-Regular"
OUTPUT_DIR="checkpoints/MoE-LLaVA-Phi2-finetune-MoE"

deepspeed --num_gpus=1 --master_port "${MASTER_PORT}" \
  MoE-LLaVA/moellava/train/train_mem.py \
  --deepspeed "$(realpath "$DS_JSON")" \
  --moe_enable True \
  --num_experts "$num_experts" \
  --top_k_experts "$top_k_experts" \
  --capacity_factor "$capacity_factor" \
  --moe_mode "$moe_mode" \
  --use_residual "$use_residual" \
  --router_aux_loss_coef "$router_aux_loss_coef" \
  --train_modules "${train_modules[@]}" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --version phi \
  --data_path "$(realpath "$DATA_JSON")" \
  --image_folder "$(realpath "$IMAGE_FOLDER")" \
  --image_tower openai/clip-vit-large-patch14-336 \
  --image_projector_type mlp2x_gelu \
  --pretrain_mm_mlp_adapter checkpoints/MoE-LLaVA-Phi2-Pretrain/mm_projector.bin \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --learning_rate 5e-5 \
  --report_to tensorboard \
  --cache_dir "./cache_dir"
