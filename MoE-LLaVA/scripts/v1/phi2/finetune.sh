JSON_FOLDER="data/preprocess/tb_v1/v1"
IMAGE_FOLDER="data/datasets/tb_v1"

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_${USER}_ds
export TORCH_CUDA_ARCH_LIST="8.6"

export DS_BUILD_OPS=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1
export DEEPSPEED_DISABLE_FUSED_LAMB=1
export DS_DISABLE_FUSED_ADAM=1
export DS_DISABLE_FUSED_LAMB=1

CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 \
deepspeed --num_gpus=8 MoE-LLaVA/moellava/train/train_mem.py \
  --deepspeed MoE-LLaVA/scripts/zero1.json \
  --model_name_or_path microsoft/phi-2 \
  --version phi \
  --data_path ${JSON_FOLDER}/preprocess_manifest_finetune.jsonl \
  --image_folder ${IMAGE_FOLDER} \
  --image_tower openai/clip-vit-large-patch14-336 \
  --image_projector_type mlp2x_gelu \
  --pretrain_mm_mlp_adapter checkpoints/MoE-LLaVA-Phi2-Pretrain/mm_projector.bin \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
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
