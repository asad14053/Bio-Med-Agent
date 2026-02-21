#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Location-safe: run from repo root no matter where you call it from
# Script is in: Bio-Med-Agent/scripts/
# Repo root is: Bio-Med-Agent/
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------
# Log file (same path as script)
# -----------------------------
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$SCRIPT_DIR/tb_v1_moellava_pipeline_all-Regular_${TS}.txt"

# Tee all stdout+stderr to log
exec > >(tee -a "$LOG_FILE") 2>&1

section () {
  echo
  echo "================================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  echo "================================================================================"
}

run_cmd () {
  echo
  echo "+ $*"
  echo "--------------------------------------------------------------------------------"
  "$@"
  echo "--------------------------------------------------------------------------------"
  echo "✅ Completed: $*"
}

echo "Log file: $LOG_FILE"
echo "Repo root: $REPO_ROOT"
echo "Host: $(hostname)"
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES:-'(not set in environment; set per command)'}"
echo


# =============================================================================
# 0) No Memory; Just Inference (Ablation none)
# =============================================================================
#section "0) No Memory; Just Inference (Ablation mode=none)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --ablate_mode none \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/base \
    --out_json output/ablation_none.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

# =============================================================================
# 1) Build Base Memory (1-2000)
# =============================================================================
# section "1) Build Base Memory (1-2000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.build_memory \
    --preprocess_manifest data/preprocess/tb_v1/v1/preprocess_manifest_base.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/base \
    --data_root data/datasets/tb_v1 \
    --device cuda \
    --embed_dim 1024 \
    --video_frames 2'

# -------------------------------
# Snapshot Base Memory → First Writeback
# -------------------------------

SRC_DIR="data/memory_db/base/tb_v1/v1/Regular/base"
DST_DIR="data/memory_db/base/tb_v1/v1/Regular/first-writeback"

# Create destination if it doesn't exist
# Copy everything (preserve structure, timestamps, symlinks)
run_cmd mkdir -p "${DST_DIR}"
run_cmd cp -a "${SRC_DIR}/." "${DST_DIR}/"

# =============================================================================
# # 2) 1st Ablation - After Based
# =============================================================================
#section "2) 1st Ablation - After Base"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/base \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-base.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

# =============================================================================
# 2) 1st Writeback (1-3000)
# =============================================================================
# section "2) 1st Writeback (1-3000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_writeback1.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/first-writeback \
    --out_json output/eval_tb_v1_phi2_4e.during-first-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --writeback \
    --writeback_k_epi 0 \
    --writeback_k_vis 0 \
    --writeback_novelty_thr 0.65 \
    --writeback_min_impression_chars 30 \
    --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
    --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# -------------------------------
# First Writeback → Second Writeback
# -------------------------------

SRC_DIR="data/memory_db/base/tb_v1/v1/Regular/first-writeback"
DST_DIR="data/memory_db/base/tb_v1/v1/Regular/second-writeback"

# Create destination if it doesn't exist
# Copy everything (preserve structure, timestamps, symlinks)
run_cmd mkdir -p "${DST_DIR}"
run_cmd cp -a "${SRC_DIR}/." "${DST_DIR}/"


#=============================================================================
#4) 2nd Ablation - After Writeback1
#=============================================================================
section "4) 2nd Ablation - After Writeback1"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/first-writeback \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-first-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'


# =============================================================================
# 5) 2nd Writeback (3000-6000)
# =============================================================================
# section "5) 2nd Writeback (3000-6000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_writeback2.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/second-writeback \
    --out_json output/eval_tb_v1_phi2_4e.during-second-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --writeback \
    --writeback_k_epi 0 \
    --writeback_k_vis 0 \
    --writeback_novelty_thr 0.65 \
    --writeback_min_impression_chars 30 \
    --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
    --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# -------------------------------
# Second Writeback → Third Writeback
# -------------------------------

SRC_DIR="data/memory_db/base/tb_v1/v1/Regular/second-writeback"
DST_DIR="data/memory_db/base/tb_v1/v1/Regular/third-writeback"

# Create destination if it doesn't exist
# Copy everything (preserve structure, timestamps, symlinks)
run_cmd mkdir -p "${DST_DIR}"
run_cmd cp -a "${SRC_DIR}/." "${DST_DIR}/"



# =============================================================================
# 6) 3rd Ablation - After Writeback2
# =============================================================================
# section "6) 3rd Ablation - After Writeback2"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/second-writeback \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-second-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'


# =============================================================================
# 7) 3rd Writeback (6001-9000)
# =============================================================================
# section "7) 3rd Writeback (40001-50000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_writeback3.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/third-writeback \
    --out_json output/eval_tb_v1_phi2_4e.during-third-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --writeback \
    --writeback_k_epi 0 \
    --writeback_k_vis 0 \
    --writeback_novelty_thr 0.65 \
    --writeback_min_impression_chars 30 \
    --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
    --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# -------------------------------
# Third Writeback → Fourth Writeback
# -------------------------------

SRC_DIR="data/memory_db/base/tb_v1/v1/Regular/third-writeback"
DST_DIR="data/memory_db/base/tb_v1/v1/Regular/fourth-writeback"

# Create destination if it doesn't exist
# Copy everything (preserve structure, timestamps, symlinks)
run_cmd mkdir -p "${DST_DIR}"
run_cmd cp -a "${SRC_DIR}/." "${DST_DIR}/"


# =============================================================================
# 8) 4th Ablation - After Writeback3
# =============================================================================
# section "8) 4th Ablation - After Writeback3"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/third-writeback \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-third-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

# =============================================================================
# 7) 4th Writeback (9001-12000)
# =============================================================================
# section "7) 4th Writeback (9001-12000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_writeback4.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/fourth-writeback \
    --out_json output/eval_tb_v1_phi2_4e.during-fourth-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --writeback \
    --writeback_k_epi 0 \
    --writeback_k_vis 0 \
    --writeback_novelty_thr 0.65 \
    --writeback_min_impression_chars 30 \
    --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
    --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# =============================================================================
# 8) 5th Ablation - After Writeback4
# =============================================================================
# section "8) 4th Ablation - After Writeback5"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1/Regular/fourth-writeback \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-fourth-writeback.json \
    --model_path checkpoints/MoE-LLaVA-Phi2-finetune-Regular \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

section "ALL DONE ✅"
echo "Final log saved to: $LOG_FILE"
