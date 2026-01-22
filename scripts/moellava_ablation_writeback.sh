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
LOG_FILE="$SCRIPT_DIR/tb_v1_moellava_pipeline_${TS}.txt"

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
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES:-"(not set in environment; set per command)"}"
echo

# # =============================================================================
# # 0) No Memory; Just Inference (Ablation none)
# # =============================================================================
# section "0) No Memory; Just Inference (Ablation mode=none)"
# run_cmd bash -lc \
#   'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
#     --ablate \
#     --ablate_mode none \
#     --items data/preprocess/tb_v1/v1/preprocess_manifest_ablate1.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --out_json output/ablation_none.json \
#     --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
#     --conv_mode phi \
#     --max_new_tokens 128 \
#     --temperature 0.2'

# # =============================================================================
# # 1) Build Base Memory (1-20000)
# # =============================================================================
# section "1) Build Base Memory (1-20000)"
# run_cmd bash -lc \
#   'python -m pipelines.build_memory \
#     --preprocess_manifest data/preprocess/tb_v1/v1/preprocess_manifest_base.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --data_root data/datasets/tb_v1 \
#     --device cuda \
#     --embed_dim 1024 \
#     --video_frames 2'

# =============================================================================
# 2) 1st Ablation - After Base
# =============================================================================
# section "2) 1st Ablation - After Base"
# run_cmd bash -lc \
#   'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
#     --ablate \
#     --items data/preprocess/tb_v1/v1/preprocess_manifest_ablate1.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-base.json \
#     --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
#     --conv_mode phi \
#     --max_new_tokens 128 \
#     --temperature 0.2'

# =============================================================================
# 3) 1st Writeback (20001-30000)
# =============================================================================
# section "3) 1st Writeback (20001-30000)"
# run_cmd bash -lc \
#   'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
#     --items data/preprocess/tb_v1/v1/preprocess_manifest_bquery1.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --out_json output/eval_tb_v1_phi2_4e.during-first-writeback.json \
#     --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
#     --conv_mode phi \
#     --max_new_tokens 128 \
#     --writeback \
#     --writeback_k_epi 0 \
#     --writeback_k_vis 0 \
#     --writeback_novelty_thr 0.65 \
#     --writeback_min_impression_chars 30 \
#     --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
#     --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# # =============================================================================
# # 4) 2nd Ablation - After Writeback1
# # =============================================================================
# section "4) 2nd Ablation - After Writeback1"
# run_cmd bash -lc \
#   'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
#     --ablate \
#     --items data/preprocess/tb_v1/v1/preprocess_manifest_ablate1.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-first-writeback.json \
#     --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
#     --conv_mode phi \
#     --max_new_tokens 128 \
#     --temperature 0.2'

# # =============================================================================
# # 5) 2nd Writeback (30001-40000)
# # =============================================================================
# section "5) 2nd Writeback (30001-40000)"
# run_cmd bash -lc \
#   'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
#     --items data/preprocess/tb_v1/v1/preprocess_manifest_bquery2.jsonl \
#     --mem_root data/memory_db/base/tb_v1/v1 \
#     --out_json output/eval_tb_v1_phi2_4e.during-second-writeback.json \
#     --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
#     --conv_mode phi \
#     --max_new_tokens 128 \
#     --writeback \
#     --writeback_k_epi 0 \
#     --writeback_k_vis 0 \
#     --writeback_novelty_thr 0.65 \
#     --writeback_min_impression_chars 30 \
#     --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
#     --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"'

# =============================================================================
# 6) 3rd Ablation - After Writeback2
# =============================================================================
section "6) 3rd Ablation - After Writeback2"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_ablate1.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1 \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-second-writeback.json \
    --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

# =============================================================================
# 7) 3rd Writeback (40001-50000)
# =============================================================================
section "7) 3rd Writeback (40001-50000)"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_bquery3.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1 \
    --out_json output/eval_tb_v1_phi2_4e.during-third-writeback.json \
    --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
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
# 8) 4th Ablation - After Writeback3
# =============================================================================
section "8) 4th Ablation - After Writeback3"
run_cmd bash -lc \
  'CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
    --ablate \
    --items data/preprocess/tb_v1/v1/preprocess_manifest_ablate1.jsonl \
    --mem_root data/memory_db/base/tb_v1/v1 \
    --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-third-writeback.json \
    --model_path LanguageBind/MoE-LLaVA-Phi2-2.7B-4e \
    --conv_mode phi \
    --max_new_tokens 128 \
    --temperature 0.2'

section "ALL DONE ✅"
echo "Final log saved to: $LOG_FILE"
