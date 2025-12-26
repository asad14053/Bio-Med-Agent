#!/bin/bash

# =========================
# CONFIG
# =========================
ENV_NAME=moellava
MODEL_PATH="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e"
CONV_MODE="phi"

VIDEO_PATH="/path/to/video.mp4"
PATIENT_ID="P001"
QUESTION="Is there evidence of correct medication intake in this session?"

# =========================
# RUN MVP (VIDEO MODE)
# =========================
export CUDA_VISIBLE_DEVICES=0
python main_run.py \
  --patient_id ${PATIENT_ID} \
  --video ${VIDEO_PATH} \
  --question "${QUESTION}" \
  --model_path ${MODEL_PATH} \
  --conv_mode ${CONV_MODE}
