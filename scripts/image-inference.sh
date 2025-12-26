#!/bin/bash

# =========================
# CONFIG
# =========================
ENV_NAME=moellava
MODEL_PATH="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e"
CONV_MODE="phi"

IMAGE_PATH="../data/image_file/synpic53228.jpg"
PATIENT_ID="P001"
QUESTION="Are the hilar lymph nodes enlarged?"

# =========================
# RUN MVP (IMAGE MODE)
# =========================
export CUDA_VISIBLE_DEVICES=0
python ../main_run.py \
  --patient_id ${PATIENT_ID} \
  --image ${IMAGE_PATH} \
  --question "${QUESTION}" \
  --model_path ${MODEL_PATH} \
  --conv_mode ${CONV_MODE}
