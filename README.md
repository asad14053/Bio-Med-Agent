# Multimodal Video Memory MVP

This repository contains the first MVP of a multimodal video agent with **memory-augmented reasoning**.  
Given an image or video, the system:

1. Extracts keyframes (for videos).
2. Encodes frames using ImageBind.
3. Retrieves top-k similar memories from a FAISS + JSONL memory store.
4. Builds a context-rich prompt.
5. Calls a Vision-Language Model (VLM) for inference.
6. Writes back new experience into the memory DB (JSONL + FAISS update).

This MVP is designed as a **sandbox** for experimenting with world memory, retrieval-augmented VLMs, and agentic workflows.

---

## Project Structure

```text
mvp_run.py          # Main runner for the MVP pipeline
configs.yaml        # Configuration for model paths, FAISS index, etc.

mvp/
  memory/
    episodic_store.py     # Stores per-episode interactions (sessions)
    visual_store.py       # Handles visual embeddings and FAISS operations
    semantic_store.py     # Handles text / semantic memory
    prompt_builder.py     # Assembles retrieved memory into prompts

tools/
  imagebind_encoder.py    # ImageBind embedding utilities
  keyframe.py             # Video → keyframe extraction
  videollava_cli.py       # VLM client for VideoLLaVA (or compatible model)
  moellava_cli.py         # VLM client for MoE-LLaVA (if used)

data/
  memory_db/              # FAISS index + JSONL + patient / user memory (ignored in git)
  tmp_frames/             # Temporary extracted frames (ignored in git)
```

## Getting Started
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
```bash
//using .yaml file
conda env create -f environment.yml
conda activate moellava
```
## Run the MVP
```bash
// cd to /scripts

python - <<EOF
from moellava.model.builder import load_pretrained_model
from moellava.conversation import conv_templates
print("MoE-LLaVA import OK")
EOF

export CUDA_VISIBLE_DEVICES=0
./image-inference.sh
```