# Models and Checkpoints

## Backbones used in the paper

| Paper name | Architecture | Path / ID used by the scripts | How it was produced | Weights available? |
|---|---|---|---|---|
| **Base** | MoE-LLaVA-Phi2-2.7B, 4 experts | `LanguageBind/MoE-LLaVA-Phi2-2.7B-4e` (Hugging Face Hub) | public release | ✅ downloads automatically |
| **FT** (regular fine-tune, non-MoE) | `LlavaPhiForCausalLM` (`model_type: llava_phi`) | `checkpoints/MoE-LLaVA-Phi2-finetune-Regular` | `MoE-LLaVA/scripts/v1/phi2/finetune.sh`: from `microsoft/phi-2` + pretrained projector, on `preprocess_manifest_finetune.jsonl` | ❌ configs only `<TODO: author to publish weights>` |
| **MoE-FT** (router-only MoE fine-tune) | `MoELLaVAPhiForCausalLM` (`model_type: moe_llava_phi`) | `checkpoints/MoE-LLaVA-Phi2-finetune-MoE` | `MoE-LLaVA/scripts/v1/phi2/finetune_moe.sh`: initialized from the FT checkpoint, MoE-ified, training only `wg` | ❌ configs only `<TODO: author to publish weights>` |

### Shared settings (from `config.json`)

- **Vision tower:** `openai/clip-vit-large-patch14-336` (layer −2, patch features), projector `mlp2x_gelu`, `image_aspect_ratio: pad`.
- **LLM:** Phi-2 with 32 layers and 32 heads. `max_position_embeddings` is 2048.

### MoE-FT configuration (`checkpoints/MoE-LLaVA-Phi2-finetune-MoE/config.json → moe`)

| Key | Value |
|---|---|
| `num_experts` | 8 per MoE layer |
| `moe_layers_idx` | `[0, 2, 4, …, 30]` (every other layer, 16 layers) |
| `top_k_experts` | 2 |
| `train_modules` | `["wg"]` (router gate only; experts frozen) |
| `capacity_factor` / `eval_capacity_factor` | 1.25 / 2.0 |
| `router_aux_loss_coef` | 0.01 |
| `moe_mode` | `sparse` |
| `use_residual` | `False` |

### Training hyperparameters (from the scripts and `trainer_state.json`)

| | FT | MoE-FT |
|---|---|---|
| Script | `MoE-LLaVA/scripts/v1/phi2/finetune.sh` | `MoE-LLaVA/scripts/v1/phi2/finetune_moe.sh` |
| Init | `microsoft/phi-2` + `checkpoints/MoE-LLaVA-Phi2-Pretrain/mm_projector.bin` | `checkpoints/MoE-LLaVA-Phi2-finetune-Regular` |
| DeepSpeed config | `MoE-LLaVA/scripts/zero1.json` | `MoE-LLaVA/scripts/zero1-moe.json` (ZeRO stage 0, AdamW, WarmupLR with 0 warmup steps, grad clip 1.0) |
| GPUs | 8 | 1 |
| Per-device batch / grad accum | 1 / 1 | 1 / 1 |
| Epochs / steps | 1 / 3,750 | 1 / 30,000 |
| Learning rate | 5e-5 | 5e-5 |
| `model_max_length` | 1024 | 1024 |
| Precision | bf16 + tf32 | bf16 + tf32 |
| Final train loss | 0.964 | 0.886 |
| Wall-clock time | ≈ 7.0 h | ≈ 7.5 h |

Reproduce the trainings from the repo root:

```bash
bash MoE-LLaVA/scripts/v1/phi2/finetune.sh       # FT
bash MoE-LLaVA/scripts/v1/phi2/finetune_moe.sh   # MoE-FT (needs FT checkpoint)
```

### Missing for full reproduction

- `checkpoints/MoE-LLaVA-Phi2-Pretrain/mm_projector.bin` (the stage-1 projector). `<TODO: author to give its source, e.g. the public MoE-LLaVA Phi-2 pretrain release, or how it was trained with MoE-LLaVA/scripts/v1/phi2/pretrain.sh>`
- The `*.safetensors` shards for both fine-tuned checkpoints (4 shards for MoE-FT, 2 for FT).

## Visual encoder for memory keys

`imagebind_huge(pretrained=True)` from the vendored `ImageBind/` package. It downloads `imagebind_huge.pth` on first use. Embeddings are 1024-d, computed on 224×224 inputs with CLIP normalization, and L2-normalized.

## Inference wrappers

| Class | File | Used by |
|---|---|---|
| `MoELLaVAInfer` | `models/moe_llava_infer.py` | `main_run.py`, `pipelines/eval_runner.py` (all evaluations) |
| `LlavaPhiInfer` | `models/llava_phi_infer.py` | not imported anywhere (see [KNOWN_ISSUES B4](KNOWN_ISSUES.md)) |
