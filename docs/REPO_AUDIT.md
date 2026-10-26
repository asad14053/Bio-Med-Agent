# Repository Audit

This is a read-only audit of branch `main` at commit `db292e7`. It was done while preparing the documentation, and no source files were changed. The problems found are listed in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## First-party code

| Path | Lines | Purpose | Paper section |
|---|---|---|---|
| `main_run.py` | 210 | Single-query demo: encode an image or video → retrieve → prompt → MoE-LLaVA → writeback | Online loop (demo) |
| `memory/episodic_store.py` | 94 | `EpisodicStore`: FAISS `IndexFlatIP` + JSONL episodic memory | M_E |
| `memory/visual_store.py` | 93 | `VisualStore`: FAISS + JSONL visual memory | M_V |
| `memory/semantic_store.py` | 59 | `SemanticStore`: per-patient JSON rolling impression summary | M_S, U_S |
| `memory/prompt_builder.py` | 128 | `build_prompt()`: fuses S/E/V memory and the question into the prompt | Prompt fusion P(t) |
| `models/moe_llava_infer.py` | 238 | `MoELLaVAInfer`: in-process MoE-LLaVA inference, token-budget shrink, keyword stopping | Online loop |
| `models/llava_phi_infer.py` | 154 | `LlavaPhiInfer`: non-MoE LLaVA-Phi wrapper (not imported anywhere) | Online loop (FT path) |
| `tools/encoder_imagebind.py` | 204 | `ImageBindEncoder`: 1024-d L2-normalized ImageBind embeddings | f_θ |
| `tools/keyframe.py` | 110 | Video keyframe extraction (mid-frame or uniform) | Preprocessing (video) |
| `pipelines/preprocess_dataset.py` | 252 | Resolve media paths and select keyframes for each manifest row | Φ_pre |
| `pipelines/build_memory.py` | 203 | Build base memory from a manifest using ground truth | Offline loop |
| `pipelines/eval_runner.py` | 650 | Batch evaluation, 8-way memory ablation, gated writeback, metrics | Online loop, writeback, metrics |
| `pipelines/remove_bad.py` | 19 | Drop rows that have no question or answer (hard-coded paths) | Φ_pre |
| `pipelines/shuffle_jsonl.py` | 73 | Seeded subject-balanced shuffle | Dataset |
| `pipelines/split_jsonl.py` | 41 | Line-range splitter (stale ranges) | Dataset |

## Scripts and results

| Path | Purpose |
|---|---|
| `scripts/image-inference.sh` | Runs `main_run.py` on one image (run from `scripts/`) |
| `scripts/video-inference.sh` | Runs `main_run.py` on one video (run from the repo root) |
| `scripts/moellava_full_ablation.sh` | Full pipeline for the **Base** backbone (older manifests: `ablate1`, `bquery1–3`) |
| `scripts/moellava_ablation_writeback.sh` | Partial or resumed Base-backbone pipeline |
| `scripts/moellava_memory_writeback_regular.sh` | Full pipeline for the **FT** backbone (base → WB1–4, ablation after each phase) |
| `scripts/moellava_memory_writeback_moe.sh` | Pipeline for the **MoE-FT** backbone (most steps commented out) |
| `scripts/tb_v1_moellava_pipeline_*.txt` | 12 run logs written by the drivers (Jan–Feb 2026) |
| `scripts/plot/{E,EV,SEV,figure2}.csv` | Published metric values (inputs to the plots) |
| `scripts/plot/fig-{1,2,3}.py` | Plot scripts |
| `scripts/plot/*.png` | Rendered figures |

## Third-party code (vendored)

| Path | Origin | Modified? |
|---|---|---|
| `MoE-LLaVA/` | PKU-YuanGroup/MoE-LLaVA (Apache-2.0) | **Yes.** `moellava/model/builder.py`, `moellava/train/train.py`, the phi2 fine-tune scripts and the DeepSpeed configs were changed or added in `db292e7`. |
| `ImageBind/` | facebookresearch/ImageBind (CC-BY-NC-SA 4.0) | Not checked against upstream |
| `flash-attention/` | git link to commit `7b0bfcc3`, with no `.gitmodules` | Empty after clone |

## Checkpoints

| Path | Contents | Missing |
|---|---|---|
| `checkpoints/MoE-LLaVA-Phi2-finetune-MoE/` | `config.json`, tokenizer files, `trainer_state.json`, safetensors index, 32 TensorBoard event files | 4 × `model-0000x-of-00004.safetensors` |
| `checkpoints/MoE-LLaVA-Phi2-finetune-Regular/` | same file types, 8 TensorBoard event files | 2 × `model-0000x-of-00002.safetensors` |

`checkpoints/` is listed in `.gitignore`, but these files were committed anyway (force-added).

## Environment files

| File | Notes |
|---|---|
| `requirements.txt` | 17 entries, no versions, installs upstream MoE-LLaVA and ImageBind from git |
| `environment.yaml` | Python 3.10, CUDA toolkit 11.8, env name `Bio-Med-Agent`, the same pip list |
| `.gitignore` | Covers `data/`, `checkpoints/`, weights, caches and `output/` |

## Problems with the previous README (now at `docs/legacy/README_MVP.md`)

| Old README says | Reality |
|---|---|
| "Multimodal Video Memory MVP" | The paper is about chest-X-ray report generation with tri-memory |
| `mvp_run.py` | The file is `main_run.py` |
| `configs.yaml` | No config file exists; all settings are CLI flags |
| `mvp/memory/…` | The folder is `memory/` at the top level |
| `tools/imagebind_encoder.py` | The file is `tools/encoder_imagebind.py` |
| `tools/videollava_cli.py`, `tools/moellava_cli.py` | These files do not exist (inference lives in `models/`) |
| `conda env create -f environment.yml` / env `moellava` | The file is `environment.yaml` and creates the env `Bio-Med-Agent` |
| "cd to /scripts … ./image-inference.sh" | Correct for the image script only, and the image file it uses is not in the repo |
| No mention of `pipelines/` | `pipelines/` holds all experiment code |
