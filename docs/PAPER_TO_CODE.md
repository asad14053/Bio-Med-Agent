# Paper ↔ Code Map

This page maps each component of the MSM-Mem paper ([arXiv:2608.21810](https://arxiv.org/abs/2608.21810)) to the code that implements it in this repository. Where the code and the paper disagree, the **Match?** column says so. Details for every mismatch are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**Legend.** ✅ matches the paper · ⚠️ partial match or different value · ❌ not found in the repo

"Default in code" is the `argparse` or function default. "Used in scripts" is the value passed by the shell scripts in [`scripts/`](../scripts), which are the scripts that produced the runs behind the paper.

## 1. Memory store, M = {M_S, M_E, M_V}

| Paper concept | Paper location | Code location | Flag / key | Default in code | Value in paper | Match? |
|---|---|---|---|---|---|---|
| Episodic memory M_E | §Method | [`memory/episodic_store.py`](../memory/episodic_store.py): `EpisodicStore` | `--mem_root` | `data/memory_db` | — | ✅ |
| Episodic record fields | §Method | `EpisodicStore.add()` | — | `episode_id, patient_id, ts, z, question, answer_short, decision, media_path, clip_desc` | subject ID, episode ID, timestamp, z, q, a, d, image path, image description | ⚠️ `patient_id` is the subject ID and `clip_desc` is the description. There is no separate "image description" field. |
| Visual memory M_V | §Method | [`memory/visual_store.py`](../memory/visual_store.py): `VisualStore` | `--mem_root` | — | — | ✅ |
| Visual record fields | §Method | `VisualStore.add()` | — | `visual_id, patient_id, ts, z, image_path, visual_tag, source_episode_id` | subject/episode IDs, visual obs ID, timestamp, embedding, tag, image path | ✅ |
| Semantic memory M_S (per-patient summary) | §Method | [`memory/semantic_store.py`](../memory/semantic_store.py): `SemanticStore` | — | one JSON file per patient under `<mem_root>/semantic/` | per-patient JSON | ✅ |
| Semantic update operator U_S (deterministic) | §Method | `SemanticStore.update()` | — | appends the impression and keeps the last *N* | deterministic merge of impression + decision | ⚠️ The decision is not stored. The docstring says "NO decisions". |
| Semantic history buffer N | App. A.1 | `SemanticStore.__init__(max_history=3)` | not exposed as a flag | **3** | **5** | ⚠️ |
| Storage backend | App. A.1 | `faiss.IndexFlatIP` + JSONL in both stores | — | `episodic.jsonl` / `episodic.faiss`, `visual.jsonl` / `visual.faiss` | same | ✅ |

## 2. Visual encoder f_θ

| Paper concept | Paper location | Code location | Flag / key | Default in code | Value in paper | Match? |
|---|---|---|---|---|---|---|
| ImageBind encoder | §Method, App. A.1 | [`tools/encoder_imagebind.py`](../tools/encoder_imagebind.py): `ImageBindEncoder` (`imagebind_huge(pretrained=True)`) | — | — | ImageBind | ✅ |
| Input resolution | App. A.1 | `ImageBindConfig.image_size` | — | 224 | 224×224 | ✅ |
| CLIP-style normalization | App. A.1 | `ImageBindEncoder.transform` (mean `0.4815,0.4578,0.4082`, std `0.2686,0.2613,0.2758`) | — | — | CLIP norm | ✅ |
| L2 normalization | App. A.1 | `ImageBindEncoder._l2norm()`, plus `_norm()` in each store | `ImageBindConfig.normalize` | `True` | L2 | ✅ |
| Embedding dim d | App. A.1 | `--embed_dim` | `--embed_dim` | 1024 | 1024 | ✅ |
| Only the current observation is encoded at inference | §Method | `eval_runner.run_eval_pass()` → `encoder.encode_image(keyframe)` | — | — | — | ✅ |

## 3. Offline loop (base memory construction)

| Paper concept | Paper location | Code location | Flag / key | Default in code | Value in paper | Match? |
|---|---|---|---|---|---|---|
| Preprocessing Φ_pre (filter + index) | App. A.1 | [`pipelines/preprocess_dataset.py`](../pipelines/preprocess_dataset.py), [`pipelines/remove_bad.py`](../pipelines/remove_bad.py) | `--manifest`, `--out_items`, `--frames_root`, `--data_root` | — | — | ✅ (`remove_bad.py` has hard-coded paths) |
| Subject-balanced shuffle | §Dataset | [`pipelines/shuffle_jsonl.py`](../pipelines/shuffle_jsonl.py) | `--seed`, `--mode`, `--key` | seed 42, `balanced`, `subject_id` | — | ✅ |
| Split into base / WB1–WB4 / eval / fine-tune | §Dataset | [`pipelines/split_jsonl.py`](../pipelines/split_jsonl.py) | hard-coded `OUTPUT_RANGES` | base 1–20000, bquery1–3 of 10000 each | base 4%, WB 6/6/6/8%, FT 60%, val 10% | ❌ The script's ranges and file names do not produce the `*_writeback{1..4}`, `*_evaluate` or `*_finetune` manifests that the paper scripts use. |
| Memory builder | App. A.1 | [`pipelines/build_memory.py`](../pipelines/build_memory.py) | `--preprocess_manifest`, `--mem_root`, `--data_root` | — | — | ✅ |
| Base memory seeded from ground truth | App. A.1 | `build_memory.main()` writes `decision="GT"`, `answer_short=answer_gt[:220]`, and the semantic impression from the GT report | — | — | — | ✅ (implicit in the paper) |

## 4. Online loop (memory-conditioned inference)

| Paper concept | Paper location | Code location | Flag / key | Default in code | Value in paper | Match? |
|---|---|---|---|---|---|---|
| Model router `ckpt_is_moe()` → `MoELLaVAInfer` / `LlaVAPhiInfer` | App. A.1 | — | — | — | router function | ❌ `ckpt_is_moe` is not in the repo. `eval_runner.py` always builds `MoELLaVAInfer`. A non-MoE wrapper exists as `LlavaPhiInfer` in [`models/llava_phi_infer.py`](../models/llava_phi_infer.py), but nothing imports it. |
| MoE inference wrapper | App. A.1 | [`models/moe_llava_infer.py`](../models/moe_llava_infer.py): `MoELLaVAInfer` | `--model_path`, `--conv_mode` | `phi` | — | ✅ |
| Top-K cosine retrieval | §Method | `EpisodicStore.query()`, `VisualStore.query()` (inner product on L2-normalized vectors) | — | — | cosine | ✅ |
| k_epi | App. A.1 | `eval_runner.py --k_epi` | `--k_epi` | **4** (not overridden by any script) | **5** (and "Top-K = 10" in the implementation section) | ⚠️ |
| k_vis | App. A.1 | `eval_runner.py --k_vis` | `--k_vis` | **3** (not overridden by any script) | **5** | ⚠️ |
| Retrieval scope | §Method | `query(..., patient_id=owner_id)` | — | Filters to the same patient after over-fetching `k*5` global neighbors | not stated | ⚠️ Undocumented in the paper |
| Prompt fusion P(t) | §Method | [`memory/prompt_builder.py`](../memory/prompt_builder.py): `build_prompt()` | — | — | structured prompt | ✅ |
| ≤ 2 items per memory block | App. A.1 | `_format_hits(max_items=2)` | — | 2 | 2 | ✅ |
| 800 characters per component | App. A.1 | semantic block truncated at 800 chars; episodic and visual hits at `max_chars_per_item=400` | — | 800 / 400 | 800 | ⚠️ 800 applies only to the semantic block |
| Token-budget shrink `_shrink_user_prompt()` | App. A.1 | `models/moe_llava_infer.py`: `_shrink_user_prompt()` | — | budget = `context_len − max_new_tokens − 32` (min 256) | token budget | ✅ |
| Stopping `KeywordsStoppingCriteria` | App. A.1 | `MoELLaVAInfer.generate()` | — | — | — | ✅ |
| Decoding | — | `MoELLaVAInfer.generate()` uses `do_sample=False, num_beams=1` | `--temperature` | 0.2 (ignored, because decoding is greedy) | not stated | ⚠️ |
| Agentic retrieval policy (Eq. 12, memory-query or STOP) | §Method | — | — | — | Eq. 12 | ❌ Not found. Retrieval is a single fixed top-k pass. |

## 5. Outcome-driven writeback

All writeback logic is in [`pipelines/eval_runner.py`](../pipelines/eval_runner.py) (`should_writeback_radiology()` and the `if args.writeback:` block in `run_eval_pass()`).

| Paper concept | Paper location | Flag | Default in code | Used in scripts | Value in paper | Match? |
|---|---|---|---|---|---|---|
| Novelty gate: max cosine < τ | §Update | `--writeback_novelty_thr` | **0.97** | **0.65** | **0.90** | ⚠️ |
| Novelty reference set | §Update | — | top-1 of the **episodic hits for the same patient** | — | all stored embeddings | ⚠️ |
| Length gate | §Update, Eq. 17 | `--writeback_min_impression_chars` | **120 characters** | **30 characters** | **120 words** | ⚠️ Unit differs |
| Keyword gate | §Update | `--writeback_change_keywords`, `--writeback_finding_keywords` | change: `new,increased,decreased,worsened,improved,interval,progression,resolved,stable`; finding: `effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tuberculosis,tb,miliary` | same | examples include "severe", "worse", "reflux" | ⚠️ The paper's example words are not in the code lists |
| Gate combination | §Update | — | `long_enough AND novel AND (has_change OR has_finding)` | — | all three gates | ✅ |
| Capacity per subject | App. A.1 | `--writeback_k_epi`, `--writeback_k_vis` | **200** | **0 (cap disabled)** | **200** | ⚠️ |
| Disable visual writeback | App. A.1 | `--writeback_no_visual` | off | off | flag listed | ✅ |
| Run throttles | App. A.1 | `--writeback_max`, `--writeback_max_per_patient` | 0 (no limit) | not set | flags listed | ✅ |
| Semantic memory updated after **every** interaction | §Update | — | updated **only when the writeback gates pass** | — | every interaction | ⚠️ |
| Episodic summary written back | — | `make_episode_summary()` → `"Q: … \| Impression: …"` (≤ 240 chars) | — | — | — | ✅ |

## 6. Backbones and MoE adaptation

| Paper concept | Paper location | Code / artifact | Value in code / config | Value in paper | Match? |
|---|---|---|---|---|---|
| Base backbone | §Experiments | `LanguageBind/MoE-LLaVA-Phi2-2.7B-4e` (Hugging Face), used by `scripts/moellava_full_ablation.sh` and `scripts/moellava_ablation_writeback.sh` | 4 experts (per the model name) | MoE-LLaVA Base | ✅ |
| FT backbone (regular, non-MoE) | §Experiments | [`checkpoints/MoE-LLaVA-Phi2-finetune-Regular/config.json`](../checkpoints/MoE-LLaVA-Phi2-finetune-Regular/config.json) (`LlavaPhiForCausalLM`), trained by `MoE-LLaVA/scripts/v1/phi2/finetune.sh` from `microsoft/phi-2` | 1 epoch, LR 5e-5, 8 GPUs × batch 1, 3750 steps, `model_max_length` 1024 | FT | ✅ |
| MoE-FT backbone | §Experiments | [`checkpoints/MoE-LLaVA-Phi2-finetune-MoE/config.json`](../checkpoints/MoE-LLaVA-Phi2-finetune-MoE/config.json) (`MoELLaVAPhiForCausalLM`), trained by `MoE-LLaVA/scripts/v1/phi2/finetune_moe.sh`, initialized from the FT checkpoint | — | MoE-FT | ✅ |
| Only router gate W_g trained | App. A.2 | `config.json → moe.train_modules = ["wg"]` | `["wg"]` | W_g only | ✅ (`finetune_moe-multi.sh` uses `["wg","fc2"]`, but no checkpoint from it is in the repo) |
| Experts / top-k | App. A.2, Fig. A2 | `moe.num_experts = 8` (×16 layers), `moe.top_k_experts = 2`, MoE on every other layer (`moe_layers_idx = [0,2,…,30]`) | 8 experts, top-2 | top-2 of 8 | ✅ |
| Other MoE settings | — | `capacity_factor 1.25`, `router_aux_loss_coef 0.01`, `moe_mode sparse`, `use_residual False` | — | not stated | — |
| MoE-FT training run | — | `trainer_state.json`: 30,000 steps, 1 epoch, batch 1, 1 GPU, ≈ 7.5 h | — | 8× A5000 | ⚠️ The MoE-FT run used 1 GPU (`CUDA_VISIBLE_DEVICES=0`). The FT run used 8 GPUs. |

## 7. Metrics

| Paper concept | Paper location | Code location | Match? |
|---|---|---|---|
| Section extraction `extract_section()` | §Experiments | `pipelines/eval_runner.py: extract_section()` (also duplicated in `build_memory.py`) | ✅ |
| Token-level F1 | §Experiments | `token_f1()`: lowercase, collapse whitespace, split on spaces, bag-of-tokens overlap. Returns **1.0 when both sides are empty**. | ✅ (see the empty-section note in KNOWN_ISSUES) |
| Exact Match | §Experiments | `exact_match()` on normalized text | ✅ |
| Full / Impression / Findings | §Experiments | `compute_report_metrics()` → `{em,f1}_{full,impression,findings}` | ✅ |
| Reported scale | Table 1 | Code outputs fractions in [0, 1]. The paper and `scripts/plot/*.csv` report percentages. | ✅ (×100) |

## 8. Experiments → scripts → figures

| Paper result | Run script(s) | Result data in repo | Plot script | Output figure |
|---|---|---|---|---|
| Table 1 (memory ablation at WB4, 3 backbones) | `scripts/moellava_full_ablation.sh` (Base), `scripts/moellava_memory_writeback_regular.sh` (FT), `scripts/moellava_memory_writeback_moe.sh` (MoE-FT) | `scripts/plot/figure2.csv` | `scripts/plot/fig-2.py` | `Figure2_WB4_{F1,EM}_groupedbars.png` |
| Memory scaling WB0→WB4 (Figs. 3, 5, 6) | same scripts (the ablation is re-run after each writeback phase) | `scripts/plot/E.csv`, `EV.csv`, `SEV.csv` | `scripts/plot/fig-1.py` | `subplot{1,2,3}_*_F1.png`, `Figure1_F1_3panels.png` |
| MoE-FT vs. FT gain (Fig. 7) | FT and MoE-FT scripts above | `scripts/plot/figure2.csv` | `scripts/plot/fig-3.py` | `Figure3_WB4_MoE_minus_FT_gain_{F1,EM}.png` |
| Qualitative case study (Fig. 4) | — | — | — | ❌ No script or saved predictions |

> **Gap:** no script in the repo turns the `output/*.json` files written by `eval_runner.py` into the CSVs in `scripts/plot/`. See KNOWN_ISSUES §B.
