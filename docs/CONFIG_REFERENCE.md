# Configuration Reference

The repository has no config file. The old README mentioned `configs.yaml`, but it does not exist. Everything is set with command-line flags, which are listed below as defined in the source code.

## `pipelines/eval_runner.py`: evaluation, ablation and writeback

| Flag | Type | Default | Description |
|---|---|---|---|
| `--items` | str | **required** | preprocessed manifest (`.jsonl`) |
| `--mem_root` | str | **required** | memory root folder |
| `--out_json` | str | **required** | output path (a `.rank<N>.json` suffix is added when running on several GPUs) |
| `--model_path` | str | **required** | HF ID or local checkpoint |
| `--conv_mode` | str | `phi` | MoE-LLaVA conversation template |
| `--load_4bit` / `--load_8bit` | flag | off | quantized load (needs bitsandbytes) |
| `--max_new_tokens` | int | 256 | the scripts use 128 |
| `--temperature` | float | 0.2 | **ignored**, because decoding is greedy |
| `--k_epi` | int | 4 | episodic hits retrieved per query |
| `--k_vis` | int | 3 | visual hits retrieved per query |
| `--embed_dim` | int | 1024 | embedding size |
| `--device` | str | `cuda` | ImageBind device |
| `--video_frames_for_episode` | int | 2 | ImageBind video frames |
| `--limit` | int | 0 | process only the first N items (0 = all) |
| `--start` | int | 0 | skip the first N items |
| `--ablate` | flag | off | run the 8 memory combinations |
| `--ablate_mode` | str | `""` | one of `none, sem_only, epi_only, vis_only, sem_epi, sem_vis, epi_vis, all` |
| `--writeback` | flag | off | enable writeback (use only on WB passes) |
| `--writeback_k_epi` | int | 200 | per-patient episodic cap (0 = no cap) |
| `--writeback_k_vis` | int | 200 | per-patient visual cap (0 = no cap) |
| `--writeback_no_visual` | flag | off | skip visual writeback |
| `--writeback_novelty_thr` | float | 0.97 | write back only if the top-1 episodic similarity is below this value |
| `--writeback_min_impression_chars` | int | 120 | minimum impression length in **characters** |
| `--writeback_change_keywords` | csv str | `new,increased,decreased,worsened,improved,interval,progression,resolved,stable` | change terms |
| `--writeback_finding_keywords` | csv str | `effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tuberculosis,tb,miliary` | finding terms |
| `--writeback_max` | int | 0 | maximum writebacks per run (0 = no limit) |
| `--writeback_max_per_patient` | int | 0 | maximum writebacks per patient per run (0 = no limit) |
| `--use_deepspeed` | flag | off | initialize torch.distributed / DeepSpeed |
| `--local_rank` | int | -1 | set by the launcher |

## `pipelines/build_memory.py`: offline base memory

| Flag | Type | Default | Description |
|---|---|---|---|
| `--preprocess_manifest` | str | **required** | output of `preprocess_dataset.py` |
| `--mem_root` | str | **required** | where the stores are written |
| `--data_root` | str | `""` | dataset root that contains `files/` |
| `--embed_dim` | int | 1024 | |
| `--device` | str | `cuda` | |
| `--video_frames` | int | 2 | |
| `--limit` | int | 0 | 0 = all |
| `--skip_missing` | flag | off | skip missing keyframes instead of raising an error |

## `pipelines/preprocess_dataset.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--manifest` | str | **required** | raw `.jsonl` |
| `--out_items` | str | **required** | output `.jsonl` |
| `--frames_root` | str | **required** | where video frames go |
| `--n_frames` | int | 8 | uniform frames per video |
| `--data_root` | str | `data/datasets/tb_v1` | |
| `--missing_log` | str | `""` | optional `.jsonl` of skipped rows |

## `pipelines/shuffle_jsonl.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--in_jsonl` / `--out_jsonl` | str | **required** | |
| `--seed` | int | 42 | |
| `--key` | str | `subject_id` | grouping key |
| `--mode` | `balanced` \| `global` | `balanced` | round-robin interleave of subjects, or a plain shuffle |

## Scripts with hard-coded settings (no CLI flags)

| Script | Hard-coded values |
|---|---|
| `pipelines/remove_bad.py` | `src=data/datasets/tb_v1/manifest_train1.jsonl`, `dst=data/preprocess/tb_v1/v1/preprocess_manifest_bad_data.jsonl` |
| `pipelines/split_jsonl.py` | `INPUT_FILE=…/preprocess_manifest_train1.jsonl`, `OUTPUT_RANGES` (base 1–20000, bquery1–3) |

## `main_run.py`: single-query demo (image or video)

| Flag | Type | Default |
|---|---|---|
| `--patient_id`, `--question`, `--model_path` | str | **required** |
| `--image` \| `--video` | str | one is required |
| `--conv_mode` | str | `phi` |
| `--load_4bit` / `--load_8bit` / `--use_deepspeed` | flag | off |
| `--max_new_tokens` | int | 256 |
| `--temperature` | float | 0.2 |
| `--mem_root` | str | `data/memory_db` |
| `--tmp_frames` | str | `data/tmp_frames` |
| `--k_epi` / `--k_vis` | int | 4 / 3 |
| `--embed_dim` | int | 1024 |

> `main_run.py` currently fails at the semantic writeback step ([KNOWN_ISSUES D1](KNOWN_ISSUES.md)).

## Constants inside library code (not exposed as flags)

| Constant | Value | Location |
|---|---|---|
| Semantic history `max_history` | 3 | `memory/semantic_store.py` |
| Items per prompt block | 2 | `memory/prompt_builder.py: _format_hits(max_items=2)` |
| Characters per episodic or visual item | 400 | `_format_hits(max_chars_per_item=400)` |
| Characters for the semantic block | 800 | `build_prompt()` |
| Query over-fetch | `k * 5` | `EpisodicStore.query`, `VisualStore.query` |
| Prompt token budget | `context_len − max_new_tokens − 32` (min 256) | `models/moe_llava_infer.py` |
| Middle-cut head / tail chars | 2500 / 2500, then tail 3500 | `_shrink_user_prompt()` |
| Episode summary length | 240 chars | `eval_runner.make_episode_summary()` |
| ImageBind image size / normalization | 224 / CLIP mean-std | `tools/encoder_imagebind.py` |
