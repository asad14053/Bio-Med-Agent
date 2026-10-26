# Reproducing the Paper

This guide walks through every experiment in the paper using the scripts that exist in the repo. Each step lists the command, its inputs, and its outputs. Run all commands from the **repository root**.

> **Before you start:** read [KNOWN_ISSUES.md](KNOWN_ISSUES.md), especially §A (parameter differences) and §B (the gap between the committed logs and the published numbers).

**Prerequisites:**

- Environment set up: [INSTALL.md](INSTALL.md)
- Data manifests in place: [DATA.md](DATA.md)
- Checkpoints in place: [MODELS.md](MODELS.md)

**Hardware:**

- Training and evaluation used NVIDIA A5000 GPUs (8× according to the paper).
- The FT training used 8 GPUs. MoE-FT training and every evaluation script use 1 GPU (`CUDA_VISIBLE_DEVICES=0`).

**Determinism:**

- Generation is greedy (`do_sample=False`, `num_beams=1`), and FAISS `IndexFlatIP` search is exact.
- `shuffle_jsonl.py` uses seed 42.
- No global PyTorch or NumPy seed is set. `<TODO: author to confirm the seeds used for training and splitting>`

## Overview

```text
           ┌──────────── offline ────────────┐   ┌──────────────── online, per phase ────────────────┐
manifests → build_memory (base 4%) → snapshot → eval (8 ablations) → writeback pass (WB_i) → snapshot → …
                                                     ▲ Table 1 / Figs 3,5,6 use the eval after each phase
```

Each backbone has one driver script that runs the full sequence:

| Backbone | Driver script | Memory root(s) | Model |
|---|---|---|---|
| Base | `scripts/moellava_full_ablation.sh` | `data/memory_db/base/tb_v1/v1` (single root, no snapshots) | `LanguageBind/MoE-LLaVA-Phi2-2.7B-4e` |
| FT | `scripts/moellava_memory_writeback_regular.sh` | `data/memory_db/base/tb_v1/v1/Regular/{base,first-…,fourth-writeback}` | `checkpoints/MoE-LLaVA-Phi2-finetune-Regular` |
| MoE-FT | `scripts/moellava_memory_writeback_moe.sh` | `data/memory_db/base/tb_v1/v1/{base,first-…,fourth-writeback}` | `checkpoints/MoE-LLaVA-Phi2-finetune-MoE` |

> ⚠️ **Watch for commented-out steps:**
> - `scripts/moellava_memory_writeback_moe.sh` has every step before "4th Ablation – After Writeback3" commented out. It was last used to resume a partial run. Uncomment all steps for a full run.
> - `scripts/moellava_ablation_writeback.sh` is an older partial driver for the Base model.
> - `scripts/moellava_full_ablation.sh` uses the old `ablate1` / `bquery1..3` manifests and runs only three writeback phases.
>
> `<TODO: author to confirm which manifests and how many phases produced the Base column>`

Every driver writes a timestamped log next to itself: `scripts/tb_v1_moellava_pipeline_*.txt`.

## Step 0: Train backbones (optional if you have the weights)

```bash
bash MoE-LLaVA/scripts/v1/phi2/finetune.sh       # → checkpoints/MoE-LLaVA-Phi2-finetune-Regular
bash MoE-LLaVA/scripts/v1/phi2/finetune_moe.sh   # → checkpoints/MoE-LLaVA-Phi2-finetune-MoE
```

## Step 1: Build base memory (offline loop)

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipelines.build_memory \
  --preprocess_manifest data/preprocess/tb_v1/v1/preprocess_manifest_base.jsonl \
  --mem_root data/memory_db/base/tb_v1/v1/base \
  --data_root data/datasets/tb_v1 \
  --device cuda --embed_dim 1024 --video_frames 2
```

**Output:** `episodic.{jsonl,faiss}`, `visual.{jsonl,faiss}` and `semantic/*.json` under `--mem_root`. Base memory is seeded from **ground-truth** reports.

## Step 2: Memory ablation at a given phase (Table 1 rows)

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
  --ablate \
  --items data/preprocess/tb_v1/v1/preprocess_manifest_evaluate.jsonl \
  --mem_root data/memory_db/base/tb_v1/v1/<phase-folder> \
  --out_json output/eval_tb_v1_phi2_4e.ablate_all-m-after-<phase>.json \
  --model_path <backbone> \
  --conv_mode phi --max_new_tokens 128 --temperature 0.2
```

- `--ablate` runs all eight S/E/V combinations: `none, sem_only, epi_only, vis_only, sem_epi, sem_vis, epi_vis, all`.
- To run a single combination, add `--ablate_mode <name>`.
- Writeback is **off** during ablation passes, so memory is not changed.

**Output:** a JSON file `{"runs": [{"summary": {...}, "results": [...]}, ...]}`. The `summary` fields are `f1_impression_avg`, `em_impression_avg`, `f1_full_avg`, `em_full_avg`, `f1_findings_avg` and `em_findings_avg`. They are fractions; multiply by 100 to compare with the paper.

## Step 3: Writeback pass (online loop, WB_i)

Snapshot the previous phase, then run inference on the WB_i manifest with writeback enabled:

```bash
mkdir -p data/memory_db/base/tb_v1/v1/first-writeback
cp -a data/memory_db/base/tb_v1/v1/base/. data/memory_db/base/tb_v1/v1/first-writeback/

CUDA_VISIBLE_DEVICES=0 python -m pipelines.eval_runner \
  --items data/preprocess/tb_v1/v1/preprocess_manifest_writeback1.jsonl \
  --mem_root data/memory_db/base/tb_v1/v1/first-writeback \
  --out_json output/eval_tb_v1_phi2_4e.during-first-writeback.json \
  --model_path <backbone> --conv_mode phi --max_new_tokens 128 \
  --writeback \
  --writeback_k_epi 0 --writeback_k_vis 0 \
  --writeback_novelty_thr 0.65 \
  --writeback_min_impression_chars 30 \
  --writeback_change_keywords "new,increased,decreased,worsened,improved,interval,progression,resolved,stable" \
  --writeback_finding_keywords "effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tb,tuberculosis,miliary"
```

These are the values **used in the scripts**. The paper reports τ = 0.90, a 120-word minimum and a cap of 200 per subject. To run the paper's settings instead, use `--writeback_novelty_thr 0.90 --writeback_k_epi 200 --writeback_k_vis 200`; the word-based length gate has no exact flag. `<TODO: author to confirm which setting produced the published numbers>`

**Output:** each result row includes `writeback.done` and `writeback.reason` (`top1_sim`, `impression_chars`, `novel`, `has_change`, `has_finding`). `summary.writeback_total` gives the number of items written back.

Repeat Step 3 → Step 2 for WB2, WB3 and WB4 (`second-writeback`, `third-writeback`, `fourth-writeback`).

## Experiment 1: Table 1 (memory ablation at WB4, 3 backbones)

1. Run each backbone's driver script through the final "Ablation – After Writeback4" step.
2. For each backbone, read `output/eval_tb_v1_phi2_4e.ablate_all-m-after-fourth-writeback.json` and take `summary.f1_impression_avg` and `summary.em_impression_avg` for each of the 8 modes.
3. The "No memory" row (× × ×) is mode `none`.

> **Output filenames collide.** All three driver scripts write to the same `output/…json` names. Rename or move the `output/` folder between backbones.

The published values are stored in `scripts/plot/figure2.csv`. Regenerate the bar charts with:

```bash
cd scripts/plot && python fig-2.py     # → Figure2_WB4_F1_groupedbars.png, Figure2_WB4_EM_groupedbars.png
```

## Experiment 2: Memory scaling WB0 → WB4 (Figs. 3, 5, 6)

Collect `f1_impression_avg` for modes `epi_only` (E), `epi_vis` (E+V) and `all` (S+E+V) after each phase. The phases are: `none` → after base → after first-… → after fourth-writeback.

The published values are stored in `scripts/plot/E.csv`, `EV.csv` and `SEV.csv`. Regenerate the plots with:

```bash
cd scripts/plot && python fig-1.py     # → subplot1_base_model_F1.png, subplot2_FT_model_F1.png, subplot3_MoE_model_F1.png
```

`Figure1_F1_3panels.png` is a combined three-panel version of the same plots. No script in the repo produces it (`<TODO: author>`).

> The plot scripts need `pandas` and `matplotlib`, which are not in `requirements.txt`: `pip install pandas matplotlib`. All three scripts were checked: they run and regenerate the PNGs from the committed CSVs.

## Experiment 3: MoE-FT vs. regular FT (Fig. 7)

This is computed from the Table 1 data as MoE-FT − FT at WB4. For example, E gives 28.53 − 20.51 = 8.02 and S+E+V gives 54.92 − 47.67 = 7.25.

```bash
cd scripts/plot && python fig-3.py     # → Figure3_WB4_MoE_minus_FT_gain_F1.png (and _EM)
```

## Experiment 4: Qualitative case study (Fig. 4)

`<TODO: author to provide the item IDs and the per-phase predictions>`. The per-item predictions (`answer_pred`) are in the `results` list of each `output/*.json` file.

## Missing pieces

- **Aggregation script.** `<TODO: author>` A script that reads the `output/*.json` files and writes `scripts/plot/*.csv` is missing ([KNOWN_ISSUES B1](KNOWN_ISSUES.md)).
- **Expected runtime.**
  - Evaluation: about 5,000 items × 8 modes per ablation pass on one A5000. `<TODO: author to give the wall-clock time>`
  - Training: FT took ≈ 7.0 h (8 GPUs) and MoE-FT took ≈ 7.5 h (1 GPU), according to `trainer_state.json`.
