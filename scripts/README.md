# `scripts/`: experiment drivers, logs and figures

## Experiment drivers (run from anywhere; each `cd`s to the repo root)

| Script | Backbone | What it runs |
|---|---|---|
| `moellava_full_ablation.sh` | Base (`LanguageBind/MoE-LLaVA-Phi2-2.7B-4e`) | no-memory eval → build base memory → ablation → 3 × (writeback → ablation). Uses the older `ablate1` / `bquery1–3` manifests. |
| `moellava_ablation_writeback.sh` | Base | A partial or resumed version of the script above |
| `moellava_memory_writeback_regular.sh` | FT (`checkpoints/MoE-LLaVA-Phi2-finetune-Regular`) | no-memory eval → base → ablation → 4 × (snapshot → writeback → ablation) |
| `moellava_memory_writeback_moe.sh` | MoE-FT (`checkpoints/MoE-LLaVA-Phi2-finetune-MoE`) | Same as above. **Most steps are commented out**, so uncomment them for a full run. |

Each driver tees its output to `scripts/tb_v1_moellava_pipeline_<tag>_<timestamp>.txt`. The committed logs are early or partial runs ([KNOWN_ISSUES B2](../docs/KNOWN_ISSUES.md)).

## Demos

| Script | Run from | Notes |
|---|---|---|
| `image-inference.sh` | `scripts/` | Calls `../main_run.py`. The sample image is not included. |
| `video-inference.sh` | repo root | Calls `main_run.py`. Set `VIDEO_PATH`. |

## Figures: `plot/`

| Script | Input | Output | Paper |
|---|---|---|---|
| `fig-2.py` | `figure2.csv` | `Figure2_WB4_{F1,EM}_groupedbars.png` | Table 1 (memory ablation at WB4) |
| `fig-1.py` | `E.csv`, `EV.csv`, `SEV.csv` | `subplot{1,2,3}_*_F1.png` | Memory scaling (Figs. 3, 5, 6) |
| `fig-3.py` | `figure2.csv` | `Figure3_WB4_MoE_minus_FT_gain_{F1,EM}.png` | MoE-FT vs. FT (Fig. 7) |

Run them with `cd scripts/plot && python fig-N.py` (requires `pandas` and `matplotlib`).

## Training scripts

The FT and MoE-FT training scripts are in `MoE-LLaVA/scripts/v1/phi2/` (`finetune.sh`, `finetune_moe.sh`). See [`docs/MODELS.md`](../docs/MODELS.md).
