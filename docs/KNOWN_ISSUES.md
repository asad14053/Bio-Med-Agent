# Known Issues and Open Questions

This file records every mismatch, missing piece and bug found while preparing this documentation. **No code was changed to fix any of them.** Items marked **[AUTHOR]** need a decision or an artifact from the authors.

## A. Paper ↔ code parameter mismatches

| # | Item | Paper | Code default | Value used in `scripts/*.sh` | Where |
|---|---|---|---|---|---|
| A1 | Novelty threshold τ | 0.90 | 0.97 | 0.65 | `pipelines/eval_runner.py --writeback_novelty_thr` |
| A2 | Minimum impression length | 120 **words** | 120 **characters** | 30 characters | `--writeback_min_impression_chars` |
| A3 | Writeback capacity per subject | 200 | 200 | **0 (disabled)** | `--writeback_k_epi`, `--writeback_k_vis` |
| A4 | k_epi / k_vis | 5 / 5 (and "Top-K = 10" in the implementation section) | 4 / 3 | not set, so 4 / 3 | `--k_epi`, `--k_vis` |
| A5 | Semantic history N | 5 | 3 | not configurable | `SemanticStore(max_history=3)` |
| A6 | Characters per prompt component | 800 | 800 (semantic) / 400 (each episodic or visual hit) | — | `memory/prompt_builder.py` |
| A7 | Keyword examples | "severe", "worse", "new", "effusion", "reflux" | change and finding lists without "severe", "worse" or "reflux" | same as code | `--writeback_*_keywords` |
| A8 | Semantic update timing | after **every** interaction | only when all writeback gates pass | — | `run_eval_pass()` |
| A9 | Novelty reference set | max similarity to **all** stored embeddings | top-1 of the **same-patient** episodic hits (top `k*5` global neighbors, then filtered by patient) | — | `should_writeback_radiology()` |
| A10 | Semantic update content | impression **+ decision** | impression only | — | `SemanticStore.update()` |
| A11 | `temperature` | — | passed through, but `MoELLaVAInfer` decodes greedily (`do_sample=False`), so it has no effect | 0.2 | `models/moe_llava_infer.py` |

**[AUTHOR]** For each row, confirm which value produced the published numbers. Then either update the paper or record the "paper setting" in `docs/REPRODUCE.md`.

## B. Reproducing the published numbers

- **B1. No aggregation script.** `scripts/plot/figure2.csv`, `E.csv`, `EV.csv` and `SEV.csv` hold the numbers in Table 1 and the figures, but nothing in the repo builds them from the `output/*.json` files that `eval_runner.py` writes. **[AUTHOR]** Add the script, or the raw output JSONs, or both.
- **B2. The committed run logs do not match the published numbers.** The logs in `scripts/tb_v1_moellava_pipeline_*.txt` are the only raw results in the repo. In the MoE-FT logs (`…all-moe_20260129/30/31.txt`), all eight ablation modes after WB3 and WB4 report impression F1 ≈ 0.19 (19%), with almost no difference between modes. For comparison, `SEV.csv` reports 43.47 (WB3) and 54.92 (WB4) for S+E+V on MoE-FT. The same logs report `writeback_total` values of 0, 0, 1 and 0 for the WB1–WB4 writeback passes, so memory barely grew in those runs. These may be earlier or debugging runs. **[AUTHOR]** Commit the logs or outputs of the runs used in the paper, or explain how they differ.
- **B3. Empty-section effect on Impression metrics.** `token_f1()` and `exact_match()` both return 1.0 when the predicted and ground-truth sections are **both empty**. In several committed logs `em_impression_avg == f1_impression_avg` (for example 0.1914 / 0.1914), which suggests that most Impression scores come from items where neither side has an `Impression:` section. **[AUTHOR]** Report how many evaluation items have a non-empty GT impression, or score only those items.
- **B4. FT (non-MoE) evaluation path is not wired.** The paper describes a router `ckpt_is_moe()` that dispatches to `MoELLaVAInfer` or `LlaVAPhiInfer`. `ckpt_is_moe` does not exist, and `eval_runner.py` always builds `MoELLaVAInfer`. `models/llava_phi_infer.py: LlavaPhiInfer` exists but nothing imports it. All three committed FT logs (`…all-Regular_*.txt`) end with `RuntimeError: Regular LlavaPhi generate failed … model_kwargs are not used by the model: ['image']`. **[AUTHOR]** Say which code version produced the FT column.
- **B5. Retrieval policy (Eq. 12)** is not implemented. Retrieval is one fixed top-k pass.
- **B6. Qualitative case study (Fig. 4).** There is no script or saved predictions.
- **B7. Hallucination.** The paper's introduction says MSM-Mem reduces hallucination, but no hallucination metric exists in the paper or the code. The README does not repeat this claim.

## C. Data and splits

- **C1. Dataset naming.** The paper uses a curated **MIMIC-CXR** subset. Every path in the code and scripts says `data/datasets/tb_v1` and `preprocess/tb_v1`, and the keyword lists include TB terms (`tuberculosis`, `tb`, `miliary`, `cavitary`). **[AUTHOR]** Confirm that `tb_v1` is the MIMIC-CXR subset (or explain the difference) and how it was curated (QA generation, filtering, report-to-QA conversion).
- **C2. The split script does not produce the paper's splits.** `pipelines/split_jsonl.py` hard-codes `base` = lines 1–20000 and `bquery1–3` = 10000 lines each. The paper scripts instead read `preprocess_manifest_{base,writeback1..4,evaluate,finetune}.jsonl`. Comments in the scripts give base 1–2000 and writeback ranges of 3000 each. The eval logs show 5000 eval items and 4000 WB4 items. These sizes match the paper **if the curated pool is 50,000 items**: base 4% = 2000, WB1–3 6% = 3000 each, WB4 8% = 4000, eval 10% = 5000, fine-tune 60% = 30000 (which matches the 30,000 MoE-FT training steps). **[AUTHOR]** Add the split script that produced these manifests, or commit the subject-ID lists.
- **C3. Subject-level split.** The paper says the splits are subject-level. Nothing in the repo performs subject-disjoint splitting. `shuffle_jsonl.py --mode balanced` interleaves subjects but does not keep them disjoint across splits.
- **C4. Hard-coded paths.** `pipelines/remove_bad.py` and `pipelines/split_jsonl.py` take no CLI arguments.
- **C5. Data access.** MIMIC-CXR requires PhysioNet credentialing. Its data use agreement forbids sharing the data, including derived manifests that contain report text. **[AUTHOR]** Decide what can be released, for example subject IDs plus a generation script.

## D. Bugs found (recorded, not fixed)

- **D1. `main_run.py` crashes at writeback.** It calls `semantic.update(patient_id=…, episode_summary=…, decision=…)`, but `SemanticStore.update()` has the signature `(patient_id, impression_summary)`. This raises `TypeError` (the same error is in `scripts/tb_v1_moellava_pipeline_20260113_155350.txt`). The batch pipeline (`eval_runner.py`) calls it correctly.
- **D2.** A comment in `eval_runner.py` (around line 397) describes the old `SemanticStore.update` signature.
- **D3.** `scripts/image-inference.sh` must be run from inside `scripts/` (it uses `../main_run.py`), but `scripts/video-inference.sh` must be run from the repo root (it uses `main_run.py`). Both reference media files that are not in the repo.
- **D4.** `EpisodicStore.query` and `VisualStore.query` over-fetch `k*5` global neighbors and then filter by patient. A patient whose items are not in that global set gets fewer than `k` hits, or none.
- **D5.** `MoE-LLaVA/scripts/v1/phi2/finetune_regular-multi.sh` is byte-for-byte identical to `finetune_moe-multi.sh` (an MoE run). The regular FT checkpoint was produced by `finetune.sh`. `finetune_moe-multi.log` ends in an NCCL abort.
- **D6.** `MoE-LLaVA/scripts/v1/phi2/finetune.sh` sets `CUDA_VISIBLE_DEVICES=0` but launches `--num_gpus=8`. The log shows that 8 ranks ran.

## E. Environment and dependencies

- **E1. The vendored MoE-LLaVA is modified.** Commit `db292e7` changed `MoE-LLaVA/moellava/model/builder.py` (+119/−) and `MoE-LLaVA/moellava/train/train.py` (+113/−). However, `requirements.txt` and `environment.yaml` install **upstream** MoE-LLaVA from GitHub (`git+https://github.com/PKU-YuanGroup/MoE-LLaVA.git`), which does not have these changes. To reproduce, install the local copy (`pip install -e ./MoE-LLaVA`). See [INSTALL.md](INSTALL.md).
- **E2. No version pins.** `torch`, `transformers`, `deepspeed` and the other packages have no versions. MoE-LLaVA is sensitive to the `transformers` and `deepspeed` versions. **[AUTHOR]** Add a `pip freeze` from the machine that ran the experiments.
- **E3. Environment name.** The old README used `environment.yml` and the env name `moellava`. The actual file is `environment.yaml`, which creates an env named `Bio-Med-Agent`. The run scripts' `ENV_NAME=moellava` is informational only.
- **E4. `flash-attention` is a broken submodule.** It is recorded as a git link to commit `7b0bfcc3d1f6…`, but there is no `.gitmodules`, so `git clone --recursive` leaves an empty folder. No first-party code imports `flash_attn`. **[AUTHOR]** Add a `.gitmodules` entry or remove the git link.
- **E5. Missing pretrain projector.** Both fine-tune scripts need `checkpoints/MoE-LLaVA-Phi2-Pretrain/mm_projector.bin`, which is not in the repo and has no documented download location.
- **E6. Checkpoint weights are not in the repo.** `checkpoints/*/` contains configs, tokenizers, `trainer_state.json` and TensorBoard events, but no `*.safetensors` shards (they are ignored by `.gitignore`). **[AUTHOR]** Publish the weights (for example on the Hugging Face Hub) and add the links to [MODELS.md](MODELS.md).

- **E7.** The plot scripts (`scripts/plot/fig-*.py`) need `pandas` and `matplotlib`, which are not listed in `requirements.txt`. `fig-2.py` and `fig-3.py` use the category label `"None"`, but `figure2.csv` uses `"No Memory"`, which pandas 3.x reports with a deprecation warning.

## F. Repository hygiene

- **F1. No license.** The repo has no top-level `LICENSE`. **[AUTHOR]** Choose one. The choice is constrained by these components:
  - MoE-LLaVA is Apache-2.0 (see `MoE-LLaVA/LICENSE`).
  - ImageBind code and weights are **CC-BY-NC-SA 4.0**, which is non-commercial and share-alike (see `ImageBind/LICENSE`).
  - MIMIC-CXR is under the PhysioNet Credentialed Health Data License, and no data may be redistributed.
  - Phi-2 is MIT.
- **F2. Committed build and log artifacts.** These include `MoE-LLaVA/moellava.egg-info/`, a 12 MB `finetune_moe.log`, TensorBoard event files under `checkpoints/*/runs/`, and run logs in `scripts/`. They were left in place. Moves are proposed in [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md).
- **F3. Name aliases.** The paper uses both "MSM-Mem" and "Bio-Mem", and the repository is named "Bio-Med-Agent". The docs use **MSM-Mem**.
- **F4. Personal paths in logs.** Absolute paths such as `/home/myid/<user>/…` and the hostname appear in the committed logs. Decide whether to keep them in a public release.

## G. Limitations stated in the paper

The paper itself lists these limitations:

- Writeback errors can propagate and cascade.
- Retrieval can surface irrelevant prior cases, outdated patient states, or partial embedding matches, which can lead to hallucination.
- Memory can drift as clinical patterns change, which may need human review.
- Latent-space misalignment, unstable retrieval, and saturated writeback windows are possible.
