# Data

## Source

MSM-Mem is evaluated on a **manually curated subset of MIMIC-CXR** (Johnson et al., 2019). The subset contains chest radiographs, radiology reports, clinical captions and reasoning-oriented QA annotations.

- MIMIC-CXR requires **PhysioNet credentialed access** and a signed data use agreement: <https://physionet.org/content/mimic-cxr/>.
- **This repository does not redistribute any MIMIC-CXR data** (images, reports, or derived manifests that contain report text).

> In the code and scripts the dataset is called **`tb_v1`** (for example `data/datasets/tb_v1`). `<TODO: author to confirm that tb_v1 is the curated MIMIC-CXR subset and describe how it was curated>` (see [KNOWN_ISSUES C1](KNOWN_ISSUES.md)).

## Expected directory layout

`data/` is git-ignored. The scripts expect this layout:

```text
data/
├── datasets/tb_v1/                       # --data_root
│   ├── files/...                         # images (MIMIC-style "files/pXX/pXXXXXXX/sXXXXXXXX/*.jpg")
│   └── manifest_train1.jsonl             # raw manifest (read by pipelines/remove_bad.py)
├── preprocess/tb_v1/v1/                  # preprocessed manifests (one per split)
│   ├── preprocess_manifest_finetune.jsonl    # FT / MoE-FT training (60%)
│   ├── preprocess_manifest_base.jsonl        # base memory (4%)
│   ├── preprocess_manifest_writeback1.jsonl  # WB1 (6%)
│   ├── preprocess_manifest_writeback2.jsonl  # WB2 (6%)
│   ├── preprocess_manifest_writeback3.jsonl  # WB3 (6%)
│   ├── preprocess_manifest_writeback4.jsonl  # WB4 (8%)
│   └── preprocess_manifest_evaluate.jsonl    # held-out evaluation (10%)
├── memory_db/base/tb_v1/v1/...           # memory stores (see MEMORY_FORMAT.md)
└── tmp_frames/                           # temporary keyframes (video mode only)
```

## Manifest record format

Each line of a raw manifest is one QA item. The fields below are the ones the code reads. Alternative names are accepted in the order shown.

| Field | Alternatives | Used by | Notes |
|---|---|---|---|
| `qa_id` | `id`, `item_id`; otherwise an MD5 of `media_path` | all | unique item ID |
| `subject_id` | `patient_id`, `owner_id` | all | patient key for memory retrieval |
| `media_type` | `type` | preprocess, build | `image` (CXR) or `video` |
| `media_path` | `path` | preprocess, build | absolute, relative to CWD, or relative to `--data_root` (a leading `files/` is handled) |
| `question` | — | build, eval | defaults to `"What is the impression?"` in `build_memory.py` |
| `answer_gt` | `answer`, `report` | build, eval | ground-truth report text containing `Findings:` and `Impression:` sections |
| `view` | — | build | stored as the `visual_tag` in visual memory |

`pipelines/preprocess_dataset.py` adds `item_id`, `owner_id`, an absolute `media_path`, and `preprocess.keyframe_for_reasoning`. For images, `keyframe_for_reasoning` is the image itself.

Example (with synthetic values):

```json
{"qa_id": "q_000001", "subject_id": "10000032", "media_type": "image",
 "media_path": "files/p10/p10000032/s50414267/abc.jpg", "view": "PA",
 "question": "Describe the findings and impression for this chest radiograph.",
 "answer_gt": "Findings: ... Impression: No acute cardiopulmonary process."}
```

## Splits

The paper splits the data at the subject level:

| Split | Share | File | Items (if the pool is 50k) | Evidence in repo |
|---|---|---|---|---|
| Fine-tuning (FT and MoE-FT) | 60% | `preprocess_manifest_finetune.jsonl` | 30,000 | MoE-FT `trainer_state.json`: 30,000 steps at batch 1 |
| Base memory | 4% | `preprocess_manifest_base.jsonl` | 2,000 | script comment "Build Base Memory (1-2000)" |
| WB1 | 6% | `preprocess_manifest_writeback1.jsonl` | 3,000 | logs: `count_total: 3000` |
| WB2 | 6% | `preprocess_manifest_writeback2.jsonl` | 3,000 | logs: `count_total: 3000` |
| WB3 | 6% | `preprocess_manifest_writeback3.jsonl` | 3,000 | logs: `count_total: 3000` |
| WB4 | 8% | `preprocess_manifest_writeback4.jsonl` | 4,000 | logs: `count_total: 4000` |
| Evaluation / validation | 10% | `preprocess_manifest_evaluate.jsonl` | 5,000 | logs: `count_total: 5000` |

> ⚠️ **No script in the repo produces these files.** `pipelines/split_jsonl.py` has hard-coded ranges (base 1–20000, `bquery1..3`) left over from an earlier setup. `<TODO: author to add the split script, or the subject-ID lists for each split, plus the random seed>` (see [KNOWN_ISSUES C2–C3](KNOWN_ISSUES.md)).

## Preprocessing commands (as used by the authors)

```bash
# 1) drop rows without question/answer  (hard-coded paths inside the script)
python -m pipelines.remove_bad

# 2) optional: subject-balanced shuffle
python -m pipelines.shuffle_jsonl --in_jsonl <in.jsonl> --out_jsonl <out.jsonl> --seed 42 --mode balanced --key subject_id

# 3) resolve paths and pick keyframes
python -m pipelines.preprocess_dataset \
  --manifest <raw_manifest.jsonl> \
  --out_items data/preprocess/tb_v1/v1/preprocess_manifest_<split>.jsonl \
  --frames_root data/tmp_frames \
  --data_root data/datasets/tb_v1 \
  --missing_log data/preprocess/tb_v1/v1/missing_<split>.jsonl

# 4) split into the seven manifests above
#    <TODO: author to provide the script>
```
