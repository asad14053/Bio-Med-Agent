# `pipelines/`: offline and online loops, evaluation

Run these from the repository root as modules, for example `python -m pipelines.eval_runner …`. All flags are listed in [`docs/CONFIG_REFERENCE.md`](../docs/CONFIG_REFERENCE.md).

| File | Role | Paper |
|---|---|---|
| `remove_bad.py` | Drop manifest rows that have no question or answer (hard-coded paths) | Preprocessing Φ_pre |
| `shuffle_jsonl.py` | Seeded, subject-balanced shuffle | Dataset |
| `split_jsonl.py` | Line-range splitter. **The ranges are stale** and do not produce the paper's splits ([KNOWN_ISSUES C2](../docs/KNOWN_ISSUES.md)). | Dataset |
| `preprocess_dataset.py` | Resolve media paths, choose `keyframe_for_reasoning`, extract video frames | Φ_pre |
| `build_memory.py` | Encode the base split and write the episodic, visual and semantic stores from ground truth | **Offline loop** (App. A.1) |
| `eval_runner.py` | Retrieve → build prompt → generate → score. Runs the 8-way S/E/V ablation (`--ablate`) and outcome-gated writeback (`--writeback`). Computes F1 and EM on Full, Impression and Findings via `extract_section()`. | **Online loop**, **writeback** (novelty, length and keyword gates), **metrics** |

How these steps are chained for each experiment is described in [`docs/REPRODUCE.md`](../docs/REPRODUCE.md).
