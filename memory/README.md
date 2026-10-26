# `memory/`: the MSM-Mem tri-memory store

This folder implements the memory store **M = {M_S, M_E, M_V}** from the paper (§Method, App. A.1). The on-disk formats are documented in [`docs/MEMORY_FORMAT.md`](../docs/MEMORY_FORMAT.md).

| File | Class / function | Paper |
|---|---|---|
| `episodic_store.py` | `EpisodicStore`: `add()`, `query(z, k, patient_id)` | Episodic memory **M_E** (FAISS `IndexFlatIP` + `episodic.jsonl`) |
| `visual_store.py` | `VisualStore`: `add()`, `query(z, k, patient_id)` | Visual memory **M_V** (FAISS `IndexFlatIP` + `visual.jsonl`) |
| `semantic_store.py` | `SemanticStore`: `get()`, `update(patient_id, impression_summary)` | Semantic memory **M_S** and update operator **U_S** (per-patient JSON, rolling buffer `max_history=3`; the paper uses N = 5) |
| `prompt_builder.py` | `build_prompt(semantic, episodic_hits, visual_hits, question, clip_desc)` | Prompt fusion **P(t)**: ≤ 2 items per block, 800-char semantic block, 400 chars per hit |

**Notes:**

- Both vector stores L2-normalize their inputs, so FAISS inner product equals cosine similarity.
- `query()` fetches the `k*5` nearest neighbors across all patients and then keeps only items whose `patient_id` matches.
