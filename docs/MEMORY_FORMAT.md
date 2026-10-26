# Memory Store Format

A memory root (`--mem_root`) contains three stores:

```text
<mem_root>/
├── episodic.jsonl     # M_E metadata, one JSON object per line (append-only)
├── episodic.faiss     # M_E vectors: faiss.IndexFlatIP(1024), row i ↔ line i of episodic.jsonl
├── visual.jsonl       # M_V metadata
├── visual.faiss       # M_V vectors: faiss.IndexFlatIP(1024)
└── semantic/
    └── <patient_id>.json   # M_S, one file per patient
```

- **Similarity.** Vectors are L2-normalized before they are added and before each query, so the inner product equals **cosine similarity**.
- **Rebuild.** If a `.faiss` file is missing but its `.jsonl` exists, the store rebuilds the index from the `z` fields when it loads.
- **Writes.** Every `add()` appends one JSONL line and rewrites the whole `.faiss` file.
- **Snapshots between phases.** The run scripts copy the memory root between phases (`cp -a base/. first-writeback/`, and so on), so each writeback phase has its own folder:

```text
data/memory_db/base/tb_v1/v1/{base,first-writeback,second-writeback,third-writeback,fourth-writeback}/          # MoE-FT
data/memory_db/base/tb_v1/v1/Regular/{base,first-writeback,...,fourth-writeback}/                             # FT
```

## Episodic record (`episodic.jsonl`): `memory/episodic_store.py`

| Field | Type | Meaning | Paper symbol |
|---|---|---|---|
| `episode_id` | str (UUID4) | episode ID | — |
| `patient_id` | str | subject ID (MIMIC `subject_id`) | — |
| `ts` | str (ISO-8601, UTC) | timestamp | t |
| `z` | list[float] × 1024 | ImageBind embedding | z |
| `question` | str | clinical query | q |
| `answer_short` | str | GT answer (≤ 220 chars) for base memory, or `"Q: … \| Impression: …"` (≤ 240-char impression) for writeback | a |
| `decision` | str | `"GT"` for base memory, `"radiology"` for writeback | d |
| `media_path` | str | image path | image path |
| `clip_desc` | str | `offline_base::image` or `eval_writeback:<mode>` | image description |

```json
{"episode_id": "1b9d…", "patient_id": "10000032", "ts": "2026-01-29T07:41:02.118Z",
 "z": [0.0123, -0.0441, "… 1024 floats …"], "question": "What is the impression?",
 "answer_short": "Findings: … Impression: No acute cardiopulmonary process.",
 "decision": "GT", "media_path": "/abs/path/files/p10/…/abc.jpg", "clip_desc": "offline_base::image"}
```

## Visual record (`visual.jsonl`): `memory/visual_store.py`

| Field | Type | Meaning |
|---|---|---|
| `visual_id` | str (UUID4) | visual observation ID |
| `patient_id` | str | subject ID |
| `ts` | str | timestamp |
| `z` | list[float] × 1024 | ImageBind embedding (the same vector as the episodic one for images) |
| `image_path` | str | keyframe path |
| `visual_tag` | str | the manifest's `view` field (for example `PA`, `AP`) for base memory, `"unknown"` for writeback |
| `source_episode_id` | str | links to `episodic.jsonl → episode_id` |

## Semantic record (`semantic/<patient_id>.json`): `memory/semantic_store.py`

```json
{
  "patient_id": "10000032",
  "last_updated": "2026-01-29T07:41:02.118Z",
  "study_impressions": ["No acute cardiopulmonary process.", "Stable small left effusion."],
  "patient_summary": "No acute cardiopulmonary process. Stable small left effusion."
}
```

- `study_impressions` is a rolling buffer of the last `max_history` impressions. **`max_history` is 3 in the code and N = 5 in the paper** (see [KNOWN_ISSUES A5](KNOWN_ISSUES.md)).
- `patient_summary` joins the buffer with spaces. This is the text injected into the prompt, truncated to 800 characters.
- A patient with no history gets `"No prior imaging history available."`.

## Prompt layout (`memory/prompt_builder.py: build_prompt`)

```text
You are a radiology assistant. Answer the question based ONLY on the provided image and context. Use standard radiology report language.

Context Note:
<clip_desc>

Patient Context:
<semantic patient_summary, ≤ 800 chars>

Relevant Prior Episodes:
- Q: … | A: …            (≤ 2 items, ≤ 400 chars each)

Relevant Prior Visual Findings:
- <visual_tag> (ref: <image_path>)   (≤ 2 items)

Question:
<question>

Answer STRICTLY in the following format:
Findings: <describe imaging findings>
Impression: <clinical impression>
```

`MoELLaVAInfer` then prepends the `<image>` token, wraps the prompt in the `phi` conversation template, and shrinks it with `_shrink_user_prompt()` if it exceeds the token budget.
