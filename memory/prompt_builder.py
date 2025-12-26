from typing import List, Tuple, Dict, Any

def build_prompt(
    semantic: Dict[str, Any],
    episodic_hits: List[Tuple[dict, float]],
    visual_hits: List[Tuple[dict, float]],
    question: str,
    clip_desc: str = "",
) -> str:
    stats = semantic["stats"]

    epi_lines = []
    for i, (e, s) in enumerate(episodic_hits, 1):
        epi_lines.append(
            f"{i}) (sim={s:.3f}, ts={e['ts']}) decision={e.get('decision')} | "
            f"Q: {e.get('question','')} | A: {e.get('answer_short','')}"
        )

    vis_lines = []
    for i, (v, s) in enumerate(visual_hits, 1):
        vis_lines.append(
            f"{i}) (sim={s:.3f}, ts={v['ts']}) tag={v.get('visual_tag','unknown')} | "
            f"src_episode={v.get('source_episode_id','')}"
        )

    clip_block = clip_desc.strip()
    if not clip_block:
        clip_block = "(none)"

    return f"""
SYSTEM:
You are a biomedical multimodal agent. Use memory as patient history.
If evidence is insufficient, answer "Uncertain" and say what is missing.

SEMANTIC MEMORY (patient state):
{semantic.get('patient_summary','')}
Stats: total={stats['total']}, adherent={stats['adherent']}, nonadherent={stats['nonadherent']}, ambiguous={stats['ambiguous']}

EPISODIC MEMORY (most similar past sessions):
{chr(10).join(epi_lines) if epi_lines else "(none)"}

VISUAL MEMORY (similar visual prototypes):
{chr(10).join(vis_lines) if vis_lines else "(none)"}

CURRENT OBSERVATION SUMMARY (optional, from video/image perception):
{clip_block}

CURRENT QUESTION:
{question}

RESPONSE FORMAT (must follow):
Decision: <Adherent / Non-adherent / Uncertain>
Evidence: <brief, cite what you saw + relevant memory items>
Uncertainty: <what is missing or ambiguous, if any>
""".strip()
