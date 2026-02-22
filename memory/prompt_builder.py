# memory/prompt_builder.py
from typing import List, Dict, Any


def _unwrap_hit(hit: Any) -> Any:
    """
    Stores often return (item, score). Unwrap to item if possible.
    """
    if isinstance(hit, (tuple, list)) and len(hit) >= 1:
        return hit[0]
    return hit


def _to_text(hit: Any) -> str:
    """
    Convert a retrieval hit into a displayable text snippet.
    Handles dicts, tuples/lists, and raw strings.
    """
    hit = _unwrap_hit(hit)
    if hit is None:
        return ""

    # dict-like
    if isinstance(hit, dict):
        # episodic hit
        q = (hit.get("question") or "").strip()
        a = (hit.get("answer_short") or "").strip()
        if q and a:
            return f"Q: {q} | A: {a}"

        # visual hit
        tag = (hit.get("visual_tag") or "").strip()
        img = (hit.get("image_path") or "").strip()
        if tag and img:
            return f"{tag} (ref: {img})"
        if tag:
            return tag

        # generic fallbacks
        for k in ("text", "content", "caption", "note", "summary", "patient_summary"):
            v = hit.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return str(hit)

    # raw string
    if isinstance(hit, str):
        return hit.strip()

    # fallback
    return str(hit)


def _format_hits(
    hits: List[Any],
    title: str,
    max_items: int = 2,
    max_chars_per_item: int = 400,
) -> str:
    if not hits:
        return ""

    lines = [f"{title}:"]
    n = 0
    for h in hits:
        txt = " ".join(_to_text(h).split())
        if not txt:
            continue

        if len(txt) > max_chars_per_item:
            txt = txt[:max_chars_per_item] + "..."

        lines.append(f"- {txt}")
        n += 1
        if n >= max_items:
            break

    return "\n".join(lines) if n > 0 else ""


def build_prompt(
    semantic: Dict[str, Any],
    episodic_hits: List[Any],
    visual_hits: List[Any],
    question: str,
    clip_desc: str = "",
) -> str:
    sections: List[str] = []

    sections.append(
        "You are a radiology assistant. "
        "Answer the question based ONLY on the provided image and context. "
        "Use standard radiology report language."
    )

    if clip_desc:
        sections.append(f"Context Note:\n{clip_desc.strip()}")

    # ---- semantic memory (your store uses patient_summary) ----
    if semantic:
        s = (semantic.get("patient_summary") or semantic.get("summary") or semantic.get("text") or "").strip()
        if s:
            s = " ".join(s.split())
            if len(s) > 800:
                s = s[:800] + "..."
            sections.append("Patient Context:\n" + s)

    # ---- episodic + visual retrieval ----
    epi_block = _format_hits(episodic_hits, "Relevant Prior Episodes")
    if epi_block:
        sections.append(epi_block)

    vis_block = _format_hits(visual_hits, "Relevant Prior Visual Findings")
    if vis_block:
        sections.append(vis_block)

    # ---- question ----
    sections.append(f"Question:\n{(question or '').strip()}")

    # ---- output format ----
    sections.append(
        "Answer STRICTLY in the following format:\n"
        "Findings: <describe imaging findings>\n"
        "Impression: <clinical impression>"
    )

    return "\n\n".join(sections)
