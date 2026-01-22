# pipelines/build_memory.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.encoder_imagebind import ImageBindEncoder, ImageBindConfig
from memory.visual_store import VisualStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore


import re

def extract_section(text: str, section: str) -> str:
    t = (text or "").strip()
    m_find = re.search(r"\bfindings\s*:\s*", t, flags=re.IGNORECASE)
    m_impr = re.search(r"\bimpression\s*:\s*", t, flags=re.IGNORECASE)

    sec = section.lower()
    if sec == "impression":
        if not m_impr:
            return ""
        return t[m_impr.end():].strip()

    if sec == "findings":
        if not m_find:
            return ""
        start = m_find.end()
        end = m_impr.start() if m_impr and m_impr.start() > start else len(t)
        return t[start:end].strip()

    return ""


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def resolve_path(raw: str, data_root: str) -> Optional[str]:
    """
    Resolve a path from manifest to a real filesystem path.
    Handles cases:
      - raw is absolute
      - raw is relative to cwd
      - raw is relative to data_root (dataset root)
      - raw begins with "files/..." but data_root already points to tb_v1 which contains "files/"
    Returns absolute path if found, else None.
    """
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None

    # 1) as-is (absolute or relative to current working directory)
    p = Path(raw)
    if p.exists():
        return str(p.resolve())

    # 2) join with dataset root
    if data_root:
        dr = Path(data_root)

        cand = dr / raw
        if cand.exists():
            return str(cand.resolve())

        # 3) if raw starts with "files/", try dropping it
        if raw.startswith("files/"):
            cand2 = dr / raw[len("files/"):]
            if cand2.exists():
                return str(cand2.resolve())

    return None


def short_text(x: str, n: int = 220) -> str:
    x = (x or "").strip()
    if len(x) <= n:
        return x
    return x[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess_manifest", required=True)
    ap.add_argument("--mem_root", required=True, help="e.g. data/memory_db/base/tb_v1/v1")
    ap.add_argument("--data_root", default="", help="e.g. data/datasets/tb_v1 (root that contains files/)")
    ap.add_argument("--embed_dim", type=int, default=1024)

    # encoder config
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video_frames", type=int, default=2)

    # control
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--skip_missing", action="store_true", help="Skip missing files instead of crashing")
    args = ap.parse_args()

    items = read_jsonl(Path(args.preprocess_manifest))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    mem_root = Path(args.mem_root)
    mem_root.mkdir(parents=True, exist_ok=True)

    # Stores (versioned)
    episodic = EpisodicStore(dim=args.embed_dim, root=str(mem_root))
    visual = VisualStore(dim=args.embed_dim, root=str(mem_root))
    semantic = SemanticStore(root=str(mem_root / "semantic"))

    # Encoder once
    encoder = ImageBindEncoder(ImageBindConfig(device=args.device, video_frames=args.video_frames))

    n_ok = 0
    n_skip = 0
    n_missing = 0

    for it in items:
        item_id = it.get("item_id") or it.get("qa_id") or it.get("id")
        if not item_id:
            n_skip += 1
            continue

        media_type = (it.get("media_type") or it.get("type") or "").lower().strip()
        keyframe_raw = it.get("preprocess", {}).get("keyframe_for_reasoning")
        media_raw = it.get("media_path") or it.get("path")

        # Resolve paths using data_root
        keyframe = resolve_path(keyframe_raw, args.data_root) if keyframe_raw else None
        media_path = resolve_path(media_raw, args.data_root) if media_raw else None

        if not media_type or not keyframe or not media_path:
            # if keyframe/media missing, skip
            n_skip += 1
            continue

        if not Path(keyframe).exists():
            n_missing += 1
            if args.skip_missing:
                continue
            raise FileNotFoundError(f"Keyframe not found: {keyframe}")

        subject_id = str(it.get("subject_id") or it.get("patient_id") or it.get("owner_id") or "unknown").strip()
        question = it.get("question") or "What is the impression?"
        answer_gt = it.get("answer_gt") or ""

        # ----- Encode -----
        # For base memory we index by keyframe embedding (fast, consistent)
        if media_type in ("image", "video"):
            z = encoder.encode_image(keyframe)
        else:
            n_skip += 1
            continue

        # ----- Write episodic + visual -----
        episode_id = episodic.add(
            patient_id=subject_id,
            z=z,
            question=question,
            answer_short=short_text(answer_gt, 220),
            decision="GT",
            media_path=media_path,
            clip_desc=f"offline_base::{media_type}",
        )

        visual.add(
            patient_id=subject_id,
            z=z,
            image_path=keyframe,
            source_episode_id=episode_id,
            visual_tag=str(it.get("view", "unknown")),
        )

        # ----- Write semantic (GT-based, deterministic) -----
        # Store rolling summary/stats per subject_id
        # Radiology semantic writeback: store study-level impression text only
        report_text = it.get("answer_gt") or it.get("report") or ""
        impr = " ".join(extract_section(report_text, "impression").split())
        if impr:
            semantic.update(patient_id=str(subject_id), impression_summary=impr)


        n_ok += 1
        if n_ok % 200 == 0:
            print(f"processed: {n_ok}")

    print("\n✅ build_base_memory done (GT semantic enabled)")
    print(f" - mem_root: {mem_root}")
    print(f" - added: {n_ok}")
    print(f" - skipped: {n_skip}")
    print(f" - missing_files: {n_missing}")


if __name__ == "__main__":
    main()
