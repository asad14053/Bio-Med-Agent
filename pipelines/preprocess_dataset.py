# pipelines/preprocess_dataset.py
import argparse
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.keyframe import extract_uniform_frames


# -------------------------
# IO
# -------------------------
def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# Field helpers (TB + MIMIC compatible)
# -------------------------
def get_owner_id(item: Dict[str, Any]) -> str:
    # TB style: patient_id
    if "patient_id" in item and str(item["patient_id"]).strip():
        return str(item["patient_id"]).strip()
    # MIMIC style: subject_id
    if "subject_id" in item and str(item["subject_id"]).strip():
        return str(item["subject_id"]).strip()
    return "unknown"


def get_media_type(item: Dict[str, Any]) -> str:
    if "media_type" in item and str(item["media_type"]).strip():
        return str(item["media_type"]).strip().lower()
    if "type" in item and str(item["type"]).strip():
        return str(item["type"]).strip().lower()
    raise KeyError("Missing media_type/type")


def get_media_path(item: Dict[str, Any]) -> str:
    if "media_path" in item and str(item["media_path"]).strip():
        return str(item["media_path"]).strip()
    if "path" in item and str(item["path"]).strip():
        return str(item["path"]).strip()
    raise KeyError("Missing media_path/path")


def stable_id_from_path(p: str) -> str:
    h = hashlib.md5(p.encode("utf-8")).hexdigest()[:12]
    return f"item_{h}"


def get_item_id(item: Dict[str, Any], idx: int) -> str:
    # Prefer QA id if present
    if "qa_id" in item and str(item["qa_id"]).strip():
        return str(item["qa_id"]).strip()
    # Otherwise "id"
    if "id" in item and str(item["id"]).strip():
        return str(item["id"]).strip()
    # Fallback to stable id from media_path
    try:
        mp = get_media_path(item)
        return stable_id_from_path(mp)
    except Exception:
        owner = get_owner_id(item)
        return f"{owner}_item{idx:06d}"


# -------------------------
# Path resolver
# -------------------------
def resolve_path(raw_path: str, data_root: str) -> Optional[str]:
    """
    raw_path can be:
      - absolute
      - relative to current working dir
      - relative to data_root (Bio-Med-Agent/data/datasets/tb_v1)
      - stored like 'files/...' (common in MIMIC exports)
    Returns absolute path if exists else None.
    """
    raw_path = str(raw_path).strip()
    if not raw_path:
        return None

    # 1) as-is (absolute or relative to cwd)
    p = Path(raw_path)
    if p.exists():
        return str(p.resolve())

    dr = Path(data_root).resolve() if data_root else None
    if dr:
        # 2) join with dataset root directly
        cand = dr / raw_path
        if cand.exists():
            return str(cand.resolve())

        # 3) sometimes you pass data_root already inside tb_v1,
        #    but manifest stores "files/..." and your data_root contains ".../tb_v1"
        #    -> dr/files/... should exist (this is already covered by #2)
        # 4) if manifest path starts with "files/" but your data_root already IS ".../tb_v1/files"
        #    then join dr / (remove "files/")
        if raw_path.startswith("files/"):
            cand2 = dr / raw_path[len("files/") :]
            if cand2.exists():
                return str(cand2.resolve())

    return None


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # Required
    ap.add_argument("--manifest", required=True, help="Input manifest.jsonl (QA items or dataset items)")
    ap.add_argument("--out_items", required=True, help="Output preprocess_manifest.jsonl")
    ap.add_argument("--frames_root", required=True, help="Where extracted frames will be written")
    ap.add_argument("--n_frames", type=int, default=8, help="Uniform frames per video")

    # New: dataset root so you don't have to change jsonl paths
    ap.add_argument(
        "--data_root",
        default="data/datasets/tb_v1",
        help="Dataset root that contains 'files/'. Example: Bio-Med-Agent/data/datasets/tb_v1",
    )
    ap.add_argument(
        "--missing_log",
        default="",
        help="Optional output jsonl for skipped/missing items",
    )

    args = ap.parse_args()

    items = read_jsonl(Path(args.manifest))
    frames_root = Path(args.frames_root)
    out_items = Path(args.out_items)

    processed: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        try:
            owner_id = get_owner_id(item)
            media_type = get_media_type(item)
            media_path_raw = get_media_path(item)
            item_id = get_item_id(item, idx)
        except Exception as e:
            missing.append({"reason": f"bad_record: {e}", "row_idx": idx, "item": item})
            continue

        # Resolve actual file path
        media_path = resolve_path(media_path_raw, args.data_root)
        if media_path is None:
            missing.append(
                {
                    "reason": "missing_media_file",
                    "row_idx": idx,
                    "item_id": item_id,
                    "owner_id": owner_id,
                    "media_type": media_type,
                    "media_path_raw": media_path_raw,
                    "data_root": args.data_root,
                }
            )
            continue

        rec = dict(item)
        rec["item_id"] = item_id
        rec["owner_id"] = owner_id
        rec["media_path"] = media_path  # overwrite with resolved absolute path (safe downstream)

        rec["preprocess"] = {
            "frames": [],
            "keyframe_for_reasoning": None,
        }

        if media_type == "video":
            frame_dir = frames_root / item_id
            try:
                frames = extract_uniform_frames(video_path=media_path, out_dir=str(frame_dir), n=args.n_frames)
            except Exception as e:
                missing.append(
                    {
                        "reason": f"video_frame_extract_failed: {e}",
                        "row_idx": idx,
                        "item_id": item_id,
                        "media_path": media_path,
                    }
                )
                continue

            if not frames:
                missing.append(
                    {
                        "reason": "video_extracted_zero_frames",
                        "row_idx": idx,
                        "item_id": item_id,
                        "media_path": media_path,
                    }
                )
                continue

            rec["preprocess"]["frames"] = frames
            rec["preprocess"]["keyframe_for_reasoning"] = frames[len(frames) // 2]

        elif media_type == "image":
            # For images, the "keyframe" is the image itself
            rec["preprocess"]["keyframe_for_reasoning"] = media_path

        else:
            missing.append(
                {
                    "reason": f"unknown_media_type:{media_type}",
                    "row_idx": idx,
                    "item_id": item_id,
                    "media_path": media_path,
                }
            )
            continue

        processed.append(rec)

    write_jsonl(out_items, processed)

    if args.missing_log:
        write_jsonl(Path(args.missing_log), missing)

    print("✅ preprocess done")
    print(" - wrote:", out_items)
    print(" - frames_root:", frames_root)
    print(" - kept:", len(processed))
    print(" - skipped:", len(missing))
    if args.missing_log:
        print(" - missing_log:", args.missing_log)


if __name__ == "__main__":
    main()
