# pipelines/eval_runner.py
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist

from tools.encoder_imagebind import ImageBindEncoder, ImageBindConfig
from memory.episodic_store import EpisodicStore
from memory.visual_store import VisualStore
from memory.semantic_store import SemanticStore
from memory.prompt_builder import build_prompt
from models.moe_llava_infer import MoELLaVAInfer


# ----------------------------
# I/O utils
# ----------------------------
def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# metrics utils
# ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(a: str, b: str) -> float:
    return 1.0 if normalize_text(a) == normalize_text(b) else 0.0


def token_f1(pred: str, gt: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gt).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    from collections import Counter
    cp = Counter(p)
    cg = Counter(g)
    common = sum((cp & cg).values())
    prec = common / max(1, len(p))
    rec = common / max(1, len(g))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def short_summary(text: str, max_len: int = 220) -> str:
    s = " ".join((text or "").strip().split())
    return s[:max_len] + ("..." if len(s) > max_len else "")


def extract_section(text: str, section: str) -> str:
    """
    Extract 'Findings' or 'Impression' section from radiology-style report text.
    Returns "" if section missing.
    """
    t = (text or "").strip()
    m_find = re.search(r"\bfindings\s*:\s*", t, flags=re.IGNORECASE)
    m_impr = re.search(r"\bimpression\s*:\s*", t, flags=re.IGNORECASE)

    sec = section.lower()
    if sec == "findings":
        if not m_find:
            return ""
        start = m_find.end()
        end = m_impr.start() if m_impr and m_impr.start() > start else len(t)
        return t[start:end].strip()

    if sec == "impression":
        if not m_impr:
            return ""
        start = m_impr.end()
        return t[start:].strip()

    return ""


def compute_report_metrics(answer_pred: str, answer_gt: str) -> Dict[str, float]:
    pred_full = answer_pred or ""
    gt_full = answer_gt or ""

    pred_impr = extract_section(pred_full, "impression")
    gt_impr = extract_section(gt_full, "impression")

    pred_find = extract_section(pred_full, "findings")
    gt_find = extract_section(gt_full, "findings")

    return {
        "em_full": exact_match(pred_full, gt_full),
        "f1_full": token_f1(pred_full, gt_full),
        "em_impression": exact_match(pred_impr, gt_impr),
        "f1_impression": token_f1(pred_impr, gt_impr),
        "em_findings": exact_match(pred_find, gt_find),
        "f1_findings": token_f1(pred_find, gt_find),
    }


# ----------------------------
# distributed init (DS-friendly)
# ----------------------------
def init_dist(args: argparse.Namespace) -> bool:
    local_rank = int(getattr(args, "local_rank", -1))
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.local_rank = local_rank

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_dist = bool(args.use_deepspeed or world_size > 1)

    if use_dist and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            import deepspeed

            try:
                deepspeed.init_distributed(dist_backend=backend, auto_mpi_discovery=False)
            except TypeError:
                deepspeed.init_distributed(dist_backend=backend)
        except Exception:
            dist.init_process_group(backend=backend, init_method="env://")

        if dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()

    return use_dist


def shard_items(items: List[Dict[str, Any]], rank: int, world_size: int) -> List[Dict[str, Any]]:
    if world_size <= 1:
        return items
    return items[rank::world_size]


# ----------------------------
# writeback policy helpers
# ----------------------------
def _csv_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def has_any_keyword(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def make_episode_summary(question: str, answer_pred: str, max_len: int = 240) -> str:
    impr = extract_section(answer_pred, "impression")
    core = impr or extract_section(answer_pred, "findings") or (answer_pred or "")
    core = " ".join(core.split())
    if len(core) > max_len:
        core = core[:max_len] + "..."
    q = " ".join((question or "").split())
    return f"Q: {q} | Impression: {core}"


def top1_similarity_from_hits(hits: List[Any]) -> float:
    """
    Expect EpisodicStore.query() -> List[Tuple[dict, score]].
    """
    if not hits:
        return 0.0
    h0 = hits[0]
    if isinstance(h0, (tuple, list)) and len(h0) >= 2 and isinstance(h0[1], (float, int)):
        return float(h0[1])
    return 0.0


def count_patient_entries(meta: List[Dict[str, Any]], patient_id: str) -> int:
    pid = str(patient_id)
    return sum(1 for m in meta if str(m.get("patient_id")) == pid)


def should_writeback_radiology(
    answer_pred: str,
    episodic_hits: List[Any],
    change_keywords: List[str],
    finding_keywords: List[str],
    novelty_thr: float,
    min_impression_chars: int,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Criteria:
      - impression length >= min_impression_chars
      - novelty: top1_score < novelty_thr  (if no hits, novelty passes)
      - and (has-change OR has-important-finding)
    """
    impr = extract_section(answer_pred, "impression")
    impr_norm = " ".join((impr or "").split())

    top1 = top1_similarity_from_hits(episodic_hits)
    is_novel = (top1 < novelty_thr) if episodic_hits else True

    long_enough = len(impr_norm) >= int(min_impression_chars)
    has_change = has_any_keyword(impr_norm, change_keywords) if change_keywords else False
    has_finding = has_any_keyword(impr_norm, finding_keywords) if finding_keywords else False

    keep = long_enough and is_novel and (has_change or has_finding)

    reason = {
        "impression_chars": len(impr_norm),
        "top1_sim": float(top1),
        "long_enough": bool(long_enough),
        "novel": bool(is_novel),
        "has_change": bool(has_change),
        "has_finding": bool(has_finding),
    }
    return keep, reason


# ----------------------------
# eval pass
# ----------------------------
def run_eval_pass(
    items: List[Dict[str, Any]],
    episodic: EpisodicStore,
    visual: VisualStore,
    semantic: SemanticStore,
    encoder: ImageBindEncoder,
    fm: MoELLaVAInfer,
    args: argparse.Namespace,
    mode_tag: str,
    use_sem: bool,
    use_epi: bool,
    use_vis: bool,
) -> Dict[str, Any]:

    results: List[Dict[str, Any]] = []

    em_full_scores: List[float] = []
    f1_full_scores: List[float] = []
    em_impr_scores: List[float] = []
    f1_impr_scores: List[float] = []
    em_find_scores: List[float] = []
    f1_find_scores: List[float] = []

    wb_count_total = 0
    wb_count_by_patient = defaultdict(int)

    change_kw = _csv_list(args.writeback_change_keywords)
    finding_kw = _csv_list(args.writeback_finding_keywords)

    for i, it in enumerate(items):
        qa_id = it.get("qa_id") or it.get("item_id") or f"idx_{i}"
        owner_id = it.get("owner_id") or it.get("subject_id") or it.get("patient_id") or "unknown"
        owner_id = str(owner_id)

        question = it.get("question") or "N/A"
        answer_gt = it.get("answer_gt") or ""

        keyframe = it.get("preprocess", {}).get("keyframe_for_reasoning") or it.get("media_path")
        if not keyframe or not Path(keyframe).exists():
            results.append(
                {
                    "qa_id": qa_id,
                    "owner_id": owner_id,
                    "status": "skipped_missing_keyframe",
                    "keyframe": keyframe,
                    "mode": mode_tag,
                }
            )
            continue

        # Encode image for retrieval
        z = encoder.encode_image(keyframe)

        # Retrieve (ablated)
        episodic_hits = episodic.query(z, k=args.k_epi, patient_id=owner_id) if use_epi else []
        visual_hits = visual.query(z, k=args.k_vis, patient_id=owner_id) if use_vis else []
        sem_state = semantic.get(owner_id) if use_sem else {}

        prompt = build_prompt(
            semantic=sem_state,
            episodic_hits=episodic_hits,
            visual_hits=visual_hits,
            question=question,
            clip_desc=f"Offline QA eval ({mode_tag}) sem={use_sem} epi={use_epi} vis={use_vis}.",
        )

        # Predict
        answer_pred = fm.generate(
            image_path=keyframe,
            user_prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Metrics
        m = compute_report_metrics(answer_pred, answer_gt)
        em_full_scores.append(m["em_full"])
        f1_full_scores.append(m["f1_full"])
        em_impr_scores.append(m["em_impression"])
        f1_impr_scores.append(m["f1_impression"])
        em_find_scores.append(m["em_findings"])
        f1_find_scores.append(m["f1_findings"])

        row = {
            "qa_id": qa_id,
            "owner_id": owner_id,
            "media_type": it.get("media_type"),
            "media_path": it.get("media_path"),
            "keyframe": keyframe,
            "question": question,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
            "answer_pred_short": short_summary(answer_pred),
            "metrics": m,
            "status": "ok",
            "mode": mode_tag,
            "ablation": {"use_sem": use_sem, "use_epi": use_epi, "use_vis": use_vis},
        }

        # ----------------------------
        # Writeback (optional, PER ITEM)
        # IMPORTANT: do NOT use writeback during ablation runs (for fair comparison).
        # ----------------------------
        if args.writeback:
            # persistent caps (per patient, based on current store size)
            epi_n = count_patient_entries(episodic.meta, owner_id)
            vis_n = count_patient_entries(visual.meta, owner_id)

            cap_ok = True
            if args.writeback_k_epi > 0 and epi_n >= args.writeback_k_epi:
                cap_ok = False
            if not args.writeback_no_visual and args.writeback_k_vis > 0 and vis_n >= args.writeback_k_vis:
                cap_ok = False

            keep, reason = should_writeback_radiology(
                answer_pred=answer_pred,
                episodic_hits=episodic_hits,  # novelty computed vs episodic hits
                change_keywords=change_kw,
                finding_keywords=finding_kw,
                novelty_thr=float(args.writeback_novelty_thr),
                min_impression_chars=int(args.writeback_min_impression_chars),
            )

            # run throttles
            if args.writeback_max > 0 and wb_count_total >= args.writeback_max:
                keep = False
                reason["blocked"] = "writeback_max_reached"

            if args.writeback_max_per_patient > 0 and wb_count_by_patient[owner_id] >= args.writeback_max_per_patient:
                keep = False
                reason["blocked"] = "writeback_max_per_patient_reached"

            if not cap_ok:
                keep = False
                reason["blocked"] = "k_cap_reached"

            if keep:
                answer_short = make_episode_summary(question, answer_pred)

                # 1) episodic writeback
                episode_id = episodic.add(
                    patient_id=owner_id,
                    z=z,
                    question=question,
                    answer_short=answer_short,
                    decision="radiology",
                    media_path=str(keyframe),
                    clip_desc=f"eval_writeback:{mode_tag}",
                )

                # 2) semantic writeback (MATCHES your current SemanticStore signature)
                # SemanticStore.update(self, patient_id: str, episode_summary: str, decision: str)
                semantic.update(owner_id, impression_summary= answer_short)

                # 3) visual writeback
                if not args.writeback_no_visual:
                    visual.add(
                        patient_id=owner_id,
                        z=z,
                        image_path=str(keyframe),
                        source_episode_id=episode_id,
                        visual_tag="unknown",
                    )

                wb_count_total += 1
                wb_count_by_patient[owner_id] += 1

                row["writeback"] = {"done": True, "episode_id": episode_id, "reason": reason}
                print(
                    f"[writeback] ok qa_id={qa_id} patient={owner_id} episode_id={episode_id} "
                    f"top1_sim={reason['top1_sim']:.4f} impr_chars={reason['impression_chars']} "
                    f"change={reason['has_change']} finding={reason['has_finding']}",
                    flush=True,
                )
            else:
                row["writeback"] = {"done": False, "reason": reason}

        results.append(row)

        if (i + 1) % 50 == 0:
            print(f"[{mode_tag}] [progress] {i+1}/{len(items)} done", flush=True)

    summary = {
        "mode": mode_tag,
        "ablation": {"use_sem": use_sem, "use_epi": use_epi, "use_vis": use_vis},
        "count_ok": int(sum(1 for r in results if r.get("status") == "ok")),
        "count_total": int(len(results)),
        "em_full_avg": float(np.mean(em_full_scores)) if em_full_scores else 0.0,
        "f1_full_avg": float(np.mean(f1_full_scores)) if f1_full_scores else 0.0,
        "em_impression_avg": float(np.mean(em_impr_scores)) if em_impr_scores else 0.0,
        "f1_impression_avg": float(np.mean(f1_impr_scores)) if f1_impr_scores else 0.0,
        "em_findings_avg": float(np.mean(em_find_scores)) if em_find_scores else 0.0,
        "f1_findings_avg": float(np.mean(f1_find_scores)) if f1_find_scores else 0.0,
        "writeback_total": int(wb_count_total),
    }

    return {"summary": summary, "results": results}


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=-1)

    ap.add_argument("--use_deepspeed", action="store_true")

    ap.add_argument("--items", required=True, help="preprocess_manifest.jsonl")
    ap.add_argument("--mem_root", required=True, help="base memory root, e.g. data/memory_db/base/tb_v1/v1")
    ap.add_argument("--out_json", required=True, help="output results json (rank-suffixed if multi-gpu)")

    # retrieval K for prompt-time retrieval
    ap.add_argument("--k_epi", type=int, default=4)
    ap.add_argument("--k_vis", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=1024)

    # encoder
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video_frames_for_episode", type=int, default=2)

    # VLM
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--conv_mode", default="phi")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)

    # control
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--start", type=int, default=0)

    # ablation
    ap.add_argument("--ablate", action="store_true", help="Run ablations over semantic/episodic/visual memory")

    # ----------------------------
    # Writeback policy (ALL from user args)
    # ----------------------------
    ap.add_argument("--writeback", action="store_true", help="Enable writeback to memory stores")
    ap.add_argument(
    "--ablate_mode",
    type=str,
    default="",
    help="Run a single ablation mode: none | sem_only | epi_only | vis_only | sem_epi | sem_vis | epi_vis | all",
)
    
    # Persistent caps (per patient, based on current store size)
    ap.add_argument("--writeback_k_epi", type=int, default=200, help="Max episodic items per patient (0 disables cap)")
    ap.add_argument("--writeback_k_vis", type=int, default=200, help="Max visual items per patient (0 disables cap)")
    ap.add_argument("--writeback_no_visual", action="store_true", help="Disable visual writeback")

    # Novelty + quality
    ap.add_argument(
        "--writeback_novelty_thr",
        type=float,
        default=0.97,
        help="Skip if top1 episodic similarity >= thr (i.e., not novel).",
    )
    ap.add_argument(
        "--writeback_min_impression_chars",
        type=int,
        default=120,
        help="Min impression chars to write back.",
    )

    # Keywords (comma-separated)
    ap.add_argument(
        "--writeback_change_keywords",
        type=str,
        default="new,increased,decreased,worsened,improved,interval,progression,resolved,stable",
        help="Comma-separated change keywords (checked in Impression).",
    )
    ap.add_argument(
        "--writeback_finding_keywords",
        type=str,
        default="effusion,pneumothorax,consolidation,cavitary,nodule,mass,atelectasis,edema,tuberculosis,tb,miliary",
        help="Comma-separated important finding keywords (checked in Impression).",
    )

    # Optional run throttles
    ap.add_argument("--writeback_max", type=int, default=0, help="Max writebacks per run (0 = no limit)")
    ap.add_argument(
        "--writeback_max_per_patient",
        type=int,
        default=0,
        help="Max writebacks per patient per run (0 = no limit)",
    )

    args = ap.parse_args()

    use_dist = init_dist(args)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print(f"[dist] use_dist={use_dist} world_size={world_size}", flush=True)
    print(
        f"[rank{rank}] local_rank={args.local_rank} cuda={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    print(f"[rank{rank}] running eval_runner from: {__file__}", flush=True)

    out_path = Path(args.out_json)
    if world_size > 1:
        out_path = out_path.with_suffix(f".rank{rank}.json")

    try:
        items = read_jsonl(Path(args.items))
        if args.start > 0:
            items = items[args.start :]
        if args.limit and args.limit > 0:
            items = items[: args.limit]

        items = shard_items(items, rank, world_size)

        episodic = EpisodicStore(dim=args.embed_dim, root=args.mem_root)
        visual = VisualStore(dim=args.embed_dim, root=args.mem_root)
        semantic = SemanticStore(root=os.path.join(args.mem_root, "semantic"))

        encoder = ImageBindEncoder(ImageBindConfig(device=args.device, video_frames=args.video_frames_for_episode))

        fm = MoELLaVAInfer(
            model_path=args.model_path,
            conv_mode=args.conv_mode,
            device=f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu",
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
        )

        if not args.ablate:
            out = run_eval_pass(
                items,
                episodic,
                visual,
                semantic,
                encoder,
                fm,
                args,
                mode_tag="both",
                use_sem=True,
                use_epi=True,
                use_vis=True,
            )
            write_json(out_path, out)
            print(f"\n✅ [rank{rank}] eval done (single mode=both)", flush=True)
            print(" - out:", str(out_path), flush=True)
            print(" - summary:", out["summary"], flush=True)
        else:
            # NOTE: For fair ablations, run with --writeback OFF.
            modes = [
                ("none", False, False, False),
                ("sem_only", True, False, False),
                ("epi_only", False, True, False),
                ("vis_only", False, False, True),
                ("sem_epi", True, True, False),
                ("sem_vis", True, False, True),
                ("epi_vis", False, True, True),
                ("all", True, True, True),
            ]

            if args.ablate_mode:
                modes = [m for m in modes if m[0] == args.ablate_mode]
                if not modes:
                    raise ValueError(f"Unknown --ablate_mode: {args.ablate_mode}")

            bundle = {"runs": []}
            for mode_tag, use_sem, use_epi, use_vis in modes:
                run = run_eval_pass(
                    items,
                    episodic,
                    visual,
                    semantic,
                    encoder,
                    fm,
                    args,
                    mode_tag=mode_tag,
                    use_sem=use_sem,
                    use_epi=use_epi,
                    use_vis=use_vis,
                )
                bundle["runs"].append(run)
                print(f"\n✅ [rank{rank}] completed mode={mode_tag} summary={run['summary']}", flush=True)

            write_json(out_path, bundle)
            print(f"\n✅ [rank{rank}] eval done (ablations)", flush=True)
            print(" - out:", str(out_path), flush=True)

    finally:
        if dist.is_initialized():
            try:
                if torch.cuda.is_available():
                    dist.barrier(device_ids=[torch.cuda.current_device()])
                else:
                    dist.barrier()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
