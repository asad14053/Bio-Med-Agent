import argparse
import os
import re
import socket
from pathlib import Path

import torch
import torch.distributed as dist

from models.moe_llava_infer import MoELLaVAInfer
from tools.keyframe import extract_mid_frame
from tools.encoder_imagebind import ImageBindEncoder, ImageBindConfig

from memory.episodic_store import EpisodicStore
from memory.visual_store import VisualStore
from memory.semantic_store import SemanticStore
from memory.prompt_builder import build_prompt


def short_summary(text: str, max_len: int = 220) -> str:
    s = " ".join(text.strip().split())
    return s[:max_len] + ("..." if len(s) > max_len else "")

def extract_decision(answer: str) -> str:
    m = re.search(r"Decision:\s*(.*)", answer, flags=re.IGNORECASE)
    if not m:
        return "Uncertain"
    d = m.group(1).strip()
    return d if d else "Uncertain"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _init_single_gpu_dist(backend: str = "nccl"):
    """
    Robust single-process init. Safe even if run multiple times.
    """
    if dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
        rank=0,
        world_size=1,
    )


def _maybe_init_deepspeed_comm():
    """
    Initialize DS comm only if MoE-LLaVA inference path requires it.
    """
    try:
        from deepspeed import comm as ds_comm
        if not ds_comm.is_initialized():
            ds_comm.init_distributed(dist_backend="nccl")
    except Exception as e:
        # If DS is not installed or fails, let your model loader decide next steps
        print(f"[WARN] DeepSpeed comm init skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient_id", required=True)
    ap.add_argument("--question", required=True)

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--video", type=str)

    ap.add_argument("--model_path", required=True, help="MoE-LLaVA checkpoint path/id")
    ap.add_argument("--conv_mode", default="phi", help="phi / qwen / stablelm")
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--load_8bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)

    ap.add_argument("--mem_root", type=str, default="data/memory_db")
    ap.add_argument("--tmp_frames", type=str, default="data/tmp_frames")

    ap.add_argument("--k_epi", type=int, default=4)
    ap.add_argument("--k_vis", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=1024)

    # Optional: turn DS on/off from CLI (useful for debugging)
    ap.add_argument("--use_deepspeed", action="store_true", help="Init DS comm for MoE inference")

    args = ap.parse_args()

    Path(args.mem_root).mkdir(parents=True, exist_ok=True)
    Path(args.tmp_frames).mkdir(parents=True, exist_ok=True)

    # Use GPU0 explicitly for single-GPU runs
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # ---- distributed init (single GPU) ----
    _init_single_gpu_dist(backend="nccl")
    if args.use_deepspeed:
        _maybe_init_deepspeed_comm()

    try:
        # 1) Select media mode + keyframe
        if args.image:
            media_path = args.image
            reasoning_image = args.image
            mode = "image"
        else:
            media_path = args.video
            reasoning_image = extract_mid_frame(args.video, out_dir=args.tmp_frames)
            mode = "video"

        # 2) Init encoder
        encoder = ImageBindEncoder(
            ImageBindConfig(device="cuda", video_frames=2, tmp_frames_dir=args.tmp_frames)
        )

        if mode == "image":
            z_episode = encoder.encode_image(media_path)
            z_visual = z_episode
            clip_desc = "Image input."
        else:
            z_episode = encoder.encode_video(media_path)
            z_visual = encoder.encode_image(reasoning_image)
            clip_desc = "Video input (MVP uses mid-frame keyframe for reasoning)."

        # 3) Init memory
        episodic = EpisodicStore(dim=args.embed_dim, root=args.mem_root)
        visual = VisualStore(dim=args.embed_dim, root=args.mem_root)
        semantic = SemanticStore(root=os.path.join(args.mem_root, "semantic"))

        # 4) Retrieve memory
        episodic_hits = episodic.query(z_episode, k=args.k_epi, patient_id=args.patient_id)
        visual_hits = visual.query(z_visual, k=args.k_vis, patient_id=args.patient_id)
        sem_state = semantic.get(args.patient_id)

        # 5) Build prompt
        prompt = build_prompt(
            semantic=sem_state,
            episodic_hits=episodic_hits,
            visual_hits=visual_hits,
            question=args.question,
            clip_desc=clip_desc,
        )

        # 6) Load MoE-LLaVA
        fm = MoELLaVAInfer(
            model_path=args.model_path,
            conv_mode=args.conv_mode,
            device="cuda:0",
            load_4bit=args.load_4bit,
            load_8bit=args.load_8bit,
        )

        # 7) Reason
        answer = fm.generate(
            image_path=reasoning_image,
            user_prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # 8) Writeback
        decision = extract_decision(answer)
        ans_short = short_summary(answer)

        episode_id = episodic.add(
            patient_id=args.patient_id,
            z=z_episode,
            question=args.question,
            answer_short=ans_short,
            decision=decision,
            media_path=media_path,
            clip_desc=clip_desc,
        )

        visual.add(
            patient_id=args.patient_id,
            z=z_visual,
            image_path=reasoning_image,
            source_episode_id=episode_id,
            visual_tag="unknown",
        )

        semantic.update(
            patient_id=args.patient_id,
            episode_summary=ans_short,
            decision=decision,
        )

        print("\n========== FINAL ANSWER ==========\n")
        print(answer)
        print("\n=================================\n")

    finally:
        # Clean shutdown so NCCL doesn't warn/leak
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
