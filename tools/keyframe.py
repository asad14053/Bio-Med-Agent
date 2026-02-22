# tools/keyframe.py
import os
from pathlib import Path
from typing import List

import numpy as np


def _safe_open_video(video_path: str):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _read_frame_at(cap, idx: int):
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def extract_mid_frame(video_path: str, out_dir: str = "data/tmp_frames") -> str:
    """
    Extract the middle frame of a video.
    Returns: path to saved jpg.
    """
    import cv2

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cap = _safe_open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = (total // 2) if total > 0 else 0

    frame = _read_frame_at(cap, idx)
    if frame is None:
        # fallback: try first frame
        frame = _read_frame_at(cap, 0)
    if frame is None:
        cap.release()
        raise RuntimeError("Failed to read any frame from video.")

    name = Path(video_path).stem
    out_path = os.path.join(out_dir, f"{name}_mid.jpg")
    cv2.imwrite(out_path, frame)

    cap.release()
    return out_path


def extract_uniform_frames(video_path: str, out_dir: str, n: int = 8) -> List[str]:
    """
    Offline preprocess: extract n uniformly spaced frames.

    Returns: list of saved jpg paths.
    Notes:
      - Handles very short videos by de-duplicating indices.
      - If some frames fail to decode, it keeps the ones that work.
    """
    import cv2

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    cap = _safe_open_video(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        # sometimes frame count is unknown; still try first frame
        total = 1

    n = max(1, int(n))
    raw_idxs = np.linspace(0, total - 1, num=n).astype(int).tolist()

    # de-duplicate while preserving order
    idxs = []
    seen = set()
    for idx in raw_idxs:
        if idx not in seen:
            idxs.append(idx)
            seen.add(idx)

    out_paths: List[str] = []
    for i, idx in enumerate(idxs):
        frame = _read_frame_at(cap, idx)
        if frame is None:
            continue
        fp = out_dir_p / f"f_{i:02d}.jpg"
        cv2.imwrite(str(fp), frame)
        out_paths.append(str(fp))

    cap.release()

    # final fallback: ensure at least one frame if possible
    if len(out_paths) == 0:
        cap = _safe_open_video(video_path)
        frame = _read_frame_at(cap, 0)
        if frame is not None:
            fp = out_dir_p / "f_00.jpg"
            cv2.imwrite(str(fp), frame)
            out_paths.append(str(fp))
        cap.release()

    return out_paths
