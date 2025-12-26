import os
from pathlib import Path

def extract_mid_frame(video_path: str, out_dir: str = "data/tmp_frames") -> str:
    """
    Extracts the middle frame of a video using OpenCV.
    Returns path to saved jpg.

    pip install opencv-python
    """
    import cv2

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fallback: read first frame
        idx = 0
    else:
        idx = total // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from video.")

    name = Path(video_path).stem
    out_path = os.path.join(out_dir, f"{name}_mid.jpg")
    cv2.imwrite(out_path, frame)
    cap.release()
    return out_path
