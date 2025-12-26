import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ImageBindConfig:
    device: str = "cuda"
    video_frames: int = 2          # matches your ImageBindModel default + PadIm2Video repeat ntimes=2
    image_size: int = 224
    normalize: bool = True         # embeddings are already normalized by ImageBind postprocessor, but keep safe
    video_frame_strategy: str = "uniform"  # uniform sampling


class ImageBindEncoder:
    """
    Real ImageBind encoder using your pasted ImageBindModel (imagebind_model.py).

    Produces:
      - encode_image(image_path) -> np.ndarray [1024]
      - encode_video(video_path) -> np.ndarray [1024]   (sample T frames, stack, forward, average if needed)

    Expected VISION input tensor shape for your model:
      [B, 3, T, 224, 224]
    """

    def __init__(self, cfg: Optional[ImageBindConfig] = None):
        self.cfg = cfg or ImageBindConfig()

        # Import your ImageBind code (adjust module path if needed)
        import torch
        from PIL import Image
        import torchvision.transforms as T

        # This import must match where your pasted file lives.
        # If your repo structure is imagebind/models/imagebind_model.py, this is correct:
        from imagebind.models.imagebind_model import imagebind_huge, ModalityType

        self.torch = torch
        self.Image = Image
        self.ModalityType = ModalityType

        self.model = imagebind_huge(pretrained=True).to(self.cfg.device).eval()

        # Standard CLIP-like normalization works fine for ImageBind vision path.
        self.transform = T.Compose([
            T.Resize(self.cfg.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(self.cfg.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _l2norm(self, x: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return x
        n = np.linalg.norm(x) + 1e-12
        return x / n

    # -------------------------
    # Public API
    # -------------------------
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Returns [1024] float32.
        IMPORTANT: For ImageBind vision, pass [B,3,H,W] (4D).
        The model's PadIm2Video will expand it to T=2 internally.
        """
        import torch

        img = self.Image.open(image_path).convert("RGB")
        x = self.transform(img)         # [3,224,224]
        x = x.unsqueeze(0)              # [1,3,224,224]  <-- 4D ONLY
        x = x.to(self.cfg.device)

        with torch.inference_mode():
            out = self.model({self.ModalityType.VISION: x})
            z = out[self.ModalityType.VISION][0].detach().float().cpu().numpy()  # [1024]

        z = z.astype(np.float32).reshape(-1)
        return self._l2norm(z)



    def encode_video(self, video_path: str) -> np.ndarray:
        """
        Returns [1024] float32.
        For video, pass [B,3,T,224,224] (5D). Use exactly T=2 frames for your model.
        """
        import torch

        frame_paths = self._sample_video_frames(video_path, n=self.cfg.video_frames)
        if len(frame_paths) == 0:
            raise RuntimeError(f"Could not sample frames from video: {video_path}")

        frames = []
        for fp in frame_paths:
            img = self.Image.open(fp).convert("RGB")
            frames.append(self.transform(img))  # [3,224,224]

        # frames: list of [3,224,224] length T
        x = torch.stack(frames, dim=0)          # [T,3,224,224]
        x = x.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,224,224]
        x = x.to(self.cfg.device)

        with torch.inference_mode():
            out = self.model({self.ModalityType.VISION: x})
            z = out[self.ModalityType.VISION][0].detach().float().cpu().numpy()  # [1024]

        z = z.astype(np.float32).reshape(-1)
        return self._l2norm(z)


    # -------------------------
    # Video frame sampling
    # -------------------------
    def _sample_video_frames(self, video_path: str, n: int) -> List[str]:
        """
        Uniformly sample n frames using OpenCV and save under data/tmp_frames/<video_stem>/.
        """
        import cv2
        from pathlib import Path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1

        if n <= 1:
            idxs = [total // 2]
        else:
            if self.cfg.video_frame_strategy == "uniform":
                idxs = np.linspace(0, total - 1, num=n).astype(int).tolist()
            else:
                # fallback uniform
                idxs = np.linspace(0, total - 1, num=n).astype(int).tolist()

        out_dir = Path("data/tmp_frames") / Path(video_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        out_paths = []
        for i, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = str(out_dir / f"f_{i:02d}.jpg")
            cv2.imwrite(out_path, frame)
            out_paths.append(out_path)

        cap.release()
        return out_paths
