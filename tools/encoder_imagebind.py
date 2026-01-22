import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import numpy as np


@dataclass
class ImageBindConfig:
    device: str = "cuda"
    video_frames: int = 2
    image_size: int = 224
    normalize: bool = True
    video_frame_strategy: str = "uniform"
    tmp_frames_dir: str = "data/tmp_frames"


class ImageBindEncoder:
    """
    ImageBind encoder that produces 1024-d embeddings.

    Image input expected by ImageBind VISION:
      - image: [B, 3, H, W]  (ImageBind internally pads/repeats to T=2)
      - video: [B, 3, T, H, W] with T=2 for imagebind_huge
    """

    def __init__(self, cfg: Optional[ImageBindConfig] = None):
        self.cfg = cfg or ImageBindConfig()

        import torch
        from PIL import Image
        import torchvision.transforms as T
        from imagebind.models.imagebind_model import imagebind_huge, ModalityType

        self.torch = torch
        self.Image = Image
        self.ModalityType = ModalityType

        self.model = imagebind_huge(pretrained=True).to(self.cfg.device).eval()

        # Standard CLIP-like normalization works well for ImageBind vision.
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
    # Robust image loading
    # -------------------------
    def _load_rgb_pil(self, path: str):
        """
        Always return a PIL Image in RGB mode (3 channels).
        Handles grayscale, LA, palette, RGBA, etc.
        """
        img = self.Image.open(path)

        # Some files are lazy-loaded; force load now so convert is reliable
        img.load()

        # Convert anything to RGB (guarantees 3 channels)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _assert_vision_shape(self, x, expect_video: bool):
        """
        Validate and hard-fail with a helpful message if shape is wrong.
        """
        if expect_video:
            # [B, 3, T, H, W]
            if x.ndim != 5:
                raise RuntimeError(f"[ImageBindEncoder] Expected 5D video tensor [B,3,T,H,W], got {tuple(x.shape)}")
            b, c, t, h, w = x.shape
            if c != 3:
                raise RuntimeError(f"[ImageBindEncoder] Video tensor must have 3 channels. Got C={c} with shape {tuple(x.shape)}")
            if t != self.cfg.video_frames:
                raise RuntimeError(f"[ImageBindEncoder] Video tensor must have T={self.cfg.video_frames}. Got T={t} with shape {tuple(x.shape)}")
            if h != self.cfg.image_size or w != self.cfg.image_size:
                raise RuntimeError(f"[ImageBindEncoder] Video tensor must be {self.cfg.image_size}x{self.cfg.image_size}. Got {h}x{w}")
        else:
            # [B, 3, H, W]
            if x.ndim != 4:
                raise RuntimeError(f"[ImageBindEncoder] Expected 4D image tensor [B,3,H,W], got {tuple(x.shape)}")
            b, c, h, w = x.shape
            if c != 3:
                raise RuntimeError(f"[ImageBindEncoder] Image tensor must have 3 channels. Got C={c} with shape {tuple(x.shape)}")
            if h != self.cfg.image_size or w != self.cfg.image_size:
                raise RuntimeError(f"[ImageBindEncoder] Image tensor must be {self.cfg.image_size}x{self.cfg.image_size}. Got {h}x{w}")

    # -------------------------
    # Public API
    # -------------------------
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Returns [1024] float32.
        Feed ImageBind VISION with [B,3,H,W]. ImageBind will pad to T=2 internally.
        """
        import torch

        img = self._load_rgb_pil(image_path)
        x = self.transform(img)          # [3,224,224]
        x = x.unsqueeze(0)               # [1,3,224,224]
        x = x.to(self.cfg.device, non_blocking=True)

        self._assert_vision_shape(x, expect_video=False)

        with torch.inference_mode():
            out = self.model({self.ModalityType.VISION: x})
            z = out[self.ModalityType.VISION][0].detach().float().cpu().numpy()

        z = z.astype(np.float32).reshape(-1)
        return self._l2norm(z)

    def encode_video(self, video_path: str) -> np.ndarray:
        """
        Returns [1024] float32.
        Feed ImageBind VISION with [B,3,T,224,224] where T=cfg.video_frames (default 2).
        """
        import torch

        frame_paths = self._sample_video_frames(video_path, n=self.cfg.video_frames)
        if len(frame_paths) == 0:
            raise RuntimeError(f"[ImageBindEncoder] Could not sample frames from video: {video_path}")

        # Load + transform frames
        frames = []
        for fp in frame_paths:
            img = self._load_rgb_pil(fp)
            frames.append(self.transform(img))  # [3,224,224]

        # If we got fewer than required frames (cv2 read can fail), pad by repeating last
        if len(frames) < self.cfg.video_frames:
            last = frames[-1]
            while len(frames) < self.cfg.video_frames:
                frames.append(last.clone())

        # If we got more frames than expected, truncate
        frames = frames[: self.cfg.video_frames]

        # [T,3,224,224] -> [1,3,T,224,224]
        x = torch.stack(frames, dim=0)           # [T,3,H,W]
        x = x.permute(1, 0, 2, 3).unsqueeze(0)   # [1,3,T,H,W]
        x = x.to(self.cfg.device, non_blocking=True)

        self._assert_vision_shape(x, expect_video=True)

        with torch.inference_mode():
            out = self.model({self.ModalityType.VISION: x})
            z = out[self.ModalityType.VISION][0].detach().float().cpu().numpy()

        z = z.astype(np.float32).reshape(-1)
        return self._l2norm(z)

    # -------------------------
    # Video frame sampling
    # -------------------------
    def _sample_video_frames(self, video_path: str, n: int) -> List[str]:
        """
        Uniformly sample n frames using OpenCV and save under tmp_frames_dir/<video_stem>/.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"[ImageBindEncoder] Could not open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1

        if n <= 1:
            idxs = [total // 2]
        else:
            idxs = np.linspace(0, total - 1, num=n).astype(int).tolist()

        out_dir = Path(self.cfg.tmp_frames_dir) / Path(video_path).stem
        out_dir.mkdir(parents=True, exist_ok=True)

        out_paths = []
        for i, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Write as jpg (OpenCV uses BGR; PIL will read and we convert to RGB anyway)
            out_path = str(out_dir / f"f_{i:02d}.jpg")
            cv2.imwrite(out_path, frame)
            out_paths.append(out_path)

        cap.release()
        return out_paths
