# models/llava_phi_infer.py
import os
import json
import torch
from PIL import Image

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor

from moellava.mm_utils import tokenizer_image_token
from moellava.constants import DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates

class LlavaPhiInfer:
    """
    Regular (non-MoE) LlavaPhi inference wrapper.
    This does NOT touch MoE code paths. It loads the checkpoint as declared by its config:
    architectures=['LlavaPhiForCausalLM'], model_type='llava_phi'
    """
    def __init__(
        self,
        model_path: str,
        conv_mode: str = "phi",
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.model_path = model_path
        self.device = device

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

        # Model (loads LlavaPhiForCausalLM if trust_remote_code is wired correctly in your repo)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        # --- Image processor ---
        # IMPORTANT: AutoProcessor may resolve to tokenizer-like processor in this folder.
        # We load the real image processor from the vision tower specified in config.json.
        self.processor = None
        try:
            cfg_path = os.path.join(model_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            vision_tower = (
                cfg.get("mm_image_tower")
                or cfg.get("mm_vision_tower")
                or cfg.get("vision_tower")
                or cfg.get("image_tower")
            )
        except Exception:
            vision_tower = None

        if not vision_tower:
            raise RuntimeError(
                "No vision tower found in config.json. Expected mm_image_tower or similar."
            )

        # Load CLIP image processor from the tower (e.g., openai/clip-vit-large-patch14-336)
        self.processor = AutoImageProcessor.from_pretrained(vision_tower)

        self.conv_mode = conv_mode
        self.context_len = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)

    def _prepare_image(self, image: Image.Image):
        if self.processor is None:
            raise RuntimeError(
                "AutoProcessor not found in this checkpoint folder. "
                "Your regular checkpoint may be missing processor/preprocessor files. "
                "Fix: copy the vision tower processor files from your base model folder into this checkpoint "
                "(e.g., preprocessor_config.json) or load processor from the base vision tower."
            )
        # Many LLaVA processors return dict with pixel_values
        proc = self.processor(images=image, return_tensors="pt")
        if "pixel_values" in proc:
            return proc["pixel_values"].to(self.device)
        # Some processors use 'images' key or nested structure
        if isinstance(proc, dict):
            # best-effort: pick first tensor
            for v in proc.values():
                if torch.is_tensor(v):
                    return v.to(self.device)
        raise RuntimeError("Could not extract pixel values from processor output.")

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        user_prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        **kwargs,
    ) -> str:
        """
        Match MoELLaVAInfer.generate() signature used by pipelines/eval_runner.py:
          fm.generate(image_path=..., user_prompt=..., max_new_tokens=..., temperature=...)
        """

        # Load image from path
        image = Image.open(image_path).convert("RGB")

        # Build conversation prompt: your user_prompt is already the full prompt built by build_prompt()
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize with image token
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt").to(self.device)

        # Prepare image tensor
        image_tensor = self._prepare_image(image)

        gen_kwargs = dict(
            do_sample=(temperature > 0),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        # Try common argument names used by llava forks
        try_args = [
            {"images": image_tensor},
            {"pixel_values": image_tensor},
            {"image": image_tensor},
        ]

        out = None
        last_err = None
        for extra in try_args:
            try:
                out = self.model.generate(input_ids=input_ids, **extra, **gen_kwargs)
                break
            except Exception as e:
                last_err = e

        if out is None:
            raise RuntimeError(
                f"Regular LlavaPhi generate failed for all image argument variants. Last error: {last_err}"
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Strip the prompt if the decoded output includes it
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

        return text
