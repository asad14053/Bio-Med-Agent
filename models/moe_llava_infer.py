# models/moe_llava_infer.py
import torch
from PIL import Image

from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from moellava.model.builder import load_pretrained_model
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.constants import DEFAULT_IMAGE_TOKEN

from typing import Tuple


def _count_tokens_for_prompt(prompt: str, tokenizer) -> int:
    ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    if ids.dim() == 1:
        return int(ids.numel())
    return int(ids.shape[1])


def _truncate_middle_keep_head_tail(text: str, head_chars: int, tail_chars: int) -> str:
    """
    Keeps the beginning and end, removes the middle. Char-based (fast).
    Works well when retrieval blocks are huge.
    """
    text = text or ""
    if len(text) <= head_chars + tail_chars:
        return text
    return text[:head_chars] + "\n...\n" + text[-tail_chars:]


def _shrink_user_prompt(user_prompt: str, tokenizer, token_budget: int) -> str:
    """
    Token-budget shrinker for the user_prompt portion only.
    Strategy:
      1) Char-truncate aggressively (middle cut) to reduce tokens quickly.
      2) If still too long, keep only the last part (question + formatting usually live at the end).
    """
    if not user_prompt:
        return user_prompt

    # fast path
    if _count_tokens_for_prompt(f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}", tokenizer) <= token_budget:
        return user_prompt

    # Step A: shrink by middle cut (keeps early instructions + tail question/format)
    # Tune these if needed
    candidate = _truncate_middle_keep_head_tail(user_prompt, head_chars=2500, tail_chars=2500)

    if _count_tokens_for_prompt(f"{DEFAULT_IMAGE_TOKEN}\n{candidate}", tokenizer) <= token_budget:
        return candidate

    # Step B: keep tail only (prioritize question + strict format)
    # Increase tail_chars if question is long
    candidate = (user_prompt[-3500:] if len(user_prompt) > 3500 else user_prompt)

    # If still too long, keep shrinking tail until fits
    while _count_tokens_for_prompt(f"{DEFAULT_IMAGE_TOKEN}\n{candidate}", tokenizer) > token_budget and len(candidate) > 500:
        candidate = candidate[-int(len(candidate) * 0.8):]

    return candidate


def _has_bitsandbytes() -> bool:
    try:
        import importlib.metadata
        importlib.metadata.version("bitsandbytes")
        return True
    except Exception:
        return False


def _unwrap_ds_model(m):
    """
    MoE-LLaVA builder often wraps HF model with DeepSpeed inference engine.
    Common patterns:
      - engine has `.module` (original model)
      - engine may or may not expose `.generate`
    We return an object that definitely has `.generate`.
    """
    if hasattr(m, "generate"):
        return m
    if hasattr(m, "module") and hasattr(m.module, "generate"):
        return m.module
    return m


class MoELLaVAInfer:
    """
    In-process MoE-LLaVA inference for IMAGE + PROMPT.

    Note:
    - For MoE checkpoints, DeepSpeed distributed init should be done by launcher or eval_runner.
    - load_pretrained_model() in MoE-LLaVA may internally call deepspeed.init_inference.
    """

    def __init__(
        self,
        model_path: str,
        conv_mode: str = "phi",
        device: str = "cuda:0",
        load_8bit: bool = False,
        load_4bit: bool = False,
        dtype: torch.dtype = torch.float16,
        padding_side: str = "right",
    ):
        disable_torch_init()

        self.device = torch.device(device if isinstance(device, str) else str(device))
        self.conv_mode = conv_mode
        self.dtype = dtype

        if self.conv_mode not in conv_templates:
            raise ValueError(
                f"conv_mode='{self.conv_mode}' not in conv_templates. "
                f"Available: {list(conv_templates.keys())}"
            )

        if (load_4bit or load_8bit) and not _has_bitsandbytes():
            print(
                "[WARN] bitsandbytes is not installed. "
                "Disabling load_4bit/load_8bit and loading in fp16."
            )
            load_4bit = False
            load_8bit = False

        model_name = get_model_name_from_path(model_path)

        tok, model_obj, processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=str(self.device),
            padding_side=padding_side,
        )

        self.tokenizer = tok
        self.context_len = context_len
        self.processor = processor

        # Processor safety
        if not isinstance(self.processor, dict) or "image" not in self.processor or self.processor["image"] is None:
            raise RuntimeError(
                "Image processor is missing. builder.py returned processor['image']=None. "
                "This usually means the vision tower was not loaded or mm_image_tower config is None."
            )
        self.image_processor = self.processor["image"]

        # Unwrap DS engine if needed
        self.model = _unwrap_ds_model(model_obj)

        # ---- Pad token safety (prevents HF warnings, stabilizes generation) ----
        # GPT-style tokenizers sometimes don't define a pad token. Reuse EOS.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass

        # Only move if it's a plain module (DeepSpeed engine/module is usually already placed/injected)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        # 1) Image -> tensor
        img = Image.open(image_path).convert("RGB")
        proc_out = self.image_processor.preprocess(img, return_tensors="pt")
        if "pixel_values" not in proc_out:
            raise RuntimeError("image_processor.preprocess did not return 'pixel_values'.")
        image_tensor = proc_out["pixel_values"].to(self.device, dtype=self.dtype)

        # 2) Conversation template
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles

        # 3) Enforce a hard token budget BEFORE generate()
        # Keep some safety margin for separators / special tokens.
        safety_margin = 32
        # Input token budget = context_len - max_new_tokens - safety
        token_budget = max(256, int(self.context_len) - int(max_new_tokens) - safety_margin)

        # shrink only user_prompt (retrieval/context) if needed
        user_prompt = _shrink_user_prompt(user_prompt, self.tokenizer, token_budget)

        inp = f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}"
        conv.append_message(roles[0], inp)
        conv.append_message(roles[1], None)
        prompt = conv.get_prompt()

        # 4) Tokenize with image tokens
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Final guard (if template adds extra tokens and still exceeds)
        if input_ids.shape[1] > token_budget:
            # Keep the tail tokens (question/format are usually near the end)
            input_ids = input_ids[:, -token_budget:]

        # IMPORTANT: attention mask (all ones, since we do not pad in this code path)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)

        # 5) Stop tokens
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # 6) Generate
        # NOTE: temperature only matters if do_sample=True; we keep deterministic by default.
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        gen = output_ids[0, input_ids.shape[1]:]
        out = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return out
