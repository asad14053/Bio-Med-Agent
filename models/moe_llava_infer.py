import torch
from PIL import Image

from moellava.utils import disable_torch_init
from moellava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from moellava.model.builder import load_pretrained_model
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.constants import DEFAULT_IMAGE_TOKEN


class MoELLaVAInfer:
    """
    In-process MoE-LLaVA inference for IMAGE + PROMPT.

    You MUST have MoE-LLaVA installed in the active env:
      cd /path/to/MoE-LLaVA && pip install -e .
    """

    def __init__(
        self,
        model_path: str,
        conv_mode: str = "phi",    # "phi", "qwen", "stablelm" depending on your checkpoint
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        disable_torch_init()
        self.device = device
        self.conv_mode = conv_mode
        self.dtype = dtype

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            device=device,
        )

        self.image_processor = self.processor["image"]
        self.model.eval()

        # ---- Force safe generation defaults (greedy, no cache, no fancy sampling) ----
        if hasattr(self.model, "generation_config"):
            gc = self.model.generation_config

            # no sampling: pure greedy
            gc.do_sample = False
            gc.num_beams = 1
            gc.temperature = 1.0

            # avoid top_k warning (unset or neutralize)
            if hasattr(gc, "top_p"):
                gc.top_p = 1.0
            if hasattr(gc, "top_k"):
                gc.top_k = None  # << key to silence that warning

            # turn OFF KV cache so buggy prepare_inputs_for_generation doesn't touch past_key_values
            if hasattr(gc, "use_cache"):
                gc.use_cache = False

            print("[MoELLaVAInfer] generation_config:", gc)

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,  # kept for API compatibility; not used when do_sample=False
    ) -> str:
        # 1) Preprocess image
        img = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor.preprocess(
            img,
            return_tensors="pt"
        )["pixel_values"]
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        # 2) Build conversation template
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles

        inp = f"{DEFAULT_IMAGE_TOKEN}\n{user_prompt}"
        conv.append_message(roles[0], inp)
        conv.append_message(roles[1], None)
        prompt = conv.get_prompt()

        # 3) Tokenize with image tokens
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)

        # 4) Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str],
            self.tokenizer,
            input_ids
        )

        # 5) Greedy generation, NO cache
        output_ids = self.model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,           # force greedy
            temperature=1.0,           # ignored when do_sample=False
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=False,           # << IMPORTANT: disable KV cache to avoid None cache bug
            stopping_criteria=[stopping_criteria],
        )

        # 6) Decode
        out = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return out
