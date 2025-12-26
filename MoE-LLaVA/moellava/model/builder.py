#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig,
)

from moellava.model import *  # noqa: F403
from moellava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
)
from moellava.model.language_model.qwen.tokenization_qwen import QWenTokenizer

from moellava.model.language_model.llava_qwen_moe import EvalMoELLaVAQWenForCausalLM
from moellava.model.language_model.llava_qwen import LlavaQWenForCausalLM

from moellava.model.language_model.llava_llama_moe import EvalMoELLaVALlamaForCausalLM
from moellava.model.language_model.llava_llama import LlavaLlamaForCausalLM

# Optional families gated by transformers version
a, b, _c = transformers.__version__.split(".")[:3]
if a == "4" and int(b) >= 34:
    from moellava.model.language_model.llava_mistral_moe import EvalMoELLaVAMistralForCausalLM
    from moellava.model.language_model.llava_mistral import LlavaMistralForCausalLM

if a == "4" and int(b) >= 36:
    from moellava.model.language_model.llava_minicpm_moe import EvalMoELLaVAMiniCPMForCausalLM
    from moellava.model.language_model.llava_minicpm import LlavaMiniCPMForCausalLM
    from moellava.model.language_model.llava_phi_moe import EvalMoELLaVAPhiForCausalLM
    from moellava.model.language_model.llava_phi import LlavaPhiForCausalLM
    from moellava.model.language_model.llava_stablelm_moe import EvalMoELLaVAStablelmForCausalLM
    from moellava.model.language_model.llava_stablelm import LlavaStablelmForCausalLM

if a == "4" and int(b) >= 37:
    from moellava.model.language_model.llava_qwen1_5_moe import EvalMoELLaVAQwen1_5ForCausalLM
    from moellava.model.language_model.llava_qwen1_5 import LlavaQwen1_5ForCausalLM


def _maybe_set_generation_config(tokenizer, model, model_id_for_config: str):
    """
    Keep original behavior: for Qwen models they customize generation_config.
    For others, leave as default unless caller sets it elsewhere.
    """
    try:
        model.generation_config = GenerationConfig.from_pretrained(
            model_id_for_config, pad_token_id=getattr(tokenizer, "pad_token_id", None)
        )
    except Exception:
        pass


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    padding_side="right",
    merge=False,
    **kwargs,
):
    """
    Full edit goals:
    1) Fix Phi routing: never instantiate Llama classes for Phi checkpoints.
    2) Remove DeepSpeed requirement for single-GPU inference (your environment).
       - No deepspeed.init_distributed
       - No ds_engine wrapping
    3) Remove broken indentation and undefined variables (dtype) in Phi model_base path.
    4) Keep LoRA merge path intact.
    5) Keep mm_projector loading and image/video tower loading behavior.
    """

    # Build kwargs
    kwargs = {"device_map": device_map, **kwargs}

    # If user explicitly sets device != cuda, pin everything there.
    if device != "cuda":
        kwargs["device_map"] = {"": device}

    # Quantization / dtype
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = None
    tokenizer = None

    if "llava" in model_name.lower():
        # ----------------------------
        # LLaVA / MoE-LLaVA model load
        # ----------------------------
        if "lora" in model_name.lower() and "moe" not in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. "
                "If you are loading a LoRA model, please provide the `model_base` argument. "
                "Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )

        # 1) LoRA non-MoE with model_base: load base then merge
        if "lora" in model_name.lower() and "moe" not in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
            print("Loading LLaVA from base model...")

            base_lower = model_base.lower()
            if "qwen" in base_lower and "1.5" not in base_lower:
                model = LlavaQWenForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                _maybe_set_generation_config(tokenizer, model, model_base)
                model.generation_config.do_sample = False
                model.generation_config.repetition_penalty = 1.0
            elif "openchat" in base_lower or "mistral" in base_lower:
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
            elif "qwen" in base_lower and "1.5" in base_lower:
                model = LlavaQwen1_5ForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "phi" in base_lower:
                model = LlavaPhiForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "minicpm" in base_lower:
                model = LlavaMiniCPMForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "stablelm" in base_lower:
                model = LlavaStablelmForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )

            # Fix token embedding resizing issues (kept from your original)
            token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )
                model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")

            non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}

            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")

        # 2) LoRA MoE path: keep as-is but REMOVE deepspeed wrapping for your single-GPU inference
        elif "lora" in model_name.lower() and "moe" in model_name.lower():
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            print("Adapting to MoE...")

            name_lower = model_name.lower()
            if "qwen" in name_lower and "1.5" not in name_lower:
                tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAQWenForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                _maybe_set_generation_config(tokenizer, model, model_path)
                model.generation_config.do_sample = False
                model.generation_config.repetition_penalty = 1.0
            elif "openchat" in name_lower or "mistral" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAMistralForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
            elif "qwen" in name_lower and "1.5" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "phi" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAPhiForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "minicpm" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            elif "stablelm" in name_lower:
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer

                tokenizer = Arcade100kTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )
                model.config.eos_token_id = tokenizer.eos_token_id
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = EvalMoELLaVALlamaForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs
                )

            # IMPORTANT: do NOT wrap with DeepSpeed for single GPU unless you truly need it.
            # The original code did this if not merge. We disable it because it forced distributed init.
            # If you later want DS inference, launch via deepspeed/torchrun, not from inside this loader.
            # if not merge: ...

            model.to(device)

        # 3) model_base not None: mm_projector-only / delta weights
        elif model_base is not None:
            print("Loading LLaVA from base model...")

            name_lower = model_name.lower()

            if "mpt" in name_lower:
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"), os.path.join(model_path, "configuration_mpt.py")
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )

            elif "openchat" in name_lower or "mistral" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAMistralForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaMistralForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )

            elif "phi" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaPhiConfig.from_pretrained(model_path)  # noqa: F403
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAPhiForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaPhiForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "qwen" in name_lower and "1.5" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaQwen1_5Config.from_pretrained(model_path)  # noqa: F403
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaQwen1_5ForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "minicpm" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = LlavaMiniCPMConfig.from_pretrained(model_path)  # noqa: F403
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaMiniCPMForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "stablelm" in name_lower:
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
                from moellava.model.language_model.stablelm.configuration_stablelm_epoch import StableLMEpochConfig

                tokenizer = Arcade100kTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = StableLMEpochConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaStablelmForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )

            elif "qwen" in name_lower and "1.5" not in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVAQWenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaQWenForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                _maybe_set_generation_config(tokenizer, model, model_base)
                model.generation_config.do_sample = False
                model.generation_config.repetition_penalty = 1.0

            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if getattr(cfg_pretrained, "moe", {}).get("moe_enable", False):
                    model = EvalMoELLaVALlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                    )

            # Load mm_projector weights (kept)
            mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
            model.to(device)

        # 4) model_base is None, non-LoRA direct loading from model_path
        else:
            name_lower = model_name.lower()

            if "mpt" in name_lower:
                if "moe" in name_lower:
                    raise NotImplementedError
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

            elif "qwen" in name_lower and "1.5" not in name_lower:
                tokenizer = QWenTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaQWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                _maybe_set_generation_config(tokenizer, model, model_path)
                model.generation_config.do_sample = False
                model.generation_config.repetition_penalty = 1.0

            elif "openchat" in name_lower or "mistral" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaMistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

            elif "phi" in name_lower:
                # FIX: Phi must use Phi classes, not Llama classes.
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "qwen" in name_lower and "1.5" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAQwen1_5ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaQwen1_5ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "minicpm" in name_lower:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                model.config.eos_token_id = tokenizer.eos_token_id

            elif "stablelm" in name_lower:
                from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer

                tokenizer = Arcade100kTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVAStablelmForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaStablelmForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                if "moe" in name_lower:
                    assert not load_8bit and not load_4bit
                    model = EvalMoELLaVALlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

            model.to(device)

    else:
        # ----------------------------
        # Plain language model loading
        # ----------------------------
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, padding_side=padding_side)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side=padding_side)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        model.to(device)

    # ==========================================================================================================
    # Processor loading (kept)
    processor = {"image": None, "video": None}

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

        model.resize_token_embeddings(len(tokenizer))

        if getattr(model.config, "mm_image_tower", None) is not None:
            image_tower = model.get_image_tower()
            if not image_tower.is_loaded:
                image_tower.load_model()
            image_tower.to(device=device, dtype=torch.float16)
            processor["image"] = image_tower.image_processor

        if getattr(model.config, "mm_video_tower", None) is not None:
            video_tower = model.get_video_tower()
            if not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=torch.float16)
            processor["video"] = video_tower.video_processor

    # Context length
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
