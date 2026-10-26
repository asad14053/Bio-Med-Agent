# `models/`: MLLM inference wrappers

This folder holds the generator used by the online loop: **y_t = MLLM(P(t), x_p(t))** (§Method, App. A.1).

| File | Class | Used by | Notes |
|---|---|---|---|
| `moe_llava_infer.py` | `MoELLaVAInfer` | `main_run.py`, `pipelines/eval_runner.py` | Loads any MoE-LLaVA / LLaVA-Phi checkpoint through `moellava.model.builder.load_pretrained_model`. It applies `_shrink_user_prompt()` (token budget = `context_len − max_new_tokens − 32`), decodes greedily, and stops with `KeywordsStoppingCriteria`. |
| `llava_phi_infer.py` | `LlavaPhiInfer` | **not imported** | A non-MoE wrapper for `LlavaPhiForCausalLM` checkpoints. The paper's router `ckpt_is_moe()` that chooses between the two wrappers is not in the repo ([KNOWN_ISSUES B4](../docs/KNOWN_ISSUES.md)). |

Checkpoints and the MoE settings (top-2 of 8 experts, router-only training) are described in [`docs/MODELS.md`](../docs/MODELS.md).
