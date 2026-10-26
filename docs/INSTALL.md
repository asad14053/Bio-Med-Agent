# Installation

## Tested setup (from the paper and the run logs)

| Component | Value | Source |
|---|---|---|
| OS | Linux (Ubuntu-class server) | run logs |
| GPUs | 8× NVIDIA A5000 (24 GB, compute capability 8.6) | paper; `TORCH_CUDA_ARCH_LIST="8.6"` in the fine-tune scripts |
| Python | 3.10 | `environment.yaml` |
| CUDA toolkit | 11.8 | `environment.yaml` |
| PyTorch / transformers / deepspeed | **not pinned** `<TODO: author to confirm exact versions>` | see [KNOWN_ISSUES E2](KNOWN_ISSUES.md#e-environment-and-dependencies) |

## 1. Clone

```bash
git clone https://github.com/asad14053/Bio-Med-Agent.git
cd Bio-Med-Agent
```

`flash-attention/` is recorded as a git submodule without a `.gitmodules` file, so it clones as an empty folder. None of the MSM-Mem code imports it (see [KNOWN_ISSUES E4](KNOWN_ISSUES.md)). If you need FlashAttention for training, install it separately.

## 2. Create the environment

### Option A: conda (recommended)

```bash
conda env create -f environment.yaml   # creates an env named "Bio-Med-Agent"
conda activate Bio-Med-Agent
```

### Option B: venv + pip

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Install the vendored MoE-LLaVA and ImageBind (important)

`requirements.txt` and `environment.yaml` install **upstream** MoE-LLaVA from GitHub. This repo ships a **modified** copy (`MoE-LLaVA/moellava/model/builder.py` and `train/train.py` were changed; see [KNOWN_ISSUES E1](KNOWN_ISSUES.md)). To reproduce the paper, install the local copies after step 2 so that they take precedence:

```bash
pip install -e ./MoE-LLaVA
pip install -e ./ImageBind
```

Check the install:

```bash
python - <<'EOF'
from moellava.model.builder import load_pretrained_model
from imagebind.models.imagebind_model import imagebind_huge
import faiss, moellava
print("moellava from:", moellava.__file__)   # should point inside ./MoE-LLaVA
EOF
```

## 4. Model weights

Model weights are documented in [MODELS.md](MODELS.md). ImageBind-huge downloads itself on first use (`imagebind_huge(pretrained=True)` writes to `.checkpoints/` in the current directory).

## 5. Run location

Run all pipeline modules **from the repository root** as Python modules so that `memory/`, `models/`, `tools/` and `pipelines/` are importable:

```bash
python -m pipelines.eval_runner --help
```

## Common problems

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: memory` / `tools` | You are not at the repo root, or you ran `python pipelines/x.py` instead of `python -m pipelines.x`. |
| `OSError: Lcheckpoints/… is not a local folder` | A typo in `--model_path` (seen in a committed log). |
| `RuntimeError: Image processor is missing` | The checkpoint's `mm_image_tower` cannot be resolved. You need network access to `openai/clip-vit-large-patch14-336`, or a local copy. |
| DeepSpeed fused-Adam build errors during fine-tuning | The fine-tune scripts already set `DS_BUILD_OPS=0` and `DS_DISABLE_FUSED_ADAM=1`. Keep these. |
| `TypeError: SemanticStore.update() got an unexpected keyword argument 'episode_summary'` | A known bug in `main_run.py` ([KNOWN_ISSUES D1](KNOWN_ISSUES.md)). Use `pipelines/eval_runner.py` for experiments. |
| `bitsandbytes is not installed. Disabling load_4bit/load_8bit` | This is a warning only. The model loads in fp16. |
