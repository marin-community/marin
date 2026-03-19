# B200 Cluster Setup

This note is for a fresh Codex session on a new CUDA cluster.

## Goal

Run the standalone PyTorch block diffusion experiments from:

- `experiments/block_diffusion_cuda/`

on a modern NVIDIA cluster such as `B200`.

## Branch

Fetch and check out the CUDA branch:

```bash
git fetch origin
git checkout pc0618/block-diffusion-cuda-bigdn
git pull --ff-only origin pc0618/block-diffusion-cuda-bigdn
```

## Environment

The repo `uv` environment in the original workstation did not have PyTorch, so
do not assume `uv run` alone is sufficient for this folder. Create a dedicated
CUDA Python environment first.

Example with conda:

```bash
conda create -n block-diffusion-cuda python=3.11 -y
conda activate block-diffusion-cuda
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets sentencepiece accelerate wandb
```

If you prefer `uv`, create the environment first and then install the same
packages into that interpreter.

## Sanity Checks

Confirm CUDA and BF16 are available:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("name", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
PY
```

## Credentials

Set the usual external auth before launching:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
```

`HF_TOKEN` is only required if you choose a tokenizer or dataset that needs
authenticated access.

## Recommended Dataset

For the first real run, use:

- dataset: `fineweb`
- tokenizer: `gpt2`
- streaming: enabled

Why:

- the code already supports `fineweb` as a dataset alias
- `gpt2` is public and avoids gated-tokenizer auth problems
- streaming avoids trying to materialize a huge corpus into host RAM

## Smoke Commands

Baseline:

```bash
torchrun --nproc_per_node=8 -m experiments.block_diffusion_cuda.train \
  --dataset fineweb \
  --tokenizer gpt2 \
  --variant baseline \
  --device cuda \
  --batch-size 4 \
  --steps 20 \
  --block-size 128 \
  --window-blocks 8 \
  --d-model 512 \
  --n-heads 8 \
  --gdn-heads 8 \
  --n-layers 8 \
  --streaming \
  --wandb-project block-diffusion-bigdn
```

Bi-GDN:

```bash
torchrun --nproc_per_node=8 -m experiments.block_diffusion_cuda.train \
  --dataset fineweb \
  --tokenizer gpt2 \
  --variant bigdn \
  --device cuda \
  --batch-size 4 \
  --steps 20 \
  --block-size 128 \
  --window-blocks 8 \
  --d-model 512 \
  --n-heads 8 \
  --gdn-heads 8 \
  --n-layers 8 \
  --streaming \
  --wandb-project block-diffusion-bigdn
```

## First Real Comparison

Once smoke is stable, scale both runs with matched settings such as:

- `block_size=128`
- `window_blocks=8`
- `d_model=1024`
- `n_heads=16`
- `gdn_heads=16`
- `n_layers=16`
- `steps=1000`

Run both:

- `--variant baseline`
- `--variant bigdn`

and compare:

- loss
- masked accuracy
- tokens per second
- GPU memory

## Known Limitation

`DeltaRuleMixer` in `experiments/block_diffusion_cuda/layers.py` is plain
PyTorch, not a fused CUDA kernel. It is suitable for correctness and early
research iteration, but if Bi-GDN is promising the next performance step is to
replace that inner recurrence with a fused implementation.
