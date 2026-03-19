# PyTorch Block Diffusion CUDA Experiment

This directory is a CUDA-native block diffusion learning scaffold intended for
running on NVIDIA GPUs such as `B200`.

It is not a verbatim vendor of upstream training stacks. Instead it is a small,
readable implementation that combines:

- a vanilla attention block diffusion denoiser
- a hybrid Bi-GDN block diffusion denoiser with a `3:1` GDN-to-attention ratio
- a simple block-autoregressive sampler

## Upstream References

- Block Diffusion baseline reference: `https://github.com/kuleshov-group/bd3lms`
- Gated DeltaNet reference: `https://github.com/NVlabs/GatedDeltaNet`

This folder keeps the implementation compact so it is easy to modify for
research ideas.

## Design

Training uses a direct block-conditional objective:

1. take a token window of `window_blocks * block_size`
2. sample one target block inside that window
3. treat all previous tokens as clean context
4. corrupt the target block by replacing a subset with the mask token
5. train the denoiser to recover the original target tokens only on masked sites

This is a clean block diffusion setup that supports both variants with the same
outer training loop.

### Variants

- `baseline`
  - every layer uses full attention over `[context | active_block]`
  - context tokens are causal
  - active block tokens are bidirectional and can attend to all context
- `bigdn`
  - every fourth layer uses full attention
  - the other three layers use a hybrid GDN mixer
  - context tokens use forward-only delta-rule recurrence
  - active block tokens use split-channel bidirectional GDN

## Files

- `config.py`: dataclasses for model, data, and training settings
- `layers.py`: RMSNorm, attention mixer, delta-rule mixers, AdaLN
- `model.py`: baseline and Bi-GDN denoisers
- `diffusion.py`: corruption schedule, loss, and block sampler
- `data.py`: toy, text-file, Hugging Face, and FineWeb streaming datasets
- `train.py`: single-node training entrypoint with optional DDP

## Example Commands

Tiny CPU smoke:

```bash
uv run python -m experiments.block_diffusion_cuda.train \
  --dataset toy \
  --variant baseline \
  --device cpu \
  --no-compile \
  --steps 5 \
  --batch-size 2 \
  --block-size 8 \
  --window-blocks 4 \
  --d-model 32 \
  --n-heads 4 \
  --gdn-heads 4 \
  --n-layers 4
```

Single-GPU CUDA run:

```bash
uv run python -m experiments.block_diffusion_cuda.train \
  --dataset text \
  --text-file /path/to/text.txt \
  --tokenizer gpt2 \
  --variant bigdn \
  --device cuda \
  --steps 1000 \
  --batch-size 16 \
  --block-size 128 \
  --window-blocks 8 \
  --d-model 1024 \
  --n-heads 16 \
  --gdn-heads 16 \
  --n-layers 16 \
  --wandb-project block-diffusion-bigdn
```

Single-node FineWeb baseline:

```bash
torchrun --nproc_per_node=8 -m experiments.block_diffusion_cuda.train \
  --dataset fineweb \
  --tokenizer gpt2 \
  --variant baseline \
  --device cuda \
  --batch-size 8 \
  --block-size 128 \
  --window-blocks 8 \
  --d-model 1024 \
  --n-heads 16 \
  --gdn-heads 16 \
  --n-layers 16 \
  --streaming \
  --wandb-project block-diffusion-bigdn
```

Single-node FineWeb Bi-GDN:

```bash
torchrun --nproc_per_node=8 -m experiments.block_diffusion_cuda.train \
  --dataset fineweb \
  --tokenizer gpt2 \
  --variant bigdn \
  --device cuda \
  --batch-size 8 \
  --block-size 128 \
  --window-blocks 8 \
  --d-model 1024 \
  --n-heads 16 \
  --gdn-heads 16 \
  --n-layers 16 \
  --streaming \
  --wandb-project block-diffusion-bigdn
```

Multi-GPU DDP:

```bash
torchrun --nproc_per_node=8 -m experiments.block_diffusion_cuda.train \
  --dataset hf \
  --hf-dataset allenai/c4 \
  --hf-text-field text \
  --tokenizer gpt2 \
  --variant baseline \
  --device cuda
```

## Tokenizer Notes

- Default tokenizer is `auto`.
- For `fineweb` and `fineweb_edu`, `auto` resolves to `gpt2`.
- If you want a different tokenizer family, pass `--tokenizer ...` explicitly.
- If you choose a gated or proprietary tokenizer that requires auth, make sure the
  new cluster has the right Hugging Face token configured before launching.

## Known Limitations

- The GDN path is written in plain PyTorch for readability.
- For maximum B200 performance, the natural next step is to replace the
  recurrent inner loop in `DeltaRuleMixer` with a fused CUDA or FLA kernel.
- The sampler is intentionally simple and should be treated as a research
  baseline rather than a production decoding stack.
