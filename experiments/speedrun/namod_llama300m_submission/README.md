# NAMO-D LLaMA-300M Speedrun Submission

This folder contains the official speedrun attempt script for LLaMA-300M with NAMO-family optimizers.

## Launch (local GPUs)

```bash
export MARIN_PREFIX=/data/suraj/speedrun/marin_store
export WANDB_API_KEY=<your_key>
export WANDB_ENTITY=suranganath-uc-san-diego
export WANDB_PROJECT=marin
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NAMO_SPEEDRUN_TOKENIZED_INPUT=tokenized/subcache/fineweb-edu-10B-ac65f6

# Primary attempt (NAMO-D, submission configuration)
export NAMO_SPEEDRUN_PROFILES=official_namod_alt
export NAMO_SPEEDRUN_GPU_COUNT=6
export NAMO_SPEEDRUN_BATCH_SIZE=144
export NAMO_SPEEDRUN_STEPS_PER_EVAL=1000
export NAMO_SPEEDRUN_STEPS_PER_EXPORT=5000
uv run python experiments/speedrun/namod_llama300m_submission/namod_llama300m_submission.py \
  --prefix "$MARIN_PREFIX" \
  --max_concurrent 1
```

## Useful overrides

- `NAMO_SPEEDRUN_PROFILES`: comma-separated list from `{official_namod, official_namod_alt, fallback_namo}`
- `NAMO_SPEEDRUN_GPU_COUNT`: number of visible GPUs to use
- `NAMO_SPEEDRUN_BATCH_SIZE`: global train batch size (default 144)
- `NAMO_SPEEDRUN_PER_DEVICE_PARALLELISM`: microbatch size per device (default 4)
- `NAMO_SPEEDRUN_TOKENIZED_INPUT`: local pretokenized cache path relative to prefix
- `NAMO_SPEEDRUN_NUM_TRAIN_STEPS`: optional override for quick smoke runs
- `NAMO_SPEEDRUN_RUN_SUFFIX`: optional suffix to create a fresh run/output path

## Dry run

```bash
uv run python experiments/speedrun/namod_llama300m_submission/namod_llama300m_submission.py \
  --prefix "$MARIN_PREFIX" \
  --dry_run true
```
