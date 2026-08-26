# d512 constant-LR TPU sweep

This variant reproduces the 25 d512 cells from
[issue #7856](https://github.com/marin-community/marin/issues/7856) on TPU,
changing only the learning-rate schedule after warmup:

- historical control: 1% warmup, then linear decay to 5% of peak LR;
- this variant: 1% warmup, then constant peak LR.

The model, batch 64, sequence length 8192, seed 0, two-stage datakit mixture,
five token budgets, five peak-LR multipliers, and Paloma evaluation cadence
match the completed `aug-hero-d512-*-v2` W&B runs. TPU training uses a v4-8 in
`us-central2-b` and reads the TensorStore caches from
`gs://marin-us-central2/datakit/store_8ac06c74`. TensorStore requires a native
GCS URI; the generic `mirror://` fsspec scheme cannot open its shard arrays.

## Launch

Submit a CPU-only Iris parent. The `StepRunner` in the parent submits v4-8
children through Fray.

Representative 30x / 1.0x cell:

```bash
uv run iris --cluster=marin job run \
  --no-wait \
  --cpu=1 \
  --memory=2G \
  --extra=cpu \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -- python -m experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch \
    --token-multiple 30 \
    --lr-multiplier 1 \
    --max-concurrent 1
```

Full 25-cell matrix, bounded to five concurrent TPU children:

```bash
uv run iris --cluster=marin job run \
  --no-wait \
  --cpu=1 \
  --memory=2G \
  --extra=cpu \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -- python -m experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch
```

W&B project: `marin-community/marin_moe`; group:
`issue-7856-d512-constant-lr-tpu`.
