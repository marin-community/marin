# Agentic SFT learning-rate sweep

The Snowball v0 sweep in [#10213](https://github.com/marin-community/marin/issues/10213)
uses prebuilt 32,768-token rows with already-shifted assistant loss weights.
Each full global batch contains 16 rows without gradient accumulation.
The optimizer is AdamW with constant LR, betas 0.9/0.999, zero weight decay
and gradient clip 1.0. Allowed rates are 3e-7, 1e-6, 3e-6 and 1e-5.

Build a plan without submitting jobs:

```bash
uv run python -m experiments.post_training.agentic_sweep.train \
  --report <prepared-manifest-uri> --report-sha256 <sha256> \
  --parent-uri <verified-regional-model-uri> --learning-rate 1e-6 --only train
```

`--run` executes the graph from an Iris coordinator. Set `MARIN_PREFIX`,
`CW_KEY_ID`, `CW_KEY_SECRET` and `WANDB_API_KEY` in its environment; supply
`HF_TOKEN` for the pinned tokenizer. The graph verifies prepared file hashes
and cache row/mask/token totals before requesting the four-node H100 gang.
GPU jobs explicitly use `NCCL_DEBUG=WARN` and zero failure/preemption retries.

The historical sampler uses shuffled block-2048 mixture windows. A nominal
epoch is `padded_rows / 16` updates, not an exact traversal of the dataset.
Token-weighted online loss uses measured active tokens, and deviations from
the manifest are recorded without aborting. With 8,768 rows, a CPU replay
found 424 duplicate and 424 missing row visits within a nominal window.
This sampler is unsuitable for experiments requiring epoch-specific rows.

Native checkpoints are retained at each completed window; the final model is
exported to BF16 HF format. `--only validate --validation-checkpoint <native>
--validation-output <json>` restores a checkpoint on one eight-GPU node with
no context sharding. It reports supervised-token-weighted family and pooled
losses with measured denominators. `--only export --export-checkpoint <native>
--export-output <hf-dir>` converts a retained checkpoint without training.
Both modes require `--run` to submit work.

The continuation option accepts only the historical 1e-6 first-window boundary.
Prior window statistics are retained in the original run's artifacts; the
continuation's epoch file contains only its newly measured windows.
