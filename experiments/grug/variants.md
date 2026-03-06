# Grug Variants

This file is the catalog for `experiments/grug/<variant>/`.

When adding a new variant, add a section here with:

- Variant path and intent.
- Architecture delta from base.
- Launch command(s).
- Monitoring and recovery notes.

## Parallel attn+mlp (no SWA)

Variant path: `experiments/grug/parallel_attn_mlp/`

Block wiring in this variant is explicitly parallel:

- Variant: `x + attn(ln(x)) + mlp(ln(x))`
- Baseline sequential residual update: `x = x + attn(ln(x)); x = x + mlp(ln(x))`

Iris launch command:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --extra marin:tpu \
  --region europe-west4 \
  -e MARIN_PREFIX gs://marin-eu-west4 \
  -e GRUG_RUN_ID grug-parallel-attn-mlp-trial-$(date +%Y%m%d-%H%M%S) \
  -- python -u experiments/grug/parallel_attn_mlp/launch.py
```

Monitoring:

- Use `track=iris` in [`/.agents/docs/job-monitoring-loop.md`](../../.agents/docs/job-monitoring-loop.md).
- For Iris recovery, treat restart as `stop -> resubmit`.
