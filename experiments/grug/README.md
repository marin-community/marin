# Grug Experiments: Templates, Not Libraries

This directory is the canonical edit surface for grug variants.

## Layout

- `base/model.py`: model config + module implementation
- `base/train.py`: training loop + eval/checkpoint wiring + metric logging
- `base/launch.py`: run-level config (data mix, run id, resources, optimizer knobs)

## Workflow

1. Copy `experiments/grug/base` to a variant directory (for example `experiments/grug/moe`).
2. Modify model/training code in that variant directly.
3. Keep shared infra helpers in library code only when they are not grug-specific.

## Metric Parity Contract

Every grug variant should log the same core keys as standard Levanter runs where applicable:

- `train/loss`
- `throughput/*` performance metrics
- `eval/*` validation metrics (including EMA when enabled)
- `mixture/*` dataset-stage metrics
- watch metrics when watch is enabled

This keeps apples-to-apples comparisons across template variants and classic runs.
