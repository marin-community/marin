# Grug Layout and Usage

`experiments/grug/` is template-first. You edit experiment code here directly.

## Directory layout

- `base/model.py`: model config and model implementation (`init` + `__call__` + loss method).
- `base/train.py`: train loop, optimizer step, callbacks, eval/checkpoint wiring.
- `base/launch.py`: experiment config and execution entrypoint (`ExecutorStep` + resources).

## Entry-point guide

- Start in `base/launch.py` for normal run edits.
- `GrugBaseLaunchConfig` is the user-facing knob surface (model/data/optimizer/trainer/eval/run metadata).
- `versioned(...)` marks config values that should affect executor step version/hash.
- `this_output_path()` resolves to the current step's output root.
- `run_grug(...)` in `base/train.py` is the runtime entry point used by the `ExecutorStep`.

## How to use it

1. Copy `experiments/grug/base` to a new variant directory (for example `experiments/grug/moe`).
2. Make model/training changes in that variant, not in shared trainer libraries.
3. Set run knobs in `<variant>/launch.py` (run id, data mix, optimizer, TPU type).
4. Launch from the variant's `launch.py` entrypoint.

## Quickstart launch

Local executor run:

```bash
uv run python experiments/grug/base/launch.py
```

Ray cluster run:

```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --env_vars WANDB_API_KEY=${WANDB_API_KEY} \
  -- python experiments/grug/base/launch.py
```

## Common edit points

- Architecture changes: `experiments/grug/base/model.py`
- Train-loop and callback behavior: `experiments/grug/base/train.py`
- Run config, resources, and launch wiring: `experiments/grug/base/launch.py`
- Copy-paste variant workflow: duplicate `experiments/grug/base/` into `experiments/grug/<variant>/` and edit there.

## Trainer knobs people ask about

- `z_loss_weight` in `GrugTrainerConfig`: weight on the logsumexp stabilization term in LM loss.
- `ema_beta` in `GrugTrainerConfig`: exponential moving average (EMA) coefficient for eval/checkpoint model; `None` disables EMA.

## Checkpoints and resume

- Checkpoints are written to `<output_path>/checkpoints` by default in `base/launch.py`.
- `run_grug` restores from `trainer.load_checkpoint_path` when set, otherwise tries the run checkpoint path.
- If `trainer.load_checkpoint=True` and no checkpoint is found, startup fails; otherwise it starts from scratch.

## Environment variables you will likely use

- `WANDB_API_KEY`: required for W&B logging in the default launch config.
- `GRUG_RUN_ID`: overrides the default run id.
- `FERRY_DATE`: appended to run id for ferry-style launches.
- `LEVANTER_LOG_TENSORSTORE_METRICS_EVERY`: optional cadence (steps) for tensorstore metrics logging.

## Where outputs show up

- Training/eval metrics: tracker backend (default W&B).
- Checkpoints: `<output_path>/checkpoints`.
- Profiler traces (if enabled): `<trainer.log_dir>/<run_id>/profiler`.
- Executor step outputs: `this_output_path()` root for the step.

## Logged metrics

- `train/loss`: training loss for the just-completed step.
- `global_step`: completed optimizer step index.
- `run_progress`: completed fraction of the configured run (`step / total_steps`).
- `optim/*`: optimizer hyperparameters from Optax state (for example `optim/learning_rate`).
- `throughput/duration`: step wall-clock duration (after loss is materialized).
- `throughput/examples_per_second`: examples processed per second for the current batch size.
- `throughput/tokens_per_second`: tokens processed per second.
- `throughput/total_tokens`: cumulative tokens processed so far (schedule-aware).
- `throughput/gflops_per_second`: model FLOP throughput from analytic FLOPs-per-example.
- `throughput/total_gflops`: cumulative model FLOPs (analytic).
- `throughput/mfu`: model FLOP utilization as percent of theoretical hardware FLOPs.
- `throughput/hook_time`: callback/logging overhead time after each step.
- `throughput/loading_time`: dataloader wait time for the current step.
- `throughput/flops_per_token_analytic`: analytic FLOPs per token summary value.
- `throughput/flops_per_example_analytic`: analytic FLOPs per example summary value.
- `throughput/flops_per_example`: FLOPs-per-example value used by throughput callback.
- `throughput/device_kind`: accelerator type string from JAX device info.
- `throughput/theoretical_flops_per_device`: theoretical peak FLOPs per device.
- `throughput/theoretical_flops`: theoretical peak FLOPs across all devices.
- `mixture/stage`: current data-mixture stage index.
- `mixture/weight/<dataset_name>`: effective sampling weight per dataset in the active stage.
- `eval/loss`, `eval/loading_time`, `eval/total_time`: tagged-eval loss and timing for current model.
- `eval/ema/*`: same eval metrics for EMA weights when EMA is enabled.
- `eval/macro_loss`: macro average loss across tags when multiple tags exist.
- `eval/<tag>/loss`, `eval/<tag>/micro_loss`, `eval/<tag>/macro_loss`: per-tag loss views.
- `eval/bpb`, `eval/macro_bpb`, `eval/<tag>/bpb`, `eval/<tag>/macro_bpb`: bits-per-byte metrics when tokenizer/BPB logging is enabled.
- `data/tensorstore/*`: optional TensorStore cache/IO counters when `LEVANTER_LOG_TENSORSTORE_METRICS_EVERY` is set.
- `grad/*`, `params/*`, `updates/*`, `opt_state/*`: optional watch metrics (norms/histograms) when watch is enabled.

## What should stay consistent

- Keep core training/eval metrics aligned with classic Levanter (`train/loss`, `throughput/*`, `eval/*`).
- Prefer shared helpers only for generic infrastructure; keep variant behavior local to the template.

## Further guidance

- Grug principles: [`/.agents/projects/grugformer.md`](../../.agents/projects/grugformer.md)
- Change workflow: [`/docs/recipes/change_grug.md`](../../docs/recipes/change_grug.md)
- Executor mechanics: [`/docs/explanations/executor.md`](../../docs/explanations/executor.md)
- Executor tutorial: [`/docs/tutorials/executor-101.md`](../../docs/tutorials/executor-101.md)
- TPU debug workflow: [`/docs/dev-guide/dev_tpu.md`](../../docs/dev-guide/dev_tpu.md)
- Cluster launch details: [`/docs/tutorials/tpu-cluster-setup.md`](../../docs/tutorials/tpu-cluster-setup.md)
