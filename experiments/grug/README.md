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
- `P` in train/model code is the usual JAX alias for `PartitionSpec`; see the JAX explicit sharding tutorial: [Explicit Sharding (JAX)](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html).

## How to use it

1. Copy `experiments/grug/base` to a new variant directory (for example `experiments/grug/moe`).
2. Make model/training changes in that variant, not in shared trainer libraries.
3. Set run knobs in `<variant>/launch.py` (run id, data mix, optimizer, TPU type).
4. Launch from the variant's `launch.py` entrypoint.
5. Add or update variant-specific notes in `experiments/grug/variants.md`.

## Variant notes

Variant-specific guidance (including modular-opt notes) lives in `experiments/grug/variants.md`.

## Quickstart launch

Local executor run:

```bash
uv run python experiments/grug/base/launch.py
```

Iris cluster run (from a dev box, on `marin` prod cluster):

```bash
uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.base.launch
```

The entrypoint job is CPU-only; `executor_main` inside it submits TPU sub-tasks via Fray. See [`lib/iris/OPS.md`](../../lib/iris/OPS.md) for flag reference and troubleshooting.

## Visual diff for template variants

When template-copying `experiments/grug/base/` to a new variant, use the HTML diff tool to review changes:

```bash
uv run python scripts/grug_dir_diff.py \
  experiments/grug/base \
  experiments/grug/<variant> \
  --out /tmp/grug-diff
```

What it does:

- Recursively compares both directories (default extensions: `.py`, `.md`)
- Builds one `/tmp/grug-diff/index.html` report with top-level summary + file table
- Renders inline side-by-side diffs on that same page (changed/added/removed by default)
- PRs that add a new `experiments/grug/<variant>/` also get this diff posted automatically by CI.

Useful flags:

- `--extensions .py,.md`: restrict file types
- `--all-files`: include everything instead of extension filtering
- `--show-unchanged`: generate pages for unchanged files too
- `--context-lines 5`: tune context around edits
- `--no-open`: generate the report without launching a browser

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

## Where outputs show up

- Training/eval metrics: tracker backend (default W&B).
- Checkpoints: `<output_path>/checkpoints`.
- Profiler traces (if enabled): `<trainer.log_dir>/<run_id>/profiler`.
- Backward-flow artifacts: `<trainer.log_dir>/<run_id>/artifacts/backward_flow`, plus `backward_flow/dag` as native HTML media in W&B. Base Grug samples every 50 steps by default; set `trainer.backward_flow.interval=0` to disable.
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
- `grad/*`, `params/*`, `updates/*`, `opt_state/*`: optional watch metrics (norms/histograms) when watch is enabled.
- `backward_flow/<scope>/*`: sampled activation and backward-gradient scale stats when `trainer.backward_flow.interval > 0`. Grug also logs `*_gradient_rms_scaled`, where gradients are multiplied by `sum(loss_weight)` to undo mean-loss scaling for the visualization.

## What should stay consistent

- Keep core training/eval metrics aligned with classic Levanter (`train/loss`, `throughput/*`, `eval/*`).
- Prefer shared helpers only for generic infrastructure; keep variant behavior local to the template.

## Variant contract (enforced by tests)

`tests/test_grug_variant_contracts.py` treats each subdirectory under `experiments/grug/` as a variant and
enforces these minimum interfaces:

- If `<variant>/model.py` exists, it must define:
  - `GrugModelConfig` constructable as `GrugModelConfig(vocab_size=...)`
  - `Transformer` with `next_token_loss(...)`
  - `debug_mesh_and_token_pspec(num_devices: int)`
- If `<variant>/train.py` exists, it must define:
  - `initial_state(model_config, *, optimizer, mp, key)` (all required)
  - `_make_train_step(...)`
  - `run_grug(...)`
- If both `model.py` and `train.py` exist, the variant must lower a one-step train path under abstract mesh via
  `eqx.filter_eval_shape`.
- Escape hatch: add `# GRUG NOVERIFY` anywhere in `<variant>/train.py` to exclude that variant from these contract
  checks.

## Further guidance

- Grug principles: [`/.agents/projects/grugformer.md`](../../.agents/projects/grugformer.md)
- Change workflow: [`.agents/skills/change-grug/`](../../.agents/skills/change-grug/SKILL.md)
- Backward-flow recipe: [`/docs/recipes/add_grug_backward_flow_logging.md`](../../docs/recipes/add_grug_backward_flow_logging.md)
- HBM/OOM tuning guide: [`/docs/references/hbm-optimization.md`](../../docs/references/hbm-optimization.md)
- Executor mechanics: [`/docs/explanations/executor.md`](../../docs/explanations/executor.md)
- Executor tutorial: [`/docs/tutorials/executor-101.md`](../../docs/tutorials/executor-101.md)
- TPU debug workflow: [`.agents/skills/dev-tpu/`](../../.agents/skills/dev-tpu/SKILL.md)
- Cluster launch details: [`lib/iris/OPS.md`](../../lib/iris/OPS.md), [`.agents/skills/ferries/SKILL.md`](../../.agents/skills/ferries/SKILL.md)
