# JPEG Tokenizer Phase 1 Setup

This note records the first runnable training setup for the `K=4` coefficient baseline.

## Prepared Artifacts

- Token store:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/artifacts/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- GCS mirror:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- Store contents:
  - train: `9469` examples
  - validation: `3925` examples
  - sequence length: `4096`
  - vocab size: `4095`

## Code Paths

- Store reader and `LmDataConfig` wiring:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/data.py`
- Store builder:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/scripts/jpeg_tokenizer/build_coeff_token_store.py`
- Runnable launch step:
  `/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/jpeg_tokenizer/base/launch.py`

## Launch Conventions

- Executor smoke step name:
  `tokexplore/jpeg-tokenizer-k4-smoke`
- Executor step name:
  `tokexplore/jpeg-tokenizer-k4-trial`
- W&B target:
  `marin-community/tokexplore`
- W&B smoke group:
  `tokexplore-jpeg-tokenizer-k4-smoke`
- W&B group:
  `tokexplore-jpeg-tokenizer-k4`
- Default token store path:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0`
- TPU target:
  `v6e-8`

## Validation

- The file-backed store round-trips through the Levanter causal LM path in tests.
- A one-step CPU smoke test runs successfully from the file-backed store.
- The launch module now resolves the token store at runtime rather than import time, so the code remains importable in clean checkouts.
- The TPU smoke run `tokexplore/jpeg-tokenizer-k4-smoke` completed successfully on `marin-eu-west4-a` with eval loss improving from `8.508` to `4.694`.

## Completed Baseline

- Ray job:
  `ray-run-dlwh-launch-20260308-085237`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k4-trial`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.476` at startup, `4.376` at step `1000`, `4.417` at step `2000`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k4-trial-603f95/checkpoints/step-2000`
- Wall time:
  `2778.14s`

The `K=4` coefficient baseline is now proven on the production Ray + TPU path. The previous tracker/config-artifact
serialization warning has since been fixed locally by switching artifact dumping to YAML over the already-materialized
hyperparameter dict. The remaining known nuisance is a W&B `BrokenPipeError` during shutdown after the run had already
synced successfully.

## Next Step

The `K=8` smoke has now also completed successfully:

- Ray job:
  `ray-run-dlwh-launch-20260308-094432`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-smoke`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.549 -> 4.446 -> 4.124`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-smoke-c2bc8c/checkpoints/step-96`

This establishes that `8192`-token coefficient sequences are viable on `v6e-8` at batch size `128`.

## Next Step

The `K=8` longer trial is now active:

- Ray job:
  `ray-run-dlwh-launch-20260308-095441`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial`
- Current status:
  `RUNNING`
- Recovery note:
  the run survived one TPU slice preemption, resumed on a replacement worker, and continued training under the same
  W&B run ID.

## Next Step

Let the `K=8` trial finish, then compare its terminal eval against `K=4` and decide whether the next bounded rung
should be a `K=16` smoke or a width/depth adjustment at `K=8`.

## K16 Staging

The `K=16` coefficient rung has now been staged:

- Token store:
  `gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0`
- Sequence length:
  `16384`
- Store size:
  about `839 MiB`
- Smoke step:
  `tokexplore/jpeg-tokenizer-k16-smoke`
- Smoke Ray job:
  `ray-run-dlwh-launch-20260308-101439`
- Smoke output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795`

The `K=16` smoke has passed packaging, executor launch, and Fray TPU dispatch on `v6e-8`. The remaining question is
whether trainer startup and the first eval complete cleanly at batch size `64`.

The `K=8` trial is still active and has already progressed well beyond initial bring-up:

- Latest observed progress:
  about step `498/2000`
- Latest observed train loss:
  `4.08`
- Reliability note:
  the run has survived two TPU preemptions and continued retrying under the same Ray submission.

In parallel, executor-side log noise from config serialization has been fixed locally in
`marin.utilities.json_encoder`, so newly launched runs should no longer emit large warning blocks for dataclass
configs and `PartitionSpec` values.

## K16 Smoke Result

The `K=16` smoke has now completed successfully:

- Ray job:
  `ray-run-dlwh-launch-20260308-101439`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k16-smoke`
- Final status:
  `SUCCEEDED`
- Eval loss trajectory:
  `8.620 -> 3.612 -> 3.435`
- Final checkpoint:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-smoke-276795/checkpoints/step-64`

This is enough to say that `16384`-token coefficient sequences are operationally viable on `v6e-8` at batch size `64`.

## K8 Reliability Note

The original `K=8` trial has now been interrupted multiple times by TPU preemptions and has restarted from scratch after
at least one resume because no checkpoint had landed before the node died. The next `K=8` trial should therefore use a
much shorter checkpoint interval instead of the original 10-minute save cadence.

## K8 Retry Status

The retried `K=8` baseline with 2-minute checkpointing is now behaving as intended:

- Ray job:
  `ray-run-dlwh-launch-20260308-103217`
- W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial-r2`
- Latest observed progress:
  step `429/2000` with train loss `4.18`
- Confirmed checkpoints:
  steps `79`, `218`, and `358`
- Checkpoint path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints`

This resolves the main reliability concern from the original `K=8` trial: useful progress is now landing in GCS well
before the next likely TPU preemption.

## K16 Trial Staging

The next rung is now prepared as a full baseline run:

- Executor step:
  `tokexplore/jpeg-tokenizer-k16-trial`
- W&B group:
  `tokexplore-jpeg-tokenizer-k16`
- Sequence length:
  `16384`
- Batch size:
  `64`
- Train steps:
  `2000`
- Eval cadence:
  every `1000` steps
- Checkpoint policy:
  every `2` minutes, keep every `500` steps

This keeps the `K=16` trial on the same comparison shape as the completed `K=4` baseline and the active `K=8` retry.

## Active Monitoring

The current live state is:

- `K=8` retry job:
  `ray-run-dlwh-launch-20260308-103217`
- `K=8` W&B run:
  `https://wandb.ai/marin-community/tokexplore/runs/jpeg-tokenizer-k8-trial-r2`
- Latest observed `K=8` progress:
  step `772/2000` with train loss `3.61`
- Latest observed `K=8` durable checkpoints:
  `step-639` and `step-777`

This is the first `K=8` run that is clearly surviving the preemptible environment without losing the useful part of the
training curve.

- `K=16` trial job:
  `ray-run-dlwh-launch-20260308-104918`
- `K=16` output path:
  `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k16-trial-a39c10`
- Current `K=16` status:
  controller `RUNNING`, executor launched, Fray dispatch completed, trainer/W&B startup not yet observed

At this point the likely blocker for `K=16` is TPU scheduling latency rather than a launch-surface bug, so the right
action is to leave it queued and keep monitoring rather than resubmitting.

## Resume Bug And Mitigation

The next failure was not another infra-only preemption issue. The active `K=8` retry successfully:

- reached `step-1000`
- wrote checkpoint `gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k8-trial-r2-ce64bd/checkpoints/step-1000`
- recorded eval loss `3.428`

But after the following TPU preemption, the replacement worker:

- discovered `step-1000`
- logged `Loading checkpoint from .../step-1000`
- then immediately fell through to `Starting from scratch`

That exposed a bug in [train.py](/Users/dlwh/.codex/worktrees/1bd2/marin/experiments/grug/base/train.py): the grug
resume path was catching any downstream `FileNotFoundError` from `load_checkpoint(...)` and treating it as "no
checkpoint exists", even after checkpoint discovery had already succeeded. The local fix now resolves checkpoint
discovery first and only soft-fails when no checkpoint is present at all; restore-time file errors now surface instead
of silently resetting the run.

Validation for the fix passed with:

- `uv run --with pytest python -m pytest -o addopts='' tests/test_grug_variant_contracts.py -k 'resume_missing_checkpoint_data_raises or grug_base_run_emits_expected_metrics_with_json_tracker'`

To keep the cluster focused on the half-finished baseline while fixing the resume path, the queued `K=16` trial
`ray-run-dlwh-launch-20260308-104918` was intentionally stopped. The next correct move is to relaunch the `K=8` retry
from fixed code so it can resume from `step-1000` instead of wasting TPU time restarting from zero.
