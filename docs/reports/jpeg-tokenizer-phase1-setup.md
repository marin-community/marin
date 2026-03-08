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
