# Debugging log for training-v2-iris

Validate fray v2 migration for Levanter training and confirm Iris submission paths (smoke -> CPU tutorial -> TPU tutorial) work with correct TPU resource requests.

## Initial status

User requested fray v2 migration for training and Iris test runs using `uv run iris run`, with correct TPU type for TPU jobs.

## Hypothesis 1

If Iris jobs fail, the most likely causes are missing extras (CPU vs TPU deps) or incorrect TPU type passed to `iris run`.

## Changes to make

- Update Levanter tutorial scripts to import `ResourceConfig` from `fray.v2`.
- Update training design doc with spiral plan and Iris run steps.
- Use `uv run iris --config ... run` for smoke/CPU/TPU jobs with explicit `--tpu` for TPU.

## Future Work

- [ ] Confirm `default_train` works on Iris with fray v2 end-to-end.
- [ ] Align remaining tutorial scripts with `fray.v2` if needed.
- [ ] Add a minimal Iris job wrapper for Levanter tests if repeated manual runs are common.

## Results

Smoke job (local Iris cluster) succeeded using:

- `uv run iris --config lib/iris/examples/local.yaml run --extra marin:cpu -- python -c "print('smoke')"`

Observed autoscaler scale-up, worker registration, and job completion with log output `smoke`.

Attempted Levanter CPU tutorial (local Iris) without MARIN_PREFIX and saw:

- `ValueError: Must specify a prefix or set the MARIN_PREFIX environment variable`

Retried with MARIN_PREFIX, but tokenization step failed due to request timeouts
from Zephyr actor RPCs (Iris local cluster). The job ended in failed state after
timeouts. A follow-up run with `ZEPHYR_NUM_WORKERS=4` reduced worker fan-out but
still stalled with repeated retries/timeouts; I terminated the run to avoid a
runaway local controller.

Retried again with `ZEPHYR_NUM_WORKERS=1`. Zephyr worker/coordinator jobs still
cycled with retries and were repeatedly relaunched. I terminated the run to
avoid a runaway controller.

Next: add targeted logging in Iris task lifecycle and local container startup to
understand why local CPU runs churn. Remove ad-hoc timeout overrides and rely
on Iris job metadata.

## Changes to make

- Add lifecycle logging in `lib/iris/src/iris/cluster/worker/task_attempt.py`
  to see bundle, build, container, and exit timing.
- Add local container startup logging in
  `lib/iris/src/iris/cluster/vm/local_platform.py`.
- Remove ad-hoc actor timeout overrides.

## Results

Logging added to Iris task attempts and local VM container start.
Ad-hoc actor timeout overrides removed.

Rerun local CPU tutorial:

- Command: `uv run iris --config lib/iris/examples/local.yaml run --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -e ZEPHYR_NUM_WORKERS 1 -- python -m experiments.tutorials.train_tiny_model_cpu`
- Iris controller registered worker and launched job quickly (<1s).
- Executor submitted nested `/train_lm` job to Iris (as expected for fray v2).
- No training logs observed in parent stream.
- Checkpoints directory created at `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc` with executor metadata files only.

Next: improve visibility into nested Iris jobs (child log streaming) to determine whether training is stalled or logs are simply not visible in the parent log stream.

## Changes to make

- Add child log streaming support in Iris client wait loop.
- Add CLI flag to include child logs during `iris run`.

## Results

Added `--include-children-logs` to `iris run` and child-log streaming in `Job.wait()`.

Rerun local CPU tutorial with fresh checkpoint path:

- Removed `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc` to force rerun.
- Command: `uv run iris --config lib/iris/examples/local.yaml run --include-children-logs --extra marin:cpu -e MARIN_PREFIX /tmp/marin-prefix -e ZEPHYR_NUM_WORKERS 1 -- python -m experiments.tutorials.train_tiny_model_cpu`
- Nested `/train_lm` job logs visible (wandb init, training progress, eval, checkpoint save).
- Training completed successfully (100 steps, eval loss ~10.029, HF checkpoint saved to `/tmp/marin-prefix/checkpoints/marin-nano-tinystories-4da1fc/hf/step-99`).
- Top-level Iris job completed successfully.
- One warning observed: `iris.managed_thread` reported `on_stop callback for task-/.../train_lm/0 did not complete` after task exit.

Next: decide whether to address the on_stop callback warning or treat as benign; proceed to TPU job once desired.
