# Debugging log for grug gatednorm resume TPU halt

Diagnose why `isoflop-moe-adamh-gatednorm-r2-3e+18-d2048-v5p16` repeatedly resumes from checkpoint and then dies near the eval/checkpoint boundary with:

- `jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly`
- `An unexpected peer shows up in the launch group with a different launch id`
- follow-on `FAILED_PRECONDITION: The program continuator has halted unexpectedly`

## Initial status

From the user logs on 2026-03-21:

- The run resumes the wandb run successfully.
- The run restores a legacy wrapped checkpoint from `.../checkpoints/step-1000`.
- wandb reports `Step 1000 is less than the current step 1199. Cowardly refusing to log metrics.`
- Eval completes and logs metrics successfully.
- The first fatal traceback appears at `experiments/grug/moe_scaling_iteration_02/train.py:521` inside `checkpointer.on_step(...)`.
- A second traceback appears from the `finally:` cleanup path at `train.py:525` when forced checkpointing retries after the runtime is already dead.

## Hypothesis 1

The checkpoint loader is restoring the wrong checkpoint, causing the crash.

## Changes to make

Read:

- `experiments/grug/moe_scaling_iteration_02/train.py`
- `experiments/grug/checkpointing.py`
- `lib/levanter/src/levanter/checkpoint.py`
- `lib/levanter/src/levanter/tracker/wandb.py`

## Results

- The recent Grug change replaced `load_checkpoint(... discover_latest=True)` with `restore_grug_state_from_checkpoint(...)`.
- The custom loader sorts candidates by checkpoint step and then timestamp, then loads the first readable candidate.
- In the reported run it intentionally restores `step-1000`, which is the latest valid checkpoint visible in storage.
- This explains the wandb mismatch: the wandb run had already advanced to step `1199`, but the last durable checkpoint was still `1000`.
- This mismatch is noisy but not fatal. It only causes wandb to drop logs until the resumed training step catches back up.

## Hypothesis 2

Checkpointing code is the direct cause of the fatal TPU error.

## Changes to make

Trace the exact failure site through:

- `train.py:521`
- `checkpoint.py:202`
- `jax_utils.py:533-539`

## Results

- `checkpointer.on_step(...)` always calls `broadcast_one_to_all(...)` to synchronize the save decision across hosts.
- The failure happens while `broadcast_one_to_all` is materializing the tiny boolean input tree, before any checkpoint payload is serialized.
- The traceback reaches `jax.Array.__array__` from `np.expand_dims(inp)` in `pre_jit`.
- This means the TPU runtime was already in a bad state before checkpoint writing could begin.
- The second traceback from `force=True` in the `finally:` block is a cleanup retry after the first failure, not a separate root cause.

## Hypothesis 3

This is a TPU/runtime peer mismatch rather than a Python checkpoint bug.

## Changes to make

Inspect the fatal runtime text and correlate it with the control flow.

## Results

- The TPU error is:
  `An unexpected peer shows up in the launch group with a different launch id than the current group leader.`
- That signature points to a multihost runtime mismatch: a stale/bad worker, mixed worker generations, or less likely nondeterministic multihost compilation.
- This is an inference from the logs: because the model already trained and evaluated successfully before the failure, and the crash surfaces only when the next host/device interaction occurs, the more likely problem is worker/runtime state rather than checkpoint file corruption.
- The run uses `v5p-16`, so a multihost TPU issue is plausible.

## Future Work

- [ ] Inspect Iris worker logs for the failing TPU worker IP or task attempt.
- [ ] Map the failing IP to a TPU VM and replace only that node if the worker is bad.
- [ ] If the same slice keeps failing, capture HLO dumps on all workers to rule out multihost nondeterminism.
- [ ] Decide whether resumed runs should keep wandb `resume="allow"` when the checkpoint step is behind the wandb step.
- [ ] Consider skipping the forced checkpoint in `finally:` after fatal `JaxRuntimeError` to avoid the redundant second traceback.
