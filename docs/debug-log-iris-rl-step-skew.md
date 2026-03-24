# Debugging log for iris-rl-step-skew

Investigate why trainer logs repeatedly show:
`Skipping stale rollout batch (rollout_step=1, current_step=0)` in a 500-step Iris RL run.

## Initial status

User report indicated a perceived sequencing bug:
- Rollout worker writes `rollout_step=1`
- Trainer remains at `current_step=0`
- Replay buffer rejects these as stale/future and training appears stuck.

Code under inspection:
- `lib/marin/src/marin/rl/replay_buffer.py`
- `lib/marin/src/marin/rl/train_worker.py`
- `lib/marin/src/marin/rl/rollout_worker.py`
- `lib/marin/src/marin/rl/orchestration.py`

Cluster logs inspected:
- `/ahmed/iris-rl-500step-v2/rl-iris-rl-direct-20260323-025037-train`
- `/ahmed/iris-rl-500step-v2/rl-iris-rl-direct-20260323-025037-rollout-0`

## Hypothesis 1

There is an off-by-one bug in normal steady-state hook ordering (trainer step propagation vs rollout stamping).

## Changes to make

No code changes. Validate with code-path reading + production logs:
- verify rollout stamp source (`weight_step=self._current_weight_step`)
- verify trainer step updates (`replay_buffer.set_current_step(info.step)`)
- verify Levanter callback step semantics (`StepInfo.step = state.step - 1`)

## Results

Steady-state logic is consistent:
- Rollout batches are stamped from transferred weight ID, not local rollout loop counter.
- Trainer `current_step` is updated from Levanter `info.step`, where `info.step` is already the completed step.
- No direct off-by-one bug found in normal execution.

## Hypothesis 2

The mismatch is caused by trainer restart after failure, while rollout worker continues with newer weights.

## Changes to make

No code changes. Validate timeline from Iris logs.

## Results

Confirmed from logs:

1. Trainer ran and progressed:
- step 0 completed, then transferred step 1 weights.
- step 1 completed at ~03:11.

2. Trainer crashed and was retried:
- log line: `Container was OOM killed by the kernel` at ~03:18.
- `job list` shows trainer child `failure_count=1`, indicating retry/restart occurred.

3. Restart did not recover a valid checkpoint:
- post-restart logs: `No checkpoints found ...`
- trainer effectively restarted from initial flow (`serve_weights(-1)`, then step 0).

4. Rollout worker did not reset and continued on newer weight lineage:
- rollout logs show `Received new weights from step 1` and continued generating step-1 rollouts.

5. Replay buffer behavior then matched current code by design:
- trainer at `current_step=0` rejected incoming `rollout_step=1` as future.
- repeated `Skipping stale rollout batch (rollout_step=1, current_step=0)` lines observed.

Conclusion:
- Root cause is restart skew after trainer OOM + retry, not a simple steady-state off-by-one bug.
- The replay-buffer `±delay` acceptance patch masks this restart skew and can mix rollouts from a different model lineage.

## Hypothesis 3

Arrow Flight coordinator stale-check prevents rollback updates after restart.

## Changes to make

Code changes:
- `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
  - `ArrowFlightCoordinator.update_server()` now accepts rollback weight IDs and only ignores exact duplicates.
- `lib/marin/src/marin/rl/replay_buffer.py`
  - Restored one-directional freshness check (`max_step = current_step`) to keep deterministic anti-future behavior.
- `tests/rl/test_weight_transfer.py`
  - Added `test_arrow_flight_coordinator_accepts_rollback_weight_ids`.

## Results

Validation:
- `./infra/pre-commit.py --fix -- <touched files>`: OK
- `uv run pytest -q tests/rl/test_weight_transfer.py -k "multiple_weight_updates or coordinator_accepts_rollback"`:
  - `3 passed, 3 deselected`

This fix aligns behavior with restart recovery:
- if trainer restarts and re-serves `-1`/`0`, coordinator now publishes those IDs;
- rollout workers can actually receive rollback weights;
- replay buffer can keep rejecting future rollouts without deadlocking the system on stale coordinator state.

## Hypothesis 4

5-step "robustness" was primarily due to a much lighter runtime envelope and not hitting checkpoint-save windows, while 500-step run repeatedly enters checkpoint-save OOM.

## Changes to make

No code changes. Validate from linked W&B runs and corresponding Iris logs:
- 5-step train run:
  - `marin-community/marin_iris_rl_debug/iris-rl-direct-20260323-013808-train`
- 5-step rollout run:
  - `marin-community/marin_iris_rl_debug/iris-rl-direct-20260323-013808-rollout-0`
- 500-step train run:
  - `marin-community/marin_post_training/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-153458-train`
- 500-step rollout run:
  - `marin-community/marin_post_training/llama-3.1-8bi-math-lr=2e-6-bs=1024-20260323-153458-rollout-0`

## Results

Confirmed from W&B + Iris logs:

1. 5-step run exact train settings (from W&B config):
- `num_train_steps=5`
- `train_batch_size=64`
- `per_device_parallelism=16`
- `checkpointer.save_interval=10m`

2. 5-step run rollout behavior (from Iris logs + W&B summary):
- weight sync: `max_weight_transfer_wait_time=300` (blocking)
- inference: `max_model_len=1024`, generation `max_tokens=512`
- per-step rollout payload scale: `Generated rollout with 4 groups ...`
- W&B rollout summary: `total_count=64`, `last_size_bytes=329,884`, `cumulative_batch_count=5`

3. 500-step run exact train settings (from W&B config + logs):
- `num_train_steps=500`
- `train_batch_size=1024`
- `per_device_parallelism=16`
- `checkpointer.save_interval=10m`
- weight sync: `max_weight_transfer_wait_time=0` (non-blocking)

4. 500-step run rollout behavior (from Iris logs + W&B summary):
- inference: effectively `max_model_len=2048` (`max_input_tokens=1024`, `max_output_tokens=1024`)
- generation includes `500`-prompt eval phases plus `64`-prompt rollout phases, `max_tokens=1024`
- W&B rollout summary: `total_count=1024`, `last_size_bytes=6,198,068`, `cumulative_batch_count=10`

5. OOM/checkpoint coupling in 500-step run remains tight:
- checkpoint save start then OOM kill after ~45-50s (repeated)
- restart then `No checkpoints found ...` and replay skew cleanup

Interpretation:
- model weights are similar, but runtime memory pressure is not:
  - much larger train batch/tokens,
  - much larger rollout payloads,
  - non-blocking sync/restart churn,
  - checkpoint save overlap.
- 5-step run did not materially exercise the same failure surface because it stayed small and short.

## Hypothesis 5

Across the full `/ahmed/iris-rl*` history, true runtime OOM appears only in long-running high-memory configurations and always checkpoint-coupled.

## Changes to make

No code changes for this hypothesis. Build a run-history census:
- enumerate all `/ahmed/iris-rl*` jobs from `iris job list --json`
- classify each top-level experiment by dominant terminal cause
- spot-check train logs for experiments that reached active training.

## Results

Historical census:
- total matching jobs: `84` across many top-level experiments.
- most early failures were setup/build/dependency issues (e.g., `libcublas`, `torch` resolver, missing module), not runtime OOM.
- many experiments were manually terminated during capacity/routing tests.

Experiments confirmed to hit runtime OOM:
1. `/ahmed/iris-rl-500step`
   - `Saving checkpoint at step 4` -> OOM kill ~45s later.
2. `/ahmed/iris-rl-500step-v2`
   - `Saving checkpoint at step 2` -> OOM kill ~49s later.
3. `/ahmed/iris-rl-500step-v12`
   - repeated save->OOM cycles (step 4/5 windows).

Shared traits in those OOM runs:
- `train_batch_size=1024`
- long-horizon run (`500` steps)
- `max_weight_transfer_wait_time=0`
- checkpointing enabled (10m interval)
- restart path shows `No checkpoints found ...` after OOM, indicating checkpoint recovery failure.

Control run that did *not* OOM:
- `/ahmed/iris-rl-v5p-3` (5-step debug)
  - much smaller runtime envelope (`train_batch_size=64`, smaller token lengths)
  - no checkpoint-save event observed during the short run.

Updated interpretation:
- this is not a random TPU node flake and not a model-weight identity issue.
- strongest explanation is checkpoint-overlap memory pressure in the heavy 500-step configuration, with restart/recovery churn as a secondary consequence.

## Future Work

- [ ] Decide restart policy: fail fast on trainer failure (`max_retries_failure=0`) vs robust resume.
- [ ] If retries remain enabled, guarantee recoverability:
  - checkpoint often enough to recover latest served weights, and
  - ensure checkpoint writes are atomic/valid before retry continues.
- [ ] Add explicit detector/log when rollout weight step is ahead of trainer step after restart (clear operator guidance).
