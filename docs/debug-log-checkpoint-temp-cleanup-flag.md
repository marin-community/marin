# Debugging log for checkpoint temp cleanup flag

Add an explicit control for callback-based temporary-checkpoint deletion so the next 500-step RL run can isolate checkpoint-save cost from prior-temp cleanup.

## Initial status

The current `delete_old_temp_checkpoints` flag only affects startup handling of a temporary checkpoint discovered from a prior attempt. It does not disable the callback that deletes the previously saved temporary checkpoint after each successful new save.

Timing evidence from `scratch/ckpt_monitor_log.md` shows that cleanup is part of the slowdown story, but it is not the whole story:

- `filesystem_ready`: `0 -> 61 -> 100 -> 177 -> 248 -> 326 -> 351s`
- `tensorstore_serialize`: `~90 -> 190 -> 255 -> 324 -> 401 -> 397 -> 538s`
- `metadata_write`: `~14 -> 74 -> 113 -> 211 -> 313 -> 293 -> 316s`

So we need a clean knob that removes callback cleanup without changing the rest of the checkpoint path.

## Hypothesis 1

The code is conflating two different behaviors:

- startup cleanup of an inherited temporary checkpoint
- in-run cleanup of the previous temporary checkpoint after each new save

## Changes to make

- Add a new `Checkpointer` / `CheckpointerConfig` flag: `delete_previous_temporary_checkpoint_after_save`
- Gate the callback deletion path on that flag
- Expose the flag in the direct RL experiment launchers used for the next 500-step run
- Add tests that distinguish startup cleanup from callback cleanup

## Future Work

- [ ] Add a lightweight checkpoint monitor mode for long runs without the full debug checkpointer
- [ ] Decide whether the next clean 500-step run should also change checkpoint frequency
- [ ] Measure whether metadata-write slowdown persists when callback cleanup is disabled

## Results

Implemented:

- `Checkpointer` and `CheckpointerConfig` now accept
  `delete_previous_temporary_checkpoint_after_save`
- the callback that previously always deleted the prior temporary checkpoint is
  now gated on that flag
- the direct RL experiment launchers expose
  `--delete-previous-temporary-checkpoint-after-save` /
  `--no-delete-previous-temporary-checkpoint-after-save`
- tests now distinguish:
  - startup cleanup of an inherited temporary checkpoint
  - in-run callback cleanup of the previous temporary checkpoint

Validation:

- `uv run pytest -q lib/levanter/tests/test_checkpoint.py`
- `./infra/pre-commit.py --fix lib/levanter/src/levanter/checkpoint.py lib/levanter/tests/test_checkpoint.py experiments/exp_iris_rl_regression_direct_gcs_prod.py experiments/xp_iris_rl_regression_direct_gcs_prod.py docs/debug-log-checkpoint-temp-cleanup-flag.md`

Outcome:

- callback deletion is now an isolated experiment knob
- the next clean 500-step RL run can remove prior-temp cleanup from the hot path
  without re-enabling the heavy debug checkpointer
