# Debugging log for stratified num_train_steps bug

Review the claimed `num_train_steps` bug for strong-tier stratified launches and
check it against the simulated-epoching contract.

## Initial status

CC reported that all four strong-tier stratified runs at scales using
`batch_size=256` were launched with `num_train_steps` equal to 2x the intended
value. The claim was that the data subsampling target budget remained correct,
but the training step count was wrong.

## Hypothesis 1

The bug is in the current stratified launch path, not just in some old
historical submission hash.

## Changes to make

No code changes yet. Review these files:

- `experiments/domain_phase_mix/relaunch_strong_tier_cell.py`
- `experiments/domain_phase_mix/launch_two_phase_many_stratified_baseline.py`
- `experiments/domain_phase_mix/scaling_study_recipes.py`
- `experiments/domain_phase_mix/two_phase_dolma3_dolmino_top_level.py`
- `experiments/domain_phase_mix/experiment.py`

## Results

Confirmed. The current stratified path is internally inconsistent:

- `build_run_spec()` in
  `launch_two_phase_many_stratified_baseline.py` computes
  `num_train_steps=experiment_budget // (spec.batch_size * spec.seq_len)`,
  which is correct.
- But `build_launch_artifacts()` then calls
  `create_two_phase_dolma3_dolmino_top_level_experiment(...)` without passing
  `batch_size` or `seq_len`.
- That top-level experiment defaults to `BATCH_SIZE = 128`, so the actual
  training experiment uses `num_train_steps=experiment_budget // (128 * 2048)`.

For the affected scales:

- `520M`: intended `batch_size=256`, actual default `128`, so steps double
  (`19836 -> 39672`, `9918 -> 19836`, `39672 -> 79345` after flooring).
- `1.2B`: intended `45776`, actual `91552`.

This matches the reported `.executor_info` values almost exactly and explains
why the issue is isolated to stratified `520M` and `1.2B`, while qsplit and
smaller stratified cells are unaffected.

The simulated-epoching semantic is also preserved exactly as reported:

- `MixtureExperiment.create_training_step()` passes `target_budget` into
  `simulated_epoching_train(...)`.
- The bad step count changes how long training runs and where final eval/checkpoint
  happen.
- It does **not** change the target-budget-based subsampling law itself.

So the bug is valid, the data impact explanation is valid, and resubmitting from
the current code would still be wrong until the launcher is fixed.

## Follow-up

- [x] Patched the stratified launch path to pass `batch_size=spec.batch_size`
      and `seq_len=spec.seq_len` into
      `create_two_phase_dolma3_dolmino_top_level_experiment(...)`
- [x] Added an assertion that the training experiment step count matches the
      manifest/run-spec step count
- [ ] Re-evaluate whether any partial checkpoints can be safely reused once the
      launch path is fixed
