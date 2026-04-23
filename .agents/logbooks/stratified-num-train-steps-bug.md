# Stratified `num_train_steps` Bug — All 4 Strong-Tier Stratified Runs Affected

## Summary

All 4 stratified strong-tier runs have `num_train_steps` set to **2x their intended value**. They train for double the intended tokens, wasting compute and producing eval data at the wrong steps.

## Evidence

Verified by reading `.executor_info` from GCS for each run:

| Run | GCS Hash | Config `num_train_steps` | Expected Target | Ratio |
|-----|----------|-------------------------|----------------|-------|
| strat-520m-standalone | `-652f5a` | 39672 | 19836 | 2.0x |
| strat-520m-0.5x | `-a2aad9` | 19836 | 9917 | 2.0x |
| strat-520m-2.0x | `-976ccb` | 79345 | 39672 | 2.0x |
| strat-1.2b | `-6dbf91` | 91552 | 45776 | 2.0x |

All qsplit and pilot runs are **correct** — the bug is isolated to the stratified path.

## Root Cause

The stratified training step's `num_train_steps` comes from:
1. `relaunch_strong_tier_cell.py:84` passes `cell.experiment_budget` to `build_stratified_launch_artifacts`
2. `build_launch_artifacts` (line 142-145) passes `experiment_budget` to `create_two_phase_dolma3_dolmino_top_level_experiment`
3. That function sets `num_train_steps = experiment_budget // (batch_size * seq_len)` (line 502 of `two_phase_dolma3_dolmino_top_level.py`)

The strong-tier cells set `cell.experiment_budget = spec.experiment_budget_for_multiplier(multiplier)` which SHOULD be the correctly-scaled value. For the current code, this produces correct results.

**The 4 affected runs were submitted from an older codebase** where the experiment_budget was wrong. The GCS hashes (`-652f5a`, `-a2aad9`, `-976ccb`, `-6dbf91`) reflect that old, incorrect config. New submissions from the current code should produce different (correct) hashes.

## Impact

### Eval data at wrong steps
- Eval runs every 1000 steps. The target step (e.g., 9917, 19836) is NOT a multiple of 1000.
- The force-save at end of training triggers eval at the final step, but with 2x `num_train_steps`, the "final step" is the wrong one.
- Example: strat-520m-0.5x target=9917, but the run's final step would be 19836. Eval at 19836, not 9917.

### Simulated epoching semantics
- `target_budget` (which controls data subsampling) was passed correctly to the experiment.
- So the data mixture is correct for the intended budget — the model sees the right epochs per domain.
- But at step 9917, no eval was triggered (it's not the "final step" for this config).
- The step-9917 eval would only exist if it fell on the `steps_per_eval=1000` schedule, which it doesn't.

### Recoverable checkpoints
| Run | Checkpoints | Usable? |
|-----|------------|---------|
| strat-520m-standalone `-263bc9` (old correct hash) | step 9756 | Yes — can resume from here with correct config |
| strat-520m-standalone `-652f5a` (wrong hash) | step 28888 | No — wrong `num_train_steps`, wrong hash |
| strat-520m-0.5x `-a2aad9` | step 19835 | No — trained past target, no eval at 9917 |
| strat-520m-2.0x `-976ccb` | step 19973 | Possibly — at 50% of the wrong target, but 50% of the correct target too |
| strat-1.2b `-6dbf91` | step 19891 | Possibly — at 43% of the wrong target |

## Actions Taken So Far

1. Set `.executor_status` to `FAILED` on all 4 bad runs to stop them from being restarted
2. Did NOT resubmit yet — waiting for review

## Proposed Fix

1. **Resubmit all 4 stratified experiments** from the current codebase, which should produce correct `num_train_steps`. The new submissions will get different version hashes (correct config).

2. **For strat-520m-standalone**: The old correct hash `-263bc9` has a checkpoint at step 9756. The new submission should produce the same hash and resume from it via `--resume-latest-checkpoints`.

3. **For strat-520m-0.5x, 2.0x, strat-1.2b**: The new submissions will start fresh (step 0) since the old checkpoints are under wrong hashes. However, if `--resume-latest-checkpoints` scans across all hashes (`run_name-*`), it might find the old checkpoints and use them via `initialize_from_checkpoint_path` (which resets step to 0 but loads weights).

4. **Verify the fix**: After the first child runs, check `.executor_info` to confirm `num_train_steps` matches the target.

## Questions for Review

1. Does the current code (`relaunch_strong_tier_cell.py` → `build_stratified_launch_artifacts`) produce correct `num_train_steps` for all 4 cells? I traced through the code and believe so, but the fact that the OLD submissions got it wrong means something changed.

2. For the strat-520m-2.0x and strat-1.2b runs that have checkpoints at 50%/43%: can we salvage their training progress by resuming from the old checkpoint under the new (correct) hash? This would use `initialize_from_checkpoint_path` which loads weights but resets step to 0 — losing the step counter but preserving the learned weights. Is this acceptable, or should we start fresh?

3. Should we add an assertion in `build_launch_artifacts` to verify `num_train_steps` matches the expected value from `experiment_budget / (batch_size * seq_len)`?
