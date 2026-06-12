# Debugging log for PR 6216 Marin unit failure

Fix the Marin unit test failure on PR 6216.

## Initial status

GitHub Actions run 27321671434 failed in the `marin-unit` job. The failing tests all import Grug launch or dispatch code and fail with:

`ImportError: cannot import name 'extras_for_resources' from 'marin.training.training'`

## Hypothesis 1

`extras_for_resources` moved out of `marin.training.training` into `marin.training.run_environment`, and `experiments/grug/dispatch.py` has a stale import.

## Changes to make

Update `experiments/grug/dispatch.py` to import `extras_for_resources` from `marin.training.run_environment` while keeping `resolve_training_env` from `marin.training.training`.

## Results

Focused Grug tests pass:

`uv run pytest tests/test_grug_variant_contracts.py tests/test_grug_launch_checkpoint_paths.py -q`

Result: `18 passed, 1 warning in 15.60s`.

The other CI failure entry points that were blocked by the same import also pass:

`uv run pytest tests/datakit/testbed/test_train.py tests/test_validate_canary_metrics.py tests/test_dry_run.py -q`

Result: `93 passed, 187 skipped in 17.96s`.

## Future work

- [x] Run the focused Grug tests that failed in CI.
