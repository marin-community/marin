# Debugging log for mixture weight logging

Investigate why mixture weights no longer appear in W&B for domain-phase-mix training runs.

## Initial status

Recent two-phase-many runs do not show `mixture/weight/*` metrics in W&B, even though that used to be expected behavior.

## Hypothesis 1

Mixture weights are still logged, but under a different key or nested config path.

## Changes to make

- Inspect recent W&B runs for `mixture/*` history keys.
- Inspect run config keys for phase-weight or mixture-weight fields.

## Future Work

- [ ] Add a regression test for mixture-weight hook installation.
- [ ] Decide whether weights should also be logged as config/hparams, not only runtime metrics.

## Results

This hypothesis is false. Recent runs like `baseline_proportional-bac80b` have no `mixture/*` history keys at all, and their W&B config does not contain flattened phase-weight entries either.

## Hypothesis 2

The logging hook in `train_lm.py` is no longer seeing the training dataset as a `MixtureDataset` because the dataset is now wrapped.

## Changes to make

- Inspect `train_lm.py` mixture hook condition.
- Inspect `LmDataConfig.train_set()` return type and wrapper structure.

## Results

This hypothesis is true. `train_lm.py` only installs the hook when `isinstance(train_dataset, MixtureDataset)`. But `LmDataConfig.train_set()` now returns `NamedLmDataset(mixture, Pos)`, not the bare `MixtureDataset`. So the hook never runs. The mixture still exists under `train_dataset.dataset`; it is just wrapped.

## Hypothesis 3

A small local fix can restore the old behavior by unwrapping dataset wrappers until a nested `MixtureDataset` is found.

## Changes to make

- Add a helper in `train_lm.py` to unwrap common dataset wrappers.
- Use the recovered inner `MixtureDataset` for stage-weight logging.
- Add a regression test.

## Results

Implemented. `train_lm.py` now unwraps common dataset wrappers via `_find_nested_mixture_dataset(...)` and installs the existing mixture-stage logging hook when an inner `MixtureDataset` is found. This restores `mixture/weight/*` logging for LM runs whose training dataset is wrapped in `NamedLmDataset`. A regression test was added in `lib/levanter/tests/test_train_lm.py`.
