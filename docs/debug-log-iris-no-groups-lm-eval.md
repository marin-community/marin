# Debugging log for iris no-groups lm_eval failure

The goal is to understand why the Iris-submitted `no-groups` domain-mix validation still fails with `ModuleNotFoundError: No module named 'lm_eval'` after the parent wrapper was updated to include `marin:eval`.

## Initial status

The resubmitted Iris parent job launched correctly and the wrapper reported both:

- `--extra marin:tpu`
- `--extra marin:eval`

But the training child still failed with:

- `ModuleNotFoundError: No module named 'lm_eval'`

The job tree also showed that tokenization children succeeded, so the remaining failure was isolated to the training child environment.

## Hypothesis 1

The parent Iris job has the correct extras, but Fray/Marin rebuilds a fresh child environment for the actual training job and drops `eval` when it computes runtime extras.

## Changes to make

Inspect:

- `lib/marin/src/marin/training/training.py`
- `experiments/defaults.py`

Trace whether `eval_harness` is present in the generated `TrainLmConfig`, and whether the child training submission preserves it when constructing `create_environment(..., extras=...)`.

## Results

This hypothesis was correct.

Findings:

- `experiments/defaults.py` sets `TrainLmConfig.eval_harness` whenever `eval_harness_tasks` is non-empty.
- `lib/marin/src/marin/training/training.py::_prepare_training_run` then rebuilt the child-job extras from scratch.
- That logic only added accelerator extras:
  - `tpu`
  - `gpu`
- It did not add `eval`, even when `TrainLmConfig.eval_harness` was configured.

This means:

- the parent Iris wrapper job had `marin:eval`
- the actual training child did not
- the child then failed when internal LM-eval code tried to import `lm_eval`

Patch:

- replace device-only runtime extras logic with training-aware logic
- include `eval` whenever:
  - the job is LM training, and
  - `train_config.eval_harness is not None`

Added regression test:

- `tests/test_training.py::test_prepare_training_run_adds_eval_extra_for_lm_eval_harness`

## Hypothesis 2

The zone mismatch report may be caused by the controller tunnel logs rather than actual child-job placement, but the training child still needs an explicit `us-east5-a` constraint because the data is zone-local.

## Changes to make

Add `zone` support to `fray.v2.types.ResourceConfig` and map it to Iris `zone_constraint(...)`, then update the no-groups launcher to request `us-east5-a`.

## Results

Implemented:

- `ResourceConfig.zone`
- Iris constraint conversion for `zone`
- a Fray unit test
- `launch_two_phase_many_genericfamily_no_groups_baseline.py` now requests `regions=[\"us-east5\"], zone=\"us-east5-a\"`

This addresses placement locality independently from the extras fix.

## Future Work

- [ ] Consider whether other LM training entry points that use internal evals should declare `eval` more explicitly in configs or launchers.
- [ ] Decide whether more `domain_phase_mix` launchers should be upgraded from region-only to explicit zone pinning.
- [ ] Add a small end-to-end Iris integration test that asserts child-job extras include `eval` when `eval_harness` is configured.
