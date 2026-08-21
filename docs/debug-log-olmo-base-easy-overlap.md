# OLMo Base Easy Overlap Rerun Debug Log

## Summary

The parent job `/calvinxu/dm-qsplit240-olmo-base-easy-overlap-20260405-222333`
failed because Levanter lm-eval steps were executed locally in executor worker
threads instead of as remote Fray jobs. Multiple threads then called
`jax.distributed.initialize()` in the same Python process, producing
`distributed.initialize should only be called once`.

## Evidence

- Parent failed with `RuntimeError: 240 step(s) failed`.
- Parent logs repeatedly showed:
  - `Error running eval harness: distributed.initialize should only be called once.`
- The failing steps were all
  `evaluation/lm_evaluation_harness_levanter/lmeval_debug_*`.

## Root Cause

`evaluate_levanter_lm_evaluation_harness(...)` built a plain `ExecutorStep`
with `fn=evaluate` and `launch_with_ray=False`.

That means:

- the executor ran each eval inline in a local worker thread
- Levanter eval ran in-process inside the parent
- concurrent eval steps shared one interpreter
- JAX distributed init collided across threads

The nested launch bug was reduced earlier by setting `launch_with_ray=False`,
but the step itself still needed to be remote-dispatched.

## Fix

Changed `evaluate_levanter_lm_evaluation_harness(...)` to wrap `evaluate` with
`remote(...)` using the requested TPU `ResourceConfig` and dependency groups
`["eval", "tpu"]`.

This makes each Levanter eval step run in its own remote Fray job while still
running the evaluator in-process inside that job.

## Validation

- `uv run pytest tests/evals/test_lm_eval.py -q`
- `uv run pytest tests/test_domain_phase_mix_determinism.py -q -k 'olmo_base_easy_overlap'`
- `CI=1 uv run python -m experiments.domain_phase_mix.launch_two_phase_many_qsplit240_olmo_base_easy_overlap_rerun`

All passed after the patch.
