# Debugging log for qsplit240 300M eval dataset rate limiting

Investigate why several children in the expanded-task qsplit240 300M swarm fail with
`ValueError: Failed to load task {'task_alias': 'mmlu_5shot', 'num_fewshot': 5}`.

## Initial status

The parent Iris job `/calvinxu/dm-qsplit240-300m-6b-20260401-031236` launched
successfully after increasing parent memory to `32GB`, and multiple TPU child
jobs entered `JOB_STATE_RUNNING`. Three children later failed while loading
`mmlu_5shot`.

## Hypothesis 1

The failures are caused by Hugging Face rate limiting during task loading, not by
TPU allocation, training configuration, or the mixture itself.

## Changes to make

- Inspect failed child logs for the precise traceback.
- Compare with healthy child logs to confirm environment parity.
- Check whether the launcher wires through the existing eval dataset cache
  support from `levanter.eval_harness`.

## Results

- Failed child logs showed:
  - initial `HfHubHTTPError: 429 Client Error: Too Many Requests`
  - repeated retry attempts in `levanter.eval_harness._call_with_retry`
  - eventual `TypeError: 'NoneType' object is not iterable` inside
    `lm_eval.tasks._get_group_and_subtask_from_config`
  - final wrapper error:
    `ValueError: Failed to load task {'task_alias': 'mmlu_5shot', 'num_fewshot': 5}`
- Healthy child logs showed normal JAX distributed startup, W&B init, and cache
  loading, which rules out worker-image drift or missing deps.
- The launcher did not create a `cache_eval_datasets_step` and did not pass an
  `eval_datasets_cache_path` into the experiment, even though
  `levanter.eval_harness` already supports syncing evaluation datasets from GCS
  specifically to avoid Hugging Face API rate limiting.

## Hypothesis 2

The correct fix is to pre-cache the exact expanded task suite to a Marin GCS path
 in the local region and point the experiment at that cache, following the same
 pattern already used by the StarCoder and validation launchers.

## Changes to make

- Patch `launch_two_phase_many_qsplit240_300m_6b.py` to:
  - define a dedicated eval-datasets cache path
  - map it to the local Marin region bucket
  - pass it through `eval_datasets_cache_path`
  - add `create_cache_eval_datasets_step(...)` before training steps

## Results

- Patched `experiments/domain_phase_mix/launch_two_phase_many_qsplit240_300m_6b.py`
  to pre-cache the expanded task set and wire the cache into the experiment.

## Future Work

- [ ] Decide whether to let the current run continue and recover only failed jobs, or stop and relaunch from the patched launcher.
- [ ] Add a regression test that the launcher includes the eval-dataset cache path and cache step.
