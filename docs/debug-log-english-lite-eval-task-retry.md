# Debugging log for 300M English-lite eval task retry

## Initial status

The 300M English-lite SNR job `/calvinxu/dm-300m-english-lite-snr-20260429-013845` failed on 248 of 253 children. The reported errors looked like bad task configs:

- `ValueError: Failed to load task {'task_alias': 'hellaswag_5shot', 'num_fewshot': 5}`
- `ValueError: Failed to load task {'task_alias': 'openbookqa_0shot', 'num_fewshot': 0}`
- `TypeError: 'NoneType' object is not iterable`

The parent config contained valid `EvalTaskConfig` rows with `name`, `task_alias`, and `num_fewshot`, so the first hypothesis was that the failure happened after conversion to Levanter/lm-eval task dictionaries.

## lm-eval mutates task dictionaries during load

Local reproduction showed that `lm_eval.tasks.get_task_dict([task_dict], manager)` mutates the input dictionary by removing the `task` key. This is harmless on a successful first attempt, but Levanter wrapped task loading in `_call_with_retry` and reused the same dictionary across retries.

If the first task-load attempt fails transiently after mutation, the next retry receives a dictionary without `task`, producing the misleading `NoneType` / missing-task cascade seen in the failed job.

## Fix

Changed `LmEvalHarnessConfig._get_task_and_rename` to pass a deep copy of the task dictionary into `lm_eval.tasks.get_task_dict` on every retry attempt.

This preserves the original task specification for subsequent retries and for useful error reporting.

## Validation

- Local smoke reproduced the mutation-then-transient-failure case and passed after the fix.
- English-lite launcher dry run prepared 252 eligible eval steps over 253 candidate checkpoints and 17 task aliases. One row (`baseline_stratified`) was correctly deferred because its checkpoint is in `us-central1`, not the east5-local checkpoint region.

## Follow-up

- Resubmit the English-lite eval with a new suffix/prefix so all failed children use the fixed Levanter retry behavior.
- Do not retry the old failed job without this code patch; it will keep failing under transient task-load errors.

## Resubmission notes

Three parent submissions failed before dispatch and did not run eval children:

- `/calvinxu/dm-300m-english-lite-snr-r3-20260429-214512`: parent OOM from launching with `uv run --with torch` under the default 1GB coordinator memory.
- `/calvinxu/dm-300m-english-lite-snr-r3-20260429-214809`: parent could not rebuild state because generated `metrics_wide.csv` is ignored and not present in the remote bundle.
- `/calvinxu/dm-300m-english-lite-snr-r3-state-20260429-215331` and `/calvinxu/dm-300m-english-lite-snr-r3-state-20260429-215646`: state was supplied from GCS, but the parent was not pinned to east5, so TPU child region validation failed.

The active corrected resubmission is:

`/calvinxu/dm-300m-english-lite-snr-r3-east5-20260429-221817`

It uses:

- Patched Levanter retry behavior.
- GCS state CSV at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_300m_english_lite_evals_20260429-r3/state/300m_english_lite_eval_state.csv`.
- Parent pinned to `us-east5-a`.
- Child evals pinned to `v5p-8` in `us-east5-a`.
- `max_concurrent=256`.

After 6.5 minutes the parent was still running with zero failures and child logs showed task loading/eval progress. The old malformed task-load signature was absent.
