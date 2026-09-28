# Debugging log for HF eval cache metadata rate limiting

Fix the strong-tier and related evaluation failures where workers still hit Hugging Face Hub rate limits during `lm_eval` task loading even after syncing the eval dataset cache from GCS.

## Initial status

CC reported:

- `130M` and `300M` strong-tier cells completed.
- all `520M` strong-tier cells were blocked on `lm_eval` task loading.
- the earlier offline-mode change (`9deacab80`) only changed the failure mode:
  - first we had HF `429` rate limiting,
  - then we had `OfflineModeIsEnabled` because the synced cache did not actually contain everything needed for offline task resolution.

The key symptom was:

- workers successfully synced `eval_datasets_cache_path`,
- then `tasks.get_task_dict(["mmlu", ...])` still touched the HF Hub API,
- and many concurrent workers exhausted the shared HF quota.

## Hypothesis 1

The cache step only uploads the datasets cache tree, not the full Hugging Face cache root. That leaves out HF Hub metadata and/or dataset modules needed for `lm_eval` task resolution.

## Changes to make

- Inspect `lib/marin/src/marin/evaluation/eval_dataset_cache.py`.
- Check where cached content is written locally and what is uploaded to GCS.

## Results

Confirmed.

Before the fix:

- `save_eval_datasets_to_gcs(...)` downloaded datasets with `cache_dir=...`
- then uploaded only that local cache directory
- `load_eval_datasets_from_gcs(...)` synced only into `~/.cache/huggingface/datasets`

That meant the cache path only guaranteed dataset payloads, not:

- HF Hub metadata cache (`hub/`)
- HF modules cache (`modules/`)

So workers still needed Hub resolution during `tasks.get_task_dict(...)`.

## Hypothesis 2

If the cache step materializes and uploads a full HF cache root, and workers sync that full root, `lm_eval` task loading can run fully offline again.

## Changes to make

- In `lib/marin/src/marin/evaluation/eval_dataset_cache.py`:
  - add an explicit HF cache-root context (`HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, `HF_MODULES_CACHE`)
  - download datasets into that cache root
  - warm task metadata centrally with `tasks.get_task_dict(...)`
  - upload the full cache root, not just `datasets/`
  - extend the manifest with layout/version fields indicating whether hub/modules caches are present
  - sync workers into the full HF cache root
- In `lib/levanter/src/levanter/eval_harness.py`:
  - use full offline mode only if the synced manifest says the full cache is present
  - otherwise fall back to datasets-only mode for legacy/incomplete caches

## Results

Implemented:

- full HF cache-root upload/sync
- task-metadata warming in the cache step
- manifest versioning with `supports_full_offline_task_loading()`
- conditional full offline mode in Levanter based on the manifest

While testing, another latent bug surfaced:

- `eval_dataset_cache.py` passed `logger=` into `call_with_hf_backoff(...)`
- but `call_with_hf_backoff(...)` does not accept that parameter

That was removed at both call sites.

Targeted regressions now pass:

- `tests/evals/test_eval_dataset_cache.py`
- `lib/levanter/tests/test_eval_harness.py` targeted cache/offline slice

## Conclusion

The real root cause was incomplete cache provenance, not just missing backoff.

The old GCS cache layout only shipped `datasets/`, so workers still had to hit HF Hub metadata paths. The correct fix is to pre-materialize and ship a full Hugging Face cache root and only enable full offline mode when that full cache is present.

## Future work

- [ ] Consider adding a small manifest checksum / task-list fingerprint so stale cache contents are easier to detect explicitly.
- [ ] Consider logging the cache layout version and offline mode selection in worker startup logs for faster future triage.
