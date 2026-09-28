# Debugging log for missing lmeval recovery

Investigate why some successful seed-study runs did not produce recoverable `lm_eval` outputs in post-run analysis.

## Initial status

The 12-run `run_00097` seed study finished successfully, but determinism analysis only recovered 8/12 runs through W&B tag queries. Four runs were marked `not_found` despite successful training and later manual MMLU recovery.

## Hypothesis 1

The missing runs never executed `lm_eval`, so no results existed to recover.

## Changes to make

- Inspect original Ray job logs for the four missing runs.
- Check W&B directly by run id for summary metrics.
- Inspect local replicate artifacts under the checkpoint prefixes.

## Future Work

- [ ] Add a post-run collector fallback that does not rely solely on W&B tag queries.
- [ ] Persist enough run metadata locally to recover W&B runs by id.
- [ ] Persist `lm_eval` outputs locally, not only in W&B.

## Results

This hypothesis is false. The original Ray logs show the missing runs did execute the eval harness and log `Finished running eval harness`, `Logged report to tracker`, `Finishing wandb run`, and W&B sync lines. Direct W&B lookup by run id also returns `lm_eval/mmlu_5shot/bpb` for all four runs. The issue is recoverability, not training/eval execution.

## Hypothesis 2

The runs exist in W&B, but the collector misses them because it queries by experiment tag and some runs lost those tags.

## Changes to make

- Compare direct W&B lookup by run id with tag-filtered W&B listing.
- Inspect the training config and tracker replicate file to see what tags were intended.
- Check whether local replicate artifacts contain enough metadata to recover the missing runs without W&B tags.

## Results

This hypothesis is true. The four missing runs are present and finished in W&B, but their `tags` field is empty. The eight visible runs still have the expected experiment tag `pinlin_calvin_xu/data_mixture/ngd3dm2_run00097_seed_study`. The current collector in `experiments/domain_phase_mix/determinism_analysis.py` uses `query_wandb_runs(..., tags=[experiment_name])`, so the four empty-tag runs are never seen.

The training config recorded in `tracker_metrics.jsonl` shows the intended W&B tags were correct for the missing runs, so the empty tags appear to be a W&B-side persistence/visibility failure rather than a deterministic truncation bug in our code.

## Hypothesis 3

Even if W&B tag queries fail, our local replicate artifacts should be enough to recover `lm_eval`.

## Changes to make

- Inspect `tracker_metrics.jsonl` contents for missing and non-missing runs.
- Check whether `lm_eval` metrics or run metadata are written locally.

## Results

This hypothesis is false. `tracker_metrics.jsonl` currently contains neither the W&B run id nor the W&B tags, and its summary omits `lm_eval/*` keys entirely. The checkpoint directory only contains `checkpoints/eval_metrics.jsonl` for perplexity-style metrics plus the exported HF checkpoint. So a successful run can still be impossible to recover locally for `lm_eval` if W&B listing misses it.

## Hypothesis 4

There are concrete code bugs worth fixing on our side even if W&B dropped the tags.

## Changes to make

- Patch result collection to avoid hard dependence on W&B tag listing.
- Persist run metadata and `lm_eval` outputs locally for future runs.

## Results

Confirmed. Two concrete robustness bugs exist on our side:

1. `collect_manifest_results` is too brittle because it relies exclusively on tag-filtered W&B listing.
2. The local replicate path is insufficient for recovery because it does not persist W&B run metadata or `lm_eval` results.

Likely fixes:
- add W&B metadata (`id`, `name`, `tags`) to `tracker_metrics.jsonl`;
- mirror `lm_eval` artifacts into the replicate path;
- add a name- or manifest-based fallback in determinism collection instead of `status=not_found` when tag lookup fails.
