# Debugging log for nextgen east5 baselines

Debugging why the east5 baseline canary diverged: `baseline_unimax` succeeded while `baseline_proportional` failed, and the nextgen fit stage then failed to find the `lm_eval/mmlu_5shot/bpb` objective metric.

## Initial status

The retry baseline job `ray-run-calvinxu-nextgen-east5-baselines-retry-20260315-224006` launched two runs on `us-east5-a`.

- `baseline_unimax-6b6af9` succeeded and wrote full training artifacts.
- `baseline_proportional-80b8b9` failed with TPU retries ending in `RuntimeError: No accelerator found. Please run on a TPU or GPU.`
- The nextgen loop then failed in fit with `ValueError: Objective metric 'lm_eval/mmlu_5shot/bpb' missing from run table`.

## Hypothesis 1

The fit failure is caused by a collection bug: the successful run did log `lm_eval/mmlu_5shot/bpb`, but `collect_new` only copied numeric eval metrics from W&B summary and did not backfill from tracker metrics or history when summary was incomplete.

## Changes to make

- Inspect `experiments/domain_phase_mix/nextgen/collect.py` and reuse existing tracker-metrics readers if possible.
- Patch collection so the objective metric, and ideally all final eval metrics, can be recovered from run-local artifacts when W&B summary is incomplete.

## Future Work

- [ ] Add a focused regression test for missing-summary / present-tracker-metrics collection.
- [ ] Consider whether imported legacy sources should use the same tracker-metrics backfill path.

## Results

Confirmed. The successful run's `tracker_metrics.jsonl` contains `lm_eval/mmlu_5shot/bpb`, but `collect_new/new_runs.json` recorded `"metrics": {}`. The bug is in nextgen collection, not in the run itself.

Implemented a fix in `experiments/domain_phase_mix/nextgen/collect.py` so `collect_new_run_data()` backfills the objective metric from W&B history when the summary row is missing it. Added a regression test covering the exact missing-summary / present-history case.

## Hypothesis 2

The proportional baseline failure is an infrastructure retry bug: a TPU slice launched without an accelerator-visible environment, and Fray retried it as a generic run failure instead of evicting that bad slice and forcing rescheduling onto a healthy slice.

## Changes to make

- Inspect `lib/fray/src/fray/v2/ray_backend/tpu.py` retry classification for worker/runtime errors.
- Patch TPU retry handling so "No accelerator found" is treated as infrastructure failure/preemption-equivalent and the bad slice is discarded before retry.

## Future Work

- [ ] Add a focused TPU retry regression test for accelerator-missing startup failures.
- [ ] Improve TPU failure logs to include slice identity in the final raised error.

## Results

Confirmed. The proportional baseline failed with `RuntimeError: No accelerator found. Please run on a TPU or GPU.`, raised inside the remote TPU worker and then retried by Fray until the retry budget was exhausted.

Implemented a fix in `lib/fray/src/fray/v2/ray_backend/tpu.py` that:

- classifies TPU startup errors such as "No accelerator found" as infrastructure-retryable,
- marks the affected slice actor as failed so the next retry rebuilds it instead of reusing the same slice.

Added a focused regression test for nested accelerator-missing exception detection.

## Hypothesis 3

Fresh east5 nextgen submissions are missing required Python modules from the runtime upload, so the canary may be failing before the training DAG even starts.

## Changes to make

- Inspect `lib/marin/src/marin/run/ray_run.py` working-dir exclusions.
- Ensure nextgen-required code under `experiments/domain_phase_mix/exploratory/` is included in the uploaded package.

## Future Work

- [ ] Trim the remaining exploratory upload exclusions to data/artifact paths only.

## Results

Confirmed. `ray_run.py` excluded the entire `experiments/domain_phase_mix/exploratory` tree by default, but nextgen imports `experiments.domain_phase_mix.exploratory.general_scaling_models`. Removed that bad default exclusion so nextgen code is present on the cluster.

## Hypothesis 4

The nextgen loop may be allowing `collect_new` to run before training finishes because the training-step dependency markers are stored in a tuple field that the executor does not recurse into.

## Changes to make

- Inspect `collect_dependencies_and_version()` and `instantiate_config()` in `lib/marin/src/marin/execution/executor.py`.
- Add tuple recursion so tuple-valued `InputName`s contribute real blocking dependencies and are instantiated into concrete paths.

## Future Work

- [ ] Audit other executor metadata fields that use tuple containers for dependencies or paths.

## Results

Confirmed. The executor ignored tuple-valued `InputName`s, so `CollectNewRunDataConfig.depends_on` silently dropped its dependency edges. Added tuple handling in both dependency collection and config instantiation, and added executor regression tests. After this fix, a fresh east5 baselines-only canary reached the desired state: both baseline checkpoint steps were `RUNNING` while `collect_new` and `fit` remained blocked.

## Hypothesis 5

The empty W&B dashboards are not a W&B bug. The new top-level Dolma 3 + Dolmino topology may still expand to hundreds of underlying tokenized partitions at training time, so the trainer could be spending tens of minutes loading cache ledgers from GCS before it reaches the first metric emission.

## Changes to make

- Inspect the live east5 training logs for `load_lm_dataset_cache` activity during the blank-W&B window.
- Check whether the "31 top-level domains" topology still expands to many underlying `DatasetComponent`s.
- Verify whether the cache metadata mismatch warnings are fatal, rebuilding caches, or merely noisy.

## Future Work

- [ ] Consider a true top-level cache representation if we want startup to scale with domains instead of raw partitions.
- [ ] Suppress one-sided `preprocessor_metadata=None` cache warnings to reduce noise during cache loads.
- [ ] Measure startup time to first metric as a function of component count before launching the 202-run swarm.

## Results

Confirmed. The two baseline jobs are not stuck in W&B or TPU launch. They are still in dataset startup inside `train_lm`, sequentially loading tokenized cache ledgers from GCS.

Evidence from the live east5 logs:

- W&B initialized successfully for both runs.
- After that, the logs show continuous `levanter.store.cache Loading cache from ...` lines for tokenized Dolma 3 and Dolmino partitions.
- The jobs progressed from Dolma 3 Common Crawl into Dolmino Common Crawl, `olmocr_pdfs_hq`, and `stack_edu_fim`, so they are moving rather than deadlocked.
- The warning `Metadata mismatch: {'type_changes': {'root.preprocessor_metadata': ...}}` is emitted on cache load, but `CacheLedger.load()` only warns and still returns the cache ledger; it does not trigger a rebuild.

The topology-level root cause is real: the supposed 31-domain experiment still expands to 441 underlying `DatasetComponent`s at training time because we preserve exact within-domain proportional allocation. That means each training run performs hundreds of remote cache-ledger loads before it reaches `parameter_count` logging or the first training step, which explains why W&B shows only the system panel for a long time.

Conclusion: this is a startup performance bug / architectural mismatch in the job setup, not a W&B logging failure and not yet evidence of a correctness failure in training.

## Hypothesis 6

Both east5 baseline runs later crashed while the merged-cache prep jobs were running, and the `.executor_status` files are stale.

## Changes to make

- Check the live checkpoint status markers for both baseline runs on `us-east5-a`.
- Inspect checkpoint artifacts (`eval_metrics.jsonl`, `checkpoints/step-*`) to verify whether training progressed after startup.
- Compare that against the cluster-wide Ray/OOM noise to separate baseline health from unrelated prep-job churn.

## Future Work

- [ ] Avoid overlapping high-memory cache-merge jobs with long-running TPU training on the same east5 cluster if the cluster remains memory-fragile.
- [ ] Add a lighter-weight per-run liveness view that does not require reading full Ray job logs.

## Results

Not confirmed. As of March 16, 2026:

- `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_baselines_fix4/baseline_pr-b9d895/.executor_status` is `RUNNING`.
- `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_baselines_fix4/baseline_un-4e9566/.executor_status` is `RUNNING`.
- Both runs have written `checkpoints/eval_metrics.jsonl`.
- Both runs have written `checkpoints/step-3424/`.
- The latest visible checkpoint metadata timestamps are around `2026-03-16T11:30Z` to `2026-03-16T11:31Z` (about `4:30 AM PDT`).

So the evidence says both baseline jobs are still alive and had progressed well into training. The cluster does show substantial Ray worker churn and OOM-related raylet warnings, but those are currently cluster-wide symptoms during the concurrent merged-cache prep jobs, not proof that either TPU training run has failed.

## Hypothesis 7

The hierarchical runtime-loading path is crashing during tagged evaluation setup because a grouped domain can have zero available child datasets in the `validation` split even though it has valid `train` children. The new lazy hierarchical loader currently raises when a non-train split ends up empty instead of skipping that grouped domain.

## Changes to make

- Inspect the hierarchical branch in `lib/levanter/src/levanter/data/text/datasets.py`.
- Keep train-split laziness intact, but make non-train splits resolve child availability early enough to:
  - skip empty grouped validation domains, and
  - compute a finite validation length from the surviving child datasets.
- Add regressions for:
  - fully empty hierarchical validation groups,
  - partially available hierarchical validation groups.

## Future Work

- [ ] If eval startup becomes noticeable, consider a lighter-weight cached split-availability map for hierarchical components so validation does not need to probe every child cache at evaluator construction time.
- [ ] Consider whether hierarchical non-train mixtures should eventually use a finite stop strategy instead of restart-plus-explicit-length.

## Results

Confirmed. The failed `us-east5-a` canary ended in:

- `ValueError: No datasets available for hierarchical component dolma3_cc/art_and_design_high`

The immediate cause was that all child validation caches under that grouped domain were absent on east5, so child-level validation components were individually skipped, but the parent `HierarchicalMixtureDatasetComponent` still returned a lazy dataset factory that only raised once the evaluator touched the first example.

Patched `lib/levanter/src/levanter/data/text/datasets.py` so non-train hierarchical splits:

- build child datasets eagerly enough to know whether anything exists,
- return `None` (skip the domain) when no validation/test children exist,
- compute the finite non-train length from the actually available child datasets.

Added two regressions in `lib/levanter/tests/test_text.py` covering empty and partial hierarchical validation availability.
